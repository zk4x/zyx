//! Bring-up probe for RISCV NC: proves NC runs from its stub, and issues on
//! both NoCs.
//!
//! Keeps the heartbeat. Each time the host bumps `SEQ` in NC's mover mailbox
//! (`tt_isa::dm::nc`), writes `LEN` bytes of L1 at `L1_ADDR` to channel
//! `CHANNEL` at `DRAM_OFFSET` through NoC #1 (port 1, its own), then reads
//! them back through NoC #0 (CMFW's port, NoC #0's) to `L1_ADDR + LEN`, and
//! publishes `DONE = SEQ` with `ERROR`. Also publishes this tile's coordinate
//! as each NIU names it, so the host can check NC sees what B sees.
//!
//! After each request it reads the initiator's ten registers back and
//! publishes which differ from what it wrote (`REGS_KEPT_NOC1`,
//! `REGS_KEPT_NOC0`, a bit per register in `Command::registers`' order), and
//! `NOC_CTRL` as read: whether a later request may leave unchanged registers
//! unwritten (checklist 9.14). Nothing here relies on it.

#![no_std]
#![no_main]

use tt_firmware::{l1_read32, l1_write32, mailbox_word, noc, publish};
use tt_isa::dm::{self, nc};
use tt_isa::dram::Dram;
use tt_isa::mailbox::offset;
use tt_isa::noc::niu::{Command, Niu, TxnId};

/// Which registers read back differently from what was written, a bit per
/// register in `Command::registers`' order, for the NoC #1 and NoC #0
/// requests; and `NOC_CTRL` as read back from each.
const REGS_KEPT_NOC1: u64 = dm::nc::MAILBOX_BASE + 0xE8;
const REGS_KEPT_NOC0: u64 = dm::nc::MAILBOX_BASE + 0xEC;
const CTRL_NOC1: u64 = dm::nc::MAILBOX_BASE + 0xF0;
const CTRL_NOC0: u64 = dm::nc::MAILBOX_BASE + 0xF4;

/// After `cmd` has been issued through `niu` and has a virtual channel
/// (`CMD_CTRL` low bit clear), read its registers back: the mask of those
/// that differ from what `Command::registers` wrote, and `NOC_CTRL`.
fn kept(niu: Niu, cmd: &Command, me: (u8, u8)) -> (u32, u32) {
    use core::ptr::read_volatile;
    use tt_isa::noc::niu::initiator;
    let base = niu.base();
    // SAFETY: initiator 0's registers, MMIO in every Tensix tile.
    unsafe { while read_volatile((base + initiator::CMD_CTRL) as *const u32) & 1 != 0 {} }
    let Ok(regs) = cmd.registers(me, TXN, niu) else {
        return (u32::MAX, 0);
    };
    let mut mask = 0;
    let mut ctrl = 0;
    for (i, (off, want)) in regs.iter().enumerate() {
        // SAFETY: as above.
        let got = unsafe { read_volatile((base + off) as *const u32) };
        if *off == initiator::CTRL {
            ctrl = got;
        }
        if got != *want {
            mask |= 1 << i;
        }
    }
    (mask, ctrl)
}

const TXN: TxnId = match TxnId::new(2) {
    Some(t) => t,
    None => panic!(),
};

/// NC's copy of a B mover mailbox word.
const fn at(b_word: u64) -> u64 {
    b_word - dm::MAILBOX_BASE + nc::MAILBOX_BASE
}

fn rd(addr: u64) -> u32 {
    // SAFETY: aligned words of NC's mailbox and the L1 the host names.
    unsafe { l1_read32(addr) }
}

fn wr(addr: u64, v: u32) {
    // SAFETY: as `rd`.
    unsafe { l1_write32(addr, v) }
}

fn run() -> Result<(), u32> {
    let usable = rd(at(dm::USABLE));
    let ch = Dram::from_usable_mask(usable as u8)
        .channel(rd(at(dm::CHANNEL)) as u8)
        .ok_or(dm::error::RANGE)?;
    let len = rd(at(dm::LEN));
    let range = ch
        .range(rd(at(dm::DRAM_OFFSET)) as u64, len as u64)
        .ok_or(dm::error::RANGE)?;
    let l1 = rd(at(dm::L1_ADDR));
    let me0 = noc::me(Niu::Noc0);
    let me1 = noc::me(Niu::Noc1);
    let write = Command::WriteDram {
        from_local: l1,
        to: range,
        port: ch.port_for(Niu::Noc1, 1),
    };
    noc::issue(Niu::Noc1, &write, me1, TXN).map_err(|_| dm::error::ALIGNMENT)?;
    let (mask, ctrl) = kept(Niu::Noc1, &write, me1);
    wr(REGS_KEPT_NOC1, mask);
    wr(CTRL_NOC1, ctrl);
    noc::wait(TXN);
    let read = Command::ReadDram {
        from: range,
        port: ch.port_for(Niu::Noc0, 1),
        to_local: l1 + len,
    };
    noc::issue(Niu::Noc0, &read, me0, TXN).map_err(|_| dm::error::ALIGNMENT)?;
    let (mask, ctrl) = kept(Niu::Noc0, &read, me0);
    wr(REGS_KEPT_NOC0, mask);
    wr(CTRL_NOC0, ctrl);
    noc::wait(TXN);
    Ok(())
}

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    let pack = |m: (u8, u8)| m.0 as u32 | (m.1 as u32) << 6;
    wr(at(dm::MY_X), pack(noc::me(Niu::Noc0)));
    wr(at(dm::MY_Y), pack(noc::me(Niu::Noc1)));
    publish();
    let mut beat: u32 = 0;
    loop {
        beat = beat.wrapping_add(1);
        wr(mailbox_word(offset::HEARTBEAT), beat);
        publish();
        let seq = rd(at(dm::SEQ));
        if seq == 0 || seq == rd(at(dm::DONE)) {
            continue;
        }
        let r = run();
        wr(at(dm::ERROR), r.err().unwrap_or(dm::error::NONE));
        publish();
        wr(at(dm::DONE), seq);
        publish();
    }
}
