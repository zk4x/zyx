//! The Ethernet-tile image, on RISCV E1: a data mover between chips.
//!
//! Protocol and layout are `tt_isa::eth::mover`, which the host shares. This
//! image is both ends of a link: it sends what the host asks it to, and forwards
//! what its partner sends it. It also keeps the heartbeat, so the host can tell a
//! stuck mover from a dead core.

#![no_std]
#![no_main]

use tt_firmware::{cycles, l1_read32, l1_write32, mailbox_word, noc, publish};
use tt_isa::eth::{self, mover, mover::event};
use tt_isa::mailbox::offset;
use tt_isa::noc::niu::{Command, Endpoint, Niu, TxnId, MAX_REQUEST_BYTES};

const TXN: TxnId = match TxnId::new(1) {
    Some(t) => t,
    None => panic!(),
};

fn rd(addr: u64) -> u32 {
    // SAFETY: every address used is a fixed, aligned word in customer L1 or an
    // Ethernet MMIO register.
    unsafe { l1_read32(addr) }
}

fn wr(addr: u64, v: u32) {
    // SAFETY: as `rd`.
    unsafe { l1_write32(addr, v) }
}

/// Wait for TXQ `DATA_QUEUE` to be ready for a command.
fn txq_idle() {
    let status = eth::txq_base(eth::DATA_QUEUE) + eth::txq::STATUS;
    while rd(status) & eth::txq::STATUS_CMD_ONGOING != 0 {}
}

/// TT-link-write `len` bytes of local `src` to the partner's `dst`, in chunks.
fn tt_link(src: u64, dst: u64, len: u32) {
    let q = eth::txq_base(eth::DATA_QUEUE);
    let cmd = match eth::TxCommand::L1Write.word(eth::DATA_QUEUE) {
        Some(c) => c,
        None => unreachable!(),
    };
    let mut done = 0u32;
    while done < len {
        let n = (len - done).min(mover::TT_LINK_CHUNK);
        txq_idle();
        wr(
            q + eth::txq::TRANSFER_START_ADDR,
            (src + done as u64) as u32,
        );
        wr(q + eth::txq::TRANSFER_SIZE_BYTES, n);
        wr(q + eth::txq::DEST_ADDR, (dst + done as u64) as u32);
        wr(q + eth::txq::CMD, cmd);
        // EthernetTxRx.md: read CMD back before STATUS.
        let _ = rd(q + eth::txq::CMD);
        done += n;
    }
    txq_idle();
}

/// NoC-copy `len` bytes between a Tensix tile and local L1, in 16 KiB requests.
fn noc_copy(me: (u8, u8), tile: (u8, u8, u32), local: u64, len: u32, into_local: bool) {
    let mut done = 0u32;
    while done < len {
        let n = (len - done).min(MAX_REQUEST_BYTES);
        let remote = Endpoint {
            x: tile.0,
            y: tile.1,
            addr: tile.2 + done,
        };
        let l = (local + done as u64) as u32;
        let cmd = if into_local {
            Command::Read {
                from: remote,
                to_local: l,
                len: n,
            }
        } else {
            Command::Write {
                from_local: l,
                to: remote,
                len: n,
            }
        };
        if noc::issue(Niu::Noc0, &cmd, me, TXN).is_err() {
            fail(mover::error::ALIGNMENT);
        }
        done += n;
    }
    noc::wait(TXN);
}

/// Record `event` for `seq` in the trace ring, if the host asked
/// (`mover::TRACE`) and there is room. Stamped by this core's own counter.
fn trace(event: u32, seq: u32) {
    if rd(mover::TRACE) == 0 {
        return;
    }
    let n = rd(mover::TRACE_COUNT);
    if n >= mover::TRACE_EVENTS {
        return;
    }
    let t = cycles();
    let at = mover::TRACE_RING + n as u64 * mover::TRACE_RECORD_BYTES;
    wr(at, event);
    wr(at + 4, seq);
    wr(at + 8, t);
    wr(at + 12, 0);
    wr(mover::TRACE_COUNT, n + 1);
}

fn fail(code: u32) -> ! {
    wr(mover::ERROR, code);
    publish();
    tt_firmware::fail(code)
}

fn send(me: (u8, u8), seq: u32) {
    trace(event::SEND_PICKUP, seq);
    let len = rd(mover::SEND_LEN);
    if len == 0 || len % 16 != 0 || len > mover::MAX_LEN {
        fail(mover::error::LENGTH);
    }
    let (sx, sy, sa) = (
        rd(mover::SEND_SRC_X),
        rd(mover::SEND_SRC_Y),
        rd(mover::SEND_SRC_ADDR),
    );
    if sx != mover::NO_TILE {
        if sa % 16 != 0 {
            fail(mover::error::ALIGNMENT);
        }
        noc_copy(me, (sx as u8, sy as u8, sa), mover::TX_STAGE, len, true);
        trace(event::NOC_IN_DONE, seq);
    }
    tt_link(mover::TX_STAGE, mover::RX_LAND, len);
    trace(event::DATA_SENT, seq);
    let record = [
        seq,
        len,
        rd(mover::SEND_DST_X),
        rd(mover::SEND_DST_Y),
        rd(mover::SEND_DST_ADDR),
        0,
        0,
        seq,
    ];
    for (i, w) in record.iter().enumerate() {
        wr(mover::RECORD_STAGE + i as u64 * 4, *w);
    }
    publish();
    tt_link(
        mover::RECORD_STAGE,
        mover::INBOX,
        mover::RECORD_BYTES as u32,
    );
    trace(event::RECORD_SENT, seq);
    wr(mover::SENT, seq);
    publish();
}

fn receive(me: (u8, u8), seq: u32) {
    trace(event::RECORD_SEEN, seq);
    let len = rd(mover::INBOX + 4);
    if len == 0 || len % 16 != 0 || len > mover::MAX_LEN {
        fail(mover::error::LENGTH);
    }
    // Every data packet was accepted before the record (TT-link is in order);
    // once none of the RX queue's writes are outstanding, all of them are in
    // L1. Bounded, so a queue that never drains is reported, not hung on.
    if rd(mover::LANDING_WAIT) != 0 {
        let outstanding = eth::rxq_base(eth::DATA_QUEUE) + eth::rxq::OUTSTANDING_WR_CNT;
        let mut tries = 0u32;
        while rd(outstanding) != 0 {
            tries += 1;
            if tries > 10_000_000 {
                fail(mover::error::LANDING);
            }
        }
    }
    trace(event::LANDED, seq);
    let (dx, dy, da) = (
        rd(mover::INBOX + 8),
        rd(mover::INBOX + 12),
        rd(mover::INBOX + 16),
    );
    if dx != mover::NO_TILE {
        if da % 16 != 0 {
            fail(mover::error::ALIGNMENT);
        }
        noc_copy(me, (dx as u8, dy as u8, da), mover::RX_LAND, len, false);
        trace(event::NOC_OUT_DONE, seq);
    }
    wr(mover::RECEIVED, seq);
    for i in 0..4u64 {
        wr(mover::ACK_STAGE + i * 4, if i == 0 { seq } else { 0 });
    }
    publish();
    tt_link(mover::ACK_STAGE, mover::ACK, 16);
    trace(event::ACK_SENT, seq);
}

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    let me = (rd(mover::MY_X) as u8, rd(mover::MY_Y) as u8);
    let (mut sent, mut received) = (rd(mover::SENT), rd(mover::RECEIVED));
    let mut beat: u32 = 0;
    loop {
        beat = beat.wrapping_add(1);
        wr(mailbox_word(offset::HEARTBEAT), beat);
        // Nothing written by the NoC or the RX queue invalidates the L0 data
        // cache (MemoryOrdering.md:59), so every poll goes through a fence.
        publish();
        let seq = rd(mover::SEND_SEQ);
        if seq != 0 && seq != sent {
            send(me, seq);
            sent = seq;
        }
        let r = rd(mover::INBOX);
        if r != 0 && r != received && rd(mover::INBOX + 28) == r {
            receive(me, r);
            received = r;
        }
        let ack = rd(mover::ACK);
        if ack != rd(mover::ACKED) {
            trace(event::ACK_SEEN, ack);
            wr(mover::ACKED, ack);
        }
    }
}
