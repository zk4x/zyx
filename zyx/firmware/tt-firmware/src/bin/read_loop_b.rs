//! A read loop on RISCV B with none of the mover's interpreter: the question of
//! `silicon_probe_read_loop`, "how much of a 4 KiB read entry is decode and
//! dispatch?" (`tt_isa::noc::read_loop`).
//!
//! The host encodes the first request (`DramMove::words`) and this image issues
//! `count` of them through the same `noc` calls the mover issues through, each
//! `len` bytes on from the last, so the only difference from the mover's read path is that
//! nothing is fetched, decoded, validated or dispatched per request. Per repetition it
//! reports the cycles from the first request to the last one landing, by the
//! tile's wall clock (the clock the mover times its throttle waits with). Between
//! repetitions, outside the timing, it checksums the destination window and refills
//! it with poison, so the host can check that every repetition landed its own data.
//!
//! Modes (`read_loop::mode`): the mover's default issue (every initiator register),
//! the X6 fast path, and a replay of requests the host encoded into a table (the
//! thinnest interpreter there is), and the mover's own decode and validation
//! of shipped list entries without its loop (`mode::DECODED`).

#![no_std]
#![no_main]

use tt_firmware::{disable_gathering, finish, l1_read32, l1_write32, noc, publish, spin};
use tt_isa::dm::Entry;
use tt_isa::noc::niu::{DramMove, Niu, TxnId};
use tt_isa::noc::read_loop::{self as rl, mode};

fn rd(at: u64) -> u32 {
    // SAFETY: a fixed aligned word of the control block, table or request.
    unsafe { l1_read32(at) }
}

fn wr(at: u64, v: u32) {
    // SAFETY: a fixed aligned word of the control block.
    unsafe { l1_write32(at, v) }
}

/// The tile counter's low word.
fn wall_clock() -> u32 {
    rd(tt_isa::tensix::timestamper::WALL_CLOCK_L)
}

/// The wrapping sum of the window's words.
#[inline(never)]
fn window_sum(start: u32, end: u32) -> u32 {
    let mut sum = 0u32;
    let mut at = start as u64;
    while at < end as u64 {
        sum = sum.wrapping_add(rd(at));
        at += 4;
    }
    sum
}

#[inline(never)]
fn poison(start: u32, end: u32) {
    let mut at = start as u64;
    while at < end as u64 {
        wr(at, rl::POISON);
        at += 4;
    }
    publish();
}

/// One repetition: `count` requests, then wait for every one to land.
#[inline(never)]
fn repetition(m: u32, count: u32, txn: TxnId) {
    let [mut targ, targ_hi, mut ret, ret_hi, tag, ctrl, len] =
        core::array::from_fn(|i| rd(rl::REQUEST + 4 * i as u64));
    let (ret0, ret_end) = (ret, rd(rl::RET_END));
    match m {
        mode::FAST => {
            for _ in 0..count {
                noc::room_for_read(txn);
                noc::issue_read(targ, targ_hi, ret, ret_hi, tag, ctrl, len, txn);
                targ += len;
                ret += len;
                if ret >= ret_end {
                    ret = ret0;
                }
            }
        }
        mode::DECODED => {
            let usable = rd(rl::USABLE);
            let me = (rd(rl::ME_X) as u8, rd(rl::ME_Y) as u8);
            for i in 0..count as u64 {
                let at = rl::TABLE + i * rl::TABLE_STRIDE;
                let w: [u32; 8] = core::array::from_fn(|k| rd(at + 4 * k as u64));
                let Ok(Entry::Move { descriptor: d, .. }) = Entry::decode(usable, w) else {
                    finish(0xBAD);
                    spin()
                };
                let port = d.range.channel().port_for(Niu::Noc0, d.port);
                let Ok(mv) = DramMove::new(d.range, port, d.l1, false, me, txn, Niu::Noc0) else {
                    finish(0xBAD);
                    spin()
                };
                for r in mv.words() {
                    noc::issue_dram_on::<false>(
                        r.targ, r.targ_hi, r.ret, r.ret_hi, r.tag, r.ctrl, r.len, txn,
                    );
                }
            }
        }
        mode::TABLE => {
            for i in 0..count as u64 {
                let at = rl::TABLE + i * rl::TABLE_STRIDE;
                noc::issue_dram_on::<false>(
                    rd(at),
                    rd(at + 4),
                    rd(at + 8),
                    rd(at + 12),
                    rd(at + 16),
                    rd(at + 20),
                    rd(at + 24),
                    txn,
                );
            }
        }
        _ => {
            for _ in 0..count {
                noc::issue_dram_on::<false>(targ, targ_hi, ret, ret_hi, tag, ctrl, len, txn);
                targ += len;
                ret += len;
                if ret >= ret_end {
                    ret = ret0;
                }
            }
        }
    }
    noc::wait(txn);
}

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    disable_gathering();
    loop {
        publish();
        let seq = rd(rl::SEQ);
        if seq == 0 || seq == rd(rl::DONE) {
            continue;
        }
        let (m, count, reps) = (rd(rl::MODE), rd(rl::COUNT), rd(rl::REPS));
        let Some(txn) = TxnId::new(rd(rl::TXN) as u8) else {
            finish(0xBAD);
            spin()
        };
        if reps == 0 || reps > rl::MAX_REPS || count == 0 || count > rl::TABLE_MAX {
            finish(0xBAD);
            spin()
        }
        noc::set_cap(txn, rd(rl::CAP));
        noc::set_read_path(m == mode::FAST);
        let (start, end) = (rd(rl::REQUEST + 8), rd(rl::RET_END));
        // One untimed repetition: the instruction cache, the NIU and the DRAM
        // pages warm, as the mover's benches' warm-up list does.
        repetition(m, count, txn);
        poison(start, end);
        for r in 0..reps {
            let t0 = wall_clock();
            repetition(m, count, txn);
            let dt = wall_clock().wrapping_sub(t0);
            wr(rl::CYCLES + 4 * r as u64, dt);
            // Outside the timing: what landed, then clean for the next one (the
            // last is left for the host to compare byte for byte).
            wr(rl::CHECKS + 4 * r as u64, window_sum(start, end));
            if r + 1 < reps {
                poison(start, end);
            }
        }
        wr(rl::STALLS, noc::stalls().count);
        publish();
        wr(rl::DONE, seq);
        publish();
    }
}
