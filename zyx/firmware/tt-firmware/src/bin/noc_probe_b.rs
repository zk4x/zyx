//! NoC atomic, multicast and completion probe on RISCV B (`tt_isa::noc::probe`).
//!
//! Replays the requests the host encoded with the typed encoders
//! (`AtomicRequest::registers`, `MulticastWrite::registers`): this image holds no
//! encoding of its own, so what it sends is exactly what the host gates model.
//! Per request it
//!
//! 1. requires the transaction ID's `NIU_MST_REQS_OUTSTANDING_ID` to read zero
//!    (otherwise reports `NOT_QUIESCENT` and sends nothing),
//! 2. waits, bounded, for the initiator's `CMD_CTRL` low bit to clear,
//! 3. writes the registers in `probe::REGISTERS` order (the eleventh,
//!    `NOC_BRCST_EXCLUDE`, only for a multicast), writes `CMD_CTRL` and reads it
//!    back (`Counters.md:42-43`: otherwise a later counter read can overtake it),
//! 4. waits, bounded, for completion: a unicast when its ID's counter is zero; a
//!    multicast when `NIU_MST_WR_ACK_RECEIVED` has advanced by the expected
//!    recipient count -- never by the ID's counter, which a multicast leaves
//!    meaningless (`Interrupts.md:19`) -- and then clears the ID's counter
//!    (`NIU_BASE + 0x60`),
//! 5. records the counters it saw.
//!
//! With `flag::PARTIAL` step 3 writes only the registers `word::MASK` selects and
//! the ten initiator words are read back into `probe::SNAPSHOTS` after completion:
//! the persistence probe of `step118_mover_fast_path`.
//!
//! With `flag::RTZ` it records `NIU_TRANS_COUNT_RTZ_SOURCE` before and after
//! (with `RTZ_CLEAR`, after first clearing the ID's bit): the polling form of the
//! completion interrupt. It never sets `NIU_TRANS_COUNT_RTZ_CFG` or the PIC's
//! enable bits and never reads `NIU_TRANS_COUNT_RTZ_NUM` (a read has a side
//! effect), so no interrupt can be raised.
//!
//! Before and after each phase a breadcrumb is published at `probe::STAGE`. The
//! image writes nowhere but the NIU's initiator 0 and counters, the result
//! records and the breadcrumb. Every wait is bounded; a timeout reports and moves
//! on to the next request, leaving the NIU as it is.

#![no_std]
#![no_main]

use core::ptr::{read_volatile, write_volatile};

use tt_firmware::{finish, l1_read32, l1_write32, publish, spin};
use tt_isa::noc::niu::{self, initiator, Niu, TxnId};
use tt_isa::noc::probe::{self, flag, kind, phase, result, status, word};

fn reg(niu: Niu, offset: u64) -> *mut u32 {
    (niu.base() + offset) as *mut u32
}

fn stage(request: u32, ph: u32) {
    // SAFETY: a fixed aligned word of the probe's region.
    unsafe { l1_write32(probe::STAGE, probe::stage(request, ph)) };
    publish();
}

/// `NIU_MST_REQS_OUTSTANDING_ID(txn)`, its low 8 bits.
fn outstanding(niu: Niu, txn: TxnId) -> u32 {
    // SAFETY: a read-only NIU counter.
    unsafe { read_volatile(reg(niu, niu::reqs_outstanding(txn))) & 0xFF }
}

fn ack_received(niu: Niu) -> u32 {
    // SAFETY: a read-only NIU counter.
    unsafe { read_volatile(reg(niu, niu::WR_ACK_RECEIVED)) }
}

fn put(base: u64, offset: u64, value: u32) {
    // SAFETY: a result record word, aligned, inside the probe's region.
    unsafe { l1_write32(base + offset, value) };
}

fn get(base: u64, offset: u64) -> u32 {
    // SAFETY: a staged record word, aligned, inside the probe's region.
    unsafe { l1_read32(base + offset) }
}

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    // SAFETY: aligned words of the staged header.
    let (requests, budget) = unsafe { (l1_read32(probe::SCRIPT), l1_read32(probe::SCRIPT + 4)) };
    stage(0, phase::BEGIN);
    if requests == 0 || requests > probe::MAX_REQUESTS || budget == 0 {
        for k in 0..probe::MAX_REQUESTS {
            put(
                probe::RESULTS + u64::from(k) * probe::RESULT_STRIDE,
                result::STATUS,
                status::BAD_RECORD,
            );
        }
        finish(0xBAD);
        spin()
    }
    for k in 0..requests {
        let rec = probe::REQUESTS + u64::from(k) * probe::REQUEST_STRIDE;
        let res = probe::RESULTS + u64::from(k) * probe::RESULT_STRIDE;
        stage(k, phase::BEGIN);
        for w in [
            result::ACK_BEFORE,
            result::ACK_AFTER,
            result::OUTSTANDING_AFTER,
            result::OUTSTANDING_BEFORE,
            result::RTZ_BEFORE,
            result::RTZ_AFTER,
            result::POLLS,
        ] {
            put(res, w, 0);
        }
        let k_kind = get(rec, word::KIND);
        let flags = get(rec, word::FLAGS);
        let expected = get(rec, word::EXPECTED_ACKS);
        let Some(txn) = TxnId::new(get(rec, word::TXN) as u8) else {
            put(res, result::STATUS, status::BAD_RECORD);
            continue;
        };
        let niu = if flags & flag::NOC1 != 0 {
            Niu::Noc1
        } else {
            Niu::Noc0
        };
        // The request initiator: 0, or 1 (`NIU_BASE + 0x800`) for the mover
        // fast path's own (`flag::INITIATOR_1`).
        let ini = if flags & flag::INITIATOR_1 != 0 {
            initiator::STRIDE
        } else {
            0
        };
        let multicast = k_kind == kind::MULTICAST;
        if k_kind > kind::MULTICAST {
            put(res, result::STATUS, status::BAD_RECORD);
            continue;
        }

        // The initiator must be free before any field of it is touched.
        let mut free = false;
        for _ in 0..budget {
            // SAFETY: the initiator's `CMD_CTRL`, MMIO in every Tensix tile.
            if unsafe { read_volatile(reg(niu, ini + initiator::CMD_CTRL)) } & 1 == 0 {
                free = true;
                break;
            }
        }
        if !free {
            put(res, result::STATUS, status::INITIATOR_BUSY);
            continue;
        }
        stage(k, phase::INITIATOR_FREE);

        let before = outstanding(niu, txn);
        put(res, result::OUTSTANDING_BEFORE, before);
        if before != 0 {
            put(res, result::STATUS, status::NOT_QUIESCENT);
            continue;
        }
        if flags & flag::RTZ != 0 {
            // SAFETY: NIU configuration registers; the write clears one bit of
            // `SOURCE` and has no other effect, the read has none.
            unsafe {
                if flags & flag::RTZ_CLEAR != 0 {
                    write_volatile(reg(niu, niu::TRANS_COUNT_RTZ_CLR), 1 << txn.index());
                }
                let src = read_volatile(reg(niu, niu::TRANS_COUNT_RTZ_SOURCE));
                put(res, result::RTZ_BEFORE, src);
            }
        }
        let acks_before = ack_received(niu);
        put(res, result::ACK_BEFORE, acks_before);

        let count = if multicast { 11 } else { 10 };
        let partial = flags & flag::PARTIAL != 0;
        let mask = if partial { get(rec, word::MASK) } else { u32::MAX };
        for (i, &offset) in probe::REGISTERS.iter().take(count).enumerate() {
            if mask & (1 << i) == 0 {
                continue;
            }
            let v = get(rec, word::REGISTERS + 4 * i as u64);
            // SAFETY: initiator 0's registers, written while it reads free.
            unsafe { write_volatile(reg(niu, ini + offset), v) };
        }
        stage(k, phase::REGISTERS_WRITTEN);
        // SAFETY: the same initiator; the read-back orders later counter reads.
        unsafe {
            write_volatile(reg(niu, ini + initiator::CMD_CTRL), 1);
            let _ = read_volatile(reg(niu, ini + initiator::CMD_CTRL));
        }
        stage(k, phase::ISSUED);

        let mut polls = 0u32;
        let mut done = false;
        let mut excess = false;
        while polls < budget {
            polls += 1;
            if multicast {
                let got = ack_received(niu).wrapping_sub(acks_before);
                if got == expected {
                    done = true;
                    break;
                }
                if got > expected {
                    done = true;
                    excess = true;
                    break;
                }
            } else if outstanding(niu, txn) == 0 {
                done = true;
                break;
            }
        }
        put(res, result::POLLS, polls);
        put(res, result::ACK_AFTER, ack_received(niu));
        put(res, result::OUTSTANDING_AFTER, outstanding(niu, txn));
        if flags & flag::RTZ != 0 {
            // SAFETY: a read-only status register.
            let src = unsafe { read_volatile(reg(niu, niu::TRANS_COUNT_RTZ_SOURCE)) };
            put(res, result::RTZ_AFTER, src);
        }
        if done && multicast && flags & flag::NO_CLEAR == 0 {
            // A broadcast leaves its ID's counter at `1 - recipients`.
            // SAFETY: write-only NIU register, one ID's bit.
            unsafe {
                write_volatile(
                    reg(niu, niu::CLEAR_OUTSTANDING),
                    tt_isa::noc::multicast::clear_outstanding(txn),
                );
                let _ = read_volatile(reg(niu, ini + initiator::CMD_CTRL));
            }
        }
        if partial && done {
            // The initiator is free again once `CMD_CTRL`'s low bit has cleared;
            // read the words back to see which kept their value.
            let mut free = false;
            for _ in 0..budget {
                // SAFETY: the initiator's `CMD_CTRL`, MMIO in every Tensix tile.
                if unsafe { read_volatile(reg(niu, ini + initiator::CMD_CTRL)) } & 1 == 0 {
                    free = true;
                    break;
                }
            }
            if free {
                let snap = probe::SNAPSHOTS + u64::from(k) * probe::SNAPSHOT_STRIDE;
                for (i, &offset) in probe::REGISTERS.iter().take(10).enumerate() {
                    // SAFETY: initiator 0's registers, read while it reads free.
                    let v = unsafe { read_volatile(reg(niu, ini + offset)) };
                    put(snap, 4 * i as u64, v);
                }
            }
        }
        put(
            res,
            result::STATUS,
            if !done {
                status::TIMEOUT
            } else if excess {
                status::EXCESS_ACKS
            } else {
                status::OK
            },
        );
        publish();
        stage(k, phase::COMPLETE);
    }
    stage(requests, phase::DONE);
    finish(requests);
    spin()
}
