//! L1 Cache Tag Search Accelerator probe on RISCV B (`tt_isa::tag_search`).
//!
//! Runs the script the host staged at `tag_search::probe::SCRIPT`. For each step
//! it applies the step's masked `Config` read-modify-writes, fences, reads the
//! eight configuration words back, waits a few cycles, fences (which flushes the
//! L0 data cache, so the load below cannot be an L0 hit) and performs one `lw` at
//! the step's address. The loaded word and the readback go to the results.
//!
//! Before and after each phase the probe publishes a breadcrumb at
//! `probe::STAGE` (`(step << 8) | phase`, `probe::phase`), so the host can tell
//! which phase of which step never completed.
//!
//! Script word 1 selects blind mode (diagnostic, for the simulator): `Config` is
//! never loaded, each store carries a whole word merged in a local shadow, and no
//! readback is taken. It exists to tell a simulator that refuses the *loads* of
//! these registers from one that refuses the stores.
//!
//! Script word 2 non-zero skips the disarm before the first step and after the
//! last (a control that touches no `Config` word, and per-word simulator probes).
//!
//! The block is disarmed (all five trigger bits cleared) before anything else
//! runs, before any arming store of the script (the whole script is validated
//! first), on every early-exit path, and at the end, so no later load by this
//! core is intercepted; `Config` outlives a process on silicon. The probe writes
//! nowhere but `Config` words 212..=219, the result records and the breadcrumb.

#![no_std]
#![no_main]

use tt_firmware::{fail, finish, l1_read32, l1_write32, publish, spin};
use tt_isa::mailbox::panic_code;
use tt_isa::tag_search::{self as ts, probe, DISARM};

/// Masked read-modify-write of one `Config` word with plain `lw`/`sw`: RISC-V can
/// write `Config` only with whole-word stores.
///
/// # Safety
///
/// The Tensix backend must be out of soft reset, and `step.word` in 212..=219.
unsafe fn apply(step: ts::WriteStep) {
    let at = ts::config_address(step.word) as *mut u32;
    // SAFETY: a word of the generated `Config` table, per the caller.
    unsafe {
        let old = core::ptr::read_volatile(at);
        core::ptr::write_volatile(at, (old & !step.mask) | (step.value & step.mask));
    }
}

/// Blind-mode counterpart of [`apply`]: one whole-word store of the merge of
/// `step` into `shadow`, with no load of `Config`.
///
/// # Safety
///
/// As [`apply`].
unsafe fn apply_blind(shadow: &mut [u32; ts::WORD_COUNT], step: ts::WriteStep) {
    let i = (step.word - ts::FIRST_WORD) as usize;
    shadow[i] = (shadow[i] & !step.mask) | (step.value & step.mask);
    // SAFETY: a word of the generated `Config` table, per the caller.
    unsafe { core::ptr::write_volatile(ts::config_address(step.word) as *mut u32, shadow[i]) };
}

fn disarm(blind: bool, shadow: &mut [u32; ts::WORD_COUNT]) {
    for s in DISARM {
        // SAFETY: words 212, 218 and 219 of the generated table; the host
        // released the backend before starting this image.
        unsafe {
            if blind {
                apply_blind(shadow, s)
            } else {
                apply(s)
            }
        };
    }
    publish();
}

/// Publish a progress breadcrumb, fenced so it is visible before what follows.
fn stage(step: u32, phase: u32) {
    // SAFETY: a fixed aligned word of the data arena.
    unsafe { l1_write32(probe::STAGE, probe::stage(step, phase)) };
    publish();
}

/// Leave the block disarmed, then stop: no exit path may leave the accelerator
/// armed while this core goes on executing loads.
fn bail(blind: bool, shadow: &mut [u32; ts::WORD_COUNT], keep: bool, step: u32) -> ! {
    if !keep {
        disarm(blind, shadow);
    }
    stage(step, probe::phase::BAD_SCRIPT);
    fail(panic_code::EXPLICIT)
}

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    // SAFETY: aligned words of the staged header.
    let (steps, blind, keep) = unsafe {
        (
            l1_read32(probe::SCRIPT),
            l1_read32(probe::SCRIPT + 4) != 0,
            l1_read32(probe::SCRIPT + 8) != 0,
        )
    };
    let mut shadow = [0u32; ts::WORD_COUNT];
    stage(0, probe::phase::PROLOGUE);
    // Disarm first: `Config` outlives a process, and nothing here has run.
    if !keep {
        disarm(blind, &mut shadow);
    }
    if steps == 0 || steps > probe::MAX_STEPS {
        bail(blind, &mut shadow, keep, 0);
    }
    // Validate the whole script before the first arming store, so a bad step
    // cannot strand an armed block half-way through.
    for k in 0..steps {
        let step = probe::STEPS + u64::from(k) * probe::STEP_STRIDE;
        // SAFETY: aligned words of the staged step.
        let writes = unsafe { l1_read32(step + 4) };
        if writes as usize > ts::PROGRAM_STEPS {
            bail(blind, &mut shadow, keep, k);
        }
        for i in 0..u64::from(writes) {
            // SAFETY: bounded by `writes`, inside the step's 256 bytes.
            let word = unsafe { l1_read32(step + 8 + i * 12) };
            if !(u32::from(ts::FIRST_WORD)..u32::from(ts::FIRST_WORD) + ts::WORD_COUNT as u32)
                .contains(&word)
            {
                bail(blind, &mut shadow, keep, k);
            }
        }
    }
    for k in 0..steps {
        let step = probe::STEPS + u64::from(k) * probe::STEP_STRIDE;
        let result = probe::RESULTS + u64::from(k) * probe::RESULT_STRIDE;
        stage(k, probe::phase::STEP_BEGIN);
        // SAFETY: aligned words of the staged step.
        let (load_at, writes) = unsafe { (l1_read32(step), l1_read32(step + 4)) };
        for i in 0..u64::from(writes) {
            let at = step + 8 + i * 12;
            // SAFETY: bounded by `writes`, inside the step's 256 bytes.
            let (word, mask, value) =
                unsafe { (l1_read32(at), l1_read32(at + 4), l1_read32(at + 8)) };
            let write = ts::WriteStep {
                word: word as u16,
                mask,
                value,
            };
            // SAFETY: as in `disarm`, and `word` was range-checked above.
            unsafe {
                if blind {
                    apply_blind(&mut shadow, write)
                } else {
                    apply(write)
                }
            };
        }
        // The stores must retire before the load that depends on them.
        publish();
        stage(k, probe::phase::CONFIG_WRITTEN);
        for w in 0..ts::WORD_COUNT as u64 {
            if blind {
                break;
            }
            // SAFETY: Config words 212..=219; result words in the data arena.
            unsafe {
                let v = core::ptr::read_volatile(
                    ts::config_address(ts::FIRST_WORD + w as u16) as *const u32
                );
                l1_write32(result + 4 + w * 4, v);
            }
        }
        publish();
        stage(k, probe::phase::READBACK_DONE);
        for _ in 0..probe::SETTLE_NOPS {
            // SAFETY: ordinary RISC-V NOP, supported by the instruction gate.
            unsafe { core::arch::asm!("nop", options(nomem, nostack)) };
        }
        // `fence` flushes the L0 data cache: the load below cannot hit in it.
        publish();
        stage(k, probe::phase::SETTLED);
        stage(k, probe::phase::LOAD_ISSUED);
        // SAFETY: an aligned L1 word; the accelerator may answer it instead.
        let got = unsafe { l1_read32(u64::from(load_at)) };
        // SAFETY: the step's result record.
        unsafe { l1_write32(result, got) };
        publish();
        stage(k, probe::phase::LOAD_RETURNED);
    }
    if !keep {
        disarm(blind, &mut shadow);
    }
    finish(steps);
    stage(steps, probe::phase::DISARMED);
    spin()
}
