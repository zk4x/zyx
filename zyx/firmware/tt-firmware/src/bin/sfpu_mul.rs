//! The step 4 gate: drive the Tensix coprocessor and read the result back.
//!
//! Computes 3.0 × 2.0 on the Vector Unit and publishes the FP32 bit pattern to the
//! L1 mailbox, where the host can read it. Expected: `0x40C0_0000`.
//!
//! This core has to exist in the middle of the chain. `INSTRN_BUF_BASE` and `Dst`
//! are both unmapped to the NoC, so the host can neither push the instructions nor
//! read the answer.

#![no_std]
#![no_main]

use tt_firmware::cfg::write_config_field;
use tt_firmware::tensix::{push, read_dst32, wait_for_coprocessor};
use tt_firmware::{fail, finish, l1_read32, publish, spin};
use tt_isa::backend;
use tt_isa::cfg::ConfigBank;
use tt_isa::mailbox::{self, panic_code};
use tt_isa::sfpu::{self, dst32_address, store_format};
use tt_isa::tensix::{RiscvT1, TensixThread, Thread1};

/// Which core this image is loaded onto, and which Tensix thread it therefore
/// drives.
///
/// These are the *only* place that fact is written down on the device side. Core
/// identity cannot be probed -- `mhartid` reads zero on every core -- so it is a
/// build-time fact, and making it a type means the push path cannot reach a buffer
/// that would hang this core (`tt_isa::tensix::PushesTo`).
///
/// T1 because it is the only core whose RISC-V view of `Dst` ttsim models
/// (`ttsim-divergence.md`, row 12). The host must agree: see `step4_tensix.rs`.
type Riscv = RiscvT1;
type Thread = Thread1;

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    // Operands come from the host, through the L1 mailbox, rather than being
    // baked in. That makes the gate a test of arithmetic rather than of one
    // constant: the host can ask for several products and check each, which a
    // firmware returning a fixed value could not satisfy.
    //
    // It also means the instruction encoder runs on the device. `SFPLOADI`
    // carries its operand as a 16-bit immediate, so a host-supplied value has to
    // be encoded into the instruction word here, at run time.
    //
    // SAFETY: fixed, aligned mailbox locations written by the host before this
    // core left reset.
    let (a, b) = unsafe { (l1_read32(mailbox::OPERAND_A), l1_read32(mailbox::OPERAND_B)) };
    let thread = unsafe { l1_read32(mailbox::THREAD_INDEX) };
    let fmt = unsafe { l1_read32(mailbox::DST_ACCESS_FMT) };
    // The core this image runs on is a build-time fact (see `Riscv`/`Thread`
    // above), so the mailbox word is a cross-check, not the source of truth: if the
    // host started this image on a different core than it thinks, the Dst mapping
    // written below would be another thread's and the failure would surface as a
    // wrong result rather than as an error.
    if thread != Thread::INDEX {
        fail(panic_code::EXPLICIT);
    }

    // Set the shape of the Dst mapping deliberately rather than inheriting
    // whatever reset left behind. `fmt = 0` is `float Dst32b[512][16]`, the shape
    // a 32-bit `lw` below is valid for; the reset default happens to be 0 too, but
    // depending on that is a bug waiting for the first kernel that changes it.
    //
    // The field is indexed per section, and `Thread::DST_ACCESS_FMT` carries the
    // association, so the thread this image drives picks it rather than a `match`
    // whose default arm would silently configure a different thread. Thread 1 is
    // the only one whose Dst mapping ttsim models.
    //
    // SAFETY: the host released the Tensix backend before releasing this core, and
    // nothing has been pushed yet, so no Tensix instruction is in flight.
    unsafe {
        // Bank 0, not `cfg::active_bank(thread)`. The live bank is selected by
        // `ThreadConfig[thread].CFG_STATE_ID_StateID`, but ttsim maps the whole
        // configuration aperture to a flat `Config` array with the bank hardcoded
        // to zero and no `ThreadConfig` region at all, so reading the state ID
        // there is fatal. Nothing in this firmware changes the state ID, and its
        // reset value is 0, so bank 0 is correct either way -- but on silicon,
        // where a kernel may have switched banks, `active_bank` is the right call.
        let bank = ConfigBank::Bank0;
        let field = Thread::DST_ACCESS_FMT;
        if !field.fits(fmt) {
            fail(panic_code::EXPLICIT);
        }
        write_config_field(field, bank, fmt);
    }
    // Configuration is written by RISC-V but read by the coprocessor. Auto TTSync
    // covers the push-then-load direction, not this one, so drain the store queue
    // before pushing anything that depends on the new value.
    publish();

    // Build the program first. Every encoding is checked, and a bad one fails
    // loudly here rather than being pushed and silently misexecuting.
    let Ok(load_a) = sfpu::load_f32(0, a) else {
        fail(panic_code::EXPLICIT)
    };
    let Ok(load_b) = sfpu::load_f32(1, b) else {
        fail(panic_code::EXPLICIT)
    };
    let Ok(multiply) = sfpu::mul(0, 1, 2) else {
        fail(panic_code::EXPLICIT)
    };
    let Ok(store) = sfpu::store(2, store_format::FP32, 0, 0) else {
        fail(panic_code::EXPLICIT)
    };
    let Ok(sfpu_barrier) = backend::wait_for_sfpu(backend::Before::EVERYTHING) else {
        fail(panic_code::EXPLICIT)
    };

    // SAFETY: `Riscv` may push to `Thread` -- the type system checked it; the host
    // released the Tensix
    // backend from soft reset before releasing this core.
    unsafe {
        push::<Riscv, Thread>(load_a[0]);
        push::<Riscv, Thread>(load_a[1]);
        push::<Riscv, Thread>(load_b[0]);
        push::<Riscv, Thread>(load_b[1]);
        push::<Riscv, Thread>(multiply);
        // No SFPNOP between the multiply and the store: on Blackhole the two-cycle
        // SFPMAD latency is covered by automatic stalling, and SFPSTORE is not one
        // of the documented cases that stalling fails to detect. See
        // `Instruction::stalls_automatically_after_mad`.
        push::<Riscv, Thread>(store);
        // Hold everything after this until the SFPU has finished, so the
        // coprocessor does not report done with the store still in flight.
        push::<Riscv, Thread>(sfpu_barrier);
    }

    // The pushes above have only reached a FIFO. Wait for them to retire before
    // looking at Dst.
    wait_for_coprocessor();

    // Lane 0 of the store lands at Dst[0][0].
    // SAFETY: the coprocessor has retired the store, and fmt is a 32-bit shape.
    let result = unsafe { read_dst32(dst32_address(0, 0)) };

    finish(result);
    spin()
}
