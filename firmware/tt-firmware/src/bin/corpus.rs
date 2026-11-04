//! A generic Tensix program runner: push what the host staged, dump what it asks
//! for.
//!
//! The step 4 firmware computes one thing. This computes whatever the host encoded,
//! which is what turns "does this instruction execute correctly" from a firmware
//! change into a test case. The host has the generated instruction table; the device
//! side only has to push words and copy `Dst` back.
//!
//! Both halves of that are forced by the hardware, not chosen. `INSTRN_BUF_BASE` is
//! unmapped to the NoC, so the host cannot push; `Dst` is unmapped too, so the host
//! cannot read a result. A baby RISC-V has to sit in the middle, and this is the
//! smallest thing that can.

#![no_std]
#![no_main]

use tt_firmware::corpus;
use tt_isa::tensix::{RiscvT1, Thread1};

/// T1 because ttsim models the RISC-V view of `Dst` only for it (divergence row
/// 12). `corpus_t0` is the silicon image; see `tt_firmware::corpus`.
#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    corpus::run::<RiscvT1, Thread1>()
}
