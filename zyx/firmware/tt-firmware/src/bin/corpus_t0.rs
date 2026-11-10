//! The generic Tensix program runner on T0 / thread 0: the silicon image.
//!
//! See `corpus.rs` for what it does, and `tt_firmware::corpus` for why the
//! thread differs by target.
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
use tt_isa::tensix::{RiscvT0, Thread0};

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    corpus::run::<RiscvT0, Thread0>()
}
