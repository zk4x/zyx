//! The unpack role of the three-thread datapath: the generic runner on
//! T0 / Tensix thread 0, reporting through role mailbox 0.
//!
//! Linked at T0's default reset PC (see `build.rs`), so the three role
//! images can sit in one tile's L1 at once. See `tt_firmware::corpus`.

#![no_std]
#![no_main]

use tt_firmware::corpus;
use tt_isa::tensix::{RiscvT0, Thread0};

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    corpus::run_role::<RiscvT0, Thread0>()
}
