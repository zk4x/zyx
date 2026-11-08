//! The pack role of the three-thread datapath: the generic runner on
//! T2 / Tensix thread 2, reporting through role mailbox 2.
//!
//! Linked at T2's default reset PC (see `build.rs`), so the three role
//! images can sit in one tile's L1 at once. See `tt_firmware::corpus`.

#![no_std]
#![no_main]

use tt_firmware::corpus;
use tt_isa::tensix::{RiscvT2, Thread2};

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    corpus::run_role::<RiscvT2, Thread2>()
}
