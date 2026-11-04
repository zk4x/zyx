//! The math role of the three-thread datapath: the generic runner on
//! T1 / Tensix thread 1, reporting through role mailbox 1.
//!
//! Linked at T1's default reset PC (see `build.rs`), so the three role
//! images can sit in one tile's L1 at once. See `tt_firmware::corpus`.

#![no_std]
#![no_main]

use tt_firmware::corpus;
use tt_isa::tensix::{RiscvT1, Thread1};

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    corpus::run_role::<RiscvT1, Thread1>()
}
