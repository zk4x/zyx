//! The step 3 gate: prove Rust is executing on a baby RISC-V.
//!
//! Increments a counter in L1 forever. The host advances the clock and watches it
//! climb. Deliberately the smallest program that cannot be faked by a stray write:
//! a counter that increases monotonically across several independent host reads
//! requires something to actually be running.

#![no_std]
#![no_main]

use tt_firmware::{l1_write32, publish};
use tt_isa::mailbox;

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    let mut count: u32 = 0;
    loop {
        count = count.wrapping_add(1);
        // SAFETY: a fixed, aligned mailbox location inside L1.
        unsafe { l1_write32(mailbox::HEARTBEAT, count) };
        publish();
    }
}
