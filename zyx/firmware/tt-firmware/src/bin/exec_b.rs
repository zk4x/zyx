//! Execute a staged RISC-V program on RISCV B.
//!
//! tt-metal's shape (`brisc.cc`, `kernel_main()`): resident firmware, a
//! launch message naming the kernel, the kernel called as a function, the core
//! reporting completion. Here the "launch message" is
//! [`tt_isa::mailbox::exec`]'s control block: program address, word count,
//! arguments base, a generation (`SEQ`). For each generation the core
//! invalidates its instruction cache, calls the program with `a0` at the
//! arguments, and publishes `RET` and `DONE`.
//!
//! The program is plain RV32IM ending in `ret`. It may use the caller-saved
//! registers only (`t0`-`t6`, `a0`-`a7`); this image makes no promises beyond
//! that, as tt-metal's firmware does not either.

#![no_std]
#![no_main]

use tt_firmware::{disable_gathering, l1_read32, l1_write32, publish, ttinsn};
use tt_isa::mailbox::exec;

fn rd(at: u64) -> u32 {
    // SAFETY: a fixed aligned word of the control block.
    unsafe { l1_read32(at) }
}

fn wr(at: u64, v: u32) {
    // SAFETY: a fixed aligned word of the control block.
    unsafe { l1_write32(at, v) }
}

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    disable_gathering();
    loop {
        publish();
        let seq = rd(exec::SEQ);
        if seq == 0 || seq == rd(exec::DONE) {
            continue;
        }
        let program = rd(exec::PROGRAM);
        let words = u64::from(rd(exec::LEN));
        let args = rd(exec::ARGS);
        // A refusal is published, never a guess at what the host meant: a
        // program outside L1, a zero length, or one that cannot end in a
        // return would otherwise execute whatever lies there.
        if program == 0 || program % 4 != 0 || words == 0 || words > exec::MAX_WORDS {
            wr(exec::ERROR, 1);
            publish();
            wr(exec::DONE, seq);
            publish();
            continue;
        }
        wr(exec::ERROR, 0);
        // SAFETY: the program's last word lies in L1 (checked above), and the
        // control block's generation made the host publish the code first.
        unsafe {
            ttinsn::invalidate_icache(u64::from(program) + 4 * (words - 1), 0x1);
            let ret = ttinsn::call_with(u64::from(program), args);
            wr(exec::RET, ret);
        }
        publish();
        wr(exec::DONE, seq);
        publish();
    }
}
