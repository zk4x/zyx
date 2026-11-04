//! Diagnostic RISC-V configuration-store seeding, separate from WRCFG.
#![no_std]
#![no_main]

use tt_firmware::cfg::write_config_field;
use tt_firmware::tensix::{push_word, read_dst32, wait_for_coprocessor};
use tt_firmware::{fail, finish, l1_read32, l1_write32, publish, spin};
use tt_isa::cfg::{generated::global::PRNG_SEED_Seed_Val, ConfigBank};
use tt_isa::mailbox::{self, panic_code};
use tt_isa::sfpu::dst32_address;
use tt_isa::tensix::{RiscvT1, TensixThread, Thread1};

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    // SAFETY: aligned mailbox words staged before this core was released.
    let (seed, len, thread) = unsafe {
        (
            l1_read32(mailbox::OPERAND_A),
            l1_read32(mailbox::PROGRAM_LEN),
            l1_read32(mailbox::THREAD_INDEX),
        )
    };
    if thread != Thread1::INDEX || len == 0 || len > mailbox::PROGRAM_MAX {
        fail(panic_code::EXPLICIT);
    }
    // SAFETY: the host released the backend; this image owns T1 and no
    // coprocessor work has been submitted yet. Each later seed store follows
    // a complete drain, not just a scheduling bubble.
    unsafe { write_config_field(Thread1::DST_ACCESS_FMT, ConfigBank::Bank0, 0) };
    for (phase, value) in [seed, seed, !seed, seed].into_iter().enumerate() {
        // SAFETY: the coprocessor is idle; only the generated full-width seed
        // field is written. Fence before pushing anything dependent on it.
        unsafe {
            core::ptr::write_volatile(
                PRNG_SEED_Seed_Val.riscv_address(ConfigBank::Bank0) as *mut u32,
                value,
            );
        }
        publish();
        // Allow the seed distribution circuitry to settle before SFPMOV.
        // Coprocessor/store retirement alone does not establish that it has.
        for _ in 0..512 {
            // SAFETY: ordinary RISC-V NOP, supported by the instruction gate.
            unsafe { core::arch::asm!("nop", options(nomem, nostack)) };
        }
        for i in 0..len {
            // SAFETY: the checked length stays in the staged program region;
            // RiscvT1 may push to Thread1 and the backend is released.
            unsafe {
                push_word::<RiscvT1, Thread1>(l1_read32(mailbox::PROGRAM + u64::from(i) * 4));
            }
        }
        wait_for_coprocessor();
        for row in 0..8 {
            for col in 0..mailbox::DUMP_ROW_WORDS {
                // SAFETY: Dst's F32 mapping is explicit, the program retired,
                // and all four eight-row snapshots fit the mailbox dump.
                unsafe {
                    l1_write32(
                        mailbox::dump_offset(phase as u32 * 8 + row, col),
                        read_dst32(dst32_address(row, col)),
                    );
                }
            }
        }
        publish();
    }
    finish(len);
    spin()
}
