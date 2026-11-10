//! Instruction-cache probe on RISCV T1 (`src/icache_probe.rs`), at T1's
//! default reset PC.

#![no_std]
#![no_main]

/// Where the timings go, as `icache_b`'s.
const RESULTS: u64 = tt_isa::mailbox::MAILBOX_BASE + 0x3300;
/// Each code block's size: two and the probe fit T1's 16 KiB slot.
const BLOCK: u32 = 6144;

include!("../icache_probe.rs");
