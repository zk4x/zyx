//! Instruction-cache probe on RISCV B (`src/icache_probe.rs`).

#![no_std]
#![no_main]

/// Where the timings go: `[size, straight-line cycles, jump-chain cycles]`
/// per size, in the mailbox region's free stretch (below RISCV NC's ring).
const RESULTS: u64 = tt_isa::mailbox::MAILBOX_BASE + 0x3000;

/// Each code block's size.
const BLOCK: u32 = 8192;

include!("../icache_probe.rs");
