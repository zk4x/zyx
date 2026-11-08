//! Instruction-cache probe on RISCV NC (`src/icache_probe.rs`), at NC's image
//! base, reached through the stub at its reset PC.

#![no_std]
#![no_main]

/// As `icache_b`'s, 256 bytes on.
const RESULTS: u64 = tt_isa::mailbox::MAILBOX_BASE + 0x3100;

/// Each code block's size.
const BLOCK: u32 = 8192;

include!("../icache_probe.rs");
