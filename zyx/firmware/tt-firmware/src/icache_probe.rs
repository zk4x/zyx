// The instruction-cache probe, shared by every core's image
// (`bin/icache_*.rs`): each binary defines `RESULTS`, where in L1 the timings
// go, and `BLOCK`, each block's size (8 KiB on B and NC; 6 KiB on T0-T2,
// whose image slots are 16 KiB), and includes this.
//
// Two `BLOCK`-byte blocks of code, each ending in `ret`. Calling into a block at
// `end - size` runs its last `size` bytes:
// * straight-line: one `nop` every 4 bytes, so every byte is fetched in
//   order (a sequential prefetcher can hide misses);
// * jump chain: one `j` every 32 bytes to the next 32, so each instruction
//   run touches a new 32-byte span (no sequential run to prefetch).
// For each size the block runs once to warm, then `PASSES` times timed by the
// tile's wall clock. Where the cycles per byte step up is where the code no
// longer fits the instruction cache.

use tt_firmware::{finish, l1_write32, publish};

core::arch::global_asm!(
    r#"
    .section .text.probe_blocks, "ax"
    .balign 64
    .global probe_line_start
probe_line_start:
    .rept {lines}
    nop
    .endr
    .global probe_line_end
probe_line_end:
    ret

    .balign 64
    .global probe_jump_start
probe_jump_start:
    .rept {jumps}
    j 1f
    .rept 7
    nop
    .endr
1:
    .endr
    .global probe_jump_end
probe_jump_end:
    ret
"#,
    lines = const BLOCK / 4,
    jumps = const BLOCK / 32,
);

extern "C" {
    static probe_line_end: u8;
    static probe_jump_end: u8;
}

/// Sizes probed, in bytes: multiples of 32; those past `BLOCK` are skipped
/// (reported with zero cycles).
const SIZES: [u32; 14] = [
    256, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192,
];
const PASSES: u32 = 16;

fn clock() -> u32 {
    // SAFETY: the tile's wall clock, a read-only debug register every baby
    // core reaches (`BabyRISCV/README.md`, memory map).
    unsafe { core::ptr::read_volatile(tt_isa::tensix::timestamper::WALL_CLOCK_L as *const u32) }
}

/// Cycles for `PASSES` runs of the last `size` bytes of the block ending at
/// `end`, after one untimed run.
fn time(end: *const u8, size: u32) -> u32 {
    // SAFETY: `end - size` is inside the block, on an instruction boundary
    // (a multiple of 32 bytes before the `ret`), and the code from there
    // runs to the `ret` touching nothing.
    let f: extern "C" fn() = unsafe { core::mem::transmute(end.sub(size as usize)) };
    f();
    let t0 = clock();
    for _ in 0..PASSES {
        f();
    }
    clock().wrapping_sub(t0)
}

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    // Linker-defined labels inside this image's code.
    let (line, jump) = (
        core::ptr::addr_of!(probe_line_end),
        core::ptr::addr_of!(probe_jump_end),
    );
    for (k, &size) in SIZES.iter().enumerate() {
        let at = RESULTS + k as u64 * 12;
        // SAFETY: `RESULTS` is free L1 this binary owns (see its definition).
        unsafe {
            l1_write32(at, size);
            let fits = size <= BLOCK;
            l1_write32(at + 4, if fits { time(line, size) } else { 0 });
            l1_write32(at + 8, if fits { time(jump, size) } else { 0 });
        }
    }
    publish();
    finish(SIZES.len() as u32);
    tt_firmware::spin()
}
