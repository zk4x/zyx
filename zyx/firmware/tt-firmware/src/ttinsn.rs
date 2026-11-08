//! Run-time translation of a staged program into fused `.ttinsn` code
//! (`docs/plans/ttinsn-push-experiment.md`, "Run-time translator"), and the
//! instruction-cache invalidation executing it needs.
//!
//! The role runners call it when a program's block repeats make the exchange pay
//! ([`crate::corpus::run_role`]). The build-time gate cannot see words written at
//! run time, so this module is the gate for them: it emits nothing but a closed
//! set of things, and refuses (returns `None`, so the caller pushes the program
//! the ordinary way) anything else.
//!
//! * a staged word becomes `.ttinsn` only if it is below `0xC000_0000` (a
//!   `.ttinsn` word is rotated left two bits, and its low two bits must not both
//!   be set) and, when `opcodes` is given, its opcode byte is a Tensix instruction
//!   (a bitmap the host generates from the instruction table; [`OPCODES`] is the
//!   firmware's own copy, built from the same table at compile time);
//! * the only RISC-V instructions emitted are `addi` (counter set, decrement),
//!   `bne` (backward, to the loop head) and `jalr x0, 0(ra)` (return), over the
//!   caller-saved `t0` and `t1`, loops at most two deep.

use crate::{l1_write32, publish};
use tt_isa::cfg::{generated::global::RISCV_IC_INVALIDATE_InvalidateAll, ConfigBank};

/// Words in an opcode bitmap: one bit per opcode byte.
pub const OPCODE_WORDS: usize = 8;

/// Every Tensix instruction's opcode byte, one bit each: the check a staged word
/// must pass before it is written as `.ttinsn` code. Built here from the same
/// table the host's generator reads, so the two cannot drift.
pub const OPCODES: [u32; OPCODE_WORDS] = opcode_bitmap();

const fn opcode_bitmap() -> [u32; OPCODE_WORDS] {
    let mut bits = [0u32; OPCODE_WORDS];
    let mut i = 0;
    while i < tt_isa::isa::generated::ALL.len() {
        let opcode = tt_isa::isa::generated::ALL[i].opcode();
        bits[(opcode >> 5) as usize] |= 1 << (opcode & 31);
        i += 1;
    }
    bits
}

const ZERO: u32 = 0;
const T0: u32 = 5;
const T1: u32 = 6;
/// The Tensix GPRs a stepped address lives in: its value and its per-iteration
/// advance. Tensix GPRs, not RISC-V ones: the config write (`WRCFG`) and the
/// advance (`ADDDMAREG`) are coprocessor instructions, so they stay in the
/// instruction FIFO and are ordered with the unpack that reads the base. A
/// RISC-V `sw` into the same config word is not (measured: the base lagged a
/// whole iteration, tiles 0, 0, 2, 3, ...).
const BASE_GPR: u32 = 24;
const STRIDE_GPR: u32 = 26;
/// The config word a stepped address writes: the unpacker's
/// `REG3_Base_address`, in 16-byte units.
const BASE_ADDRESS_CFG: u32 = tt_isa::cfg::generated::thcon::THCON_SEC0_REG3_Base_address.addr32() as u32;
const A0: u32 = 10;
/// `jalr x0, 0(ra)`.
const RET: u32 = 0x0000_8067;
/// Loop counts that fit `addi`'s signed 12-bit immediate.
const MAX_COUNT: u32 = 2047;

/// A staged word the translator turns into `WRCFG t2`: the write of the
/// address register into the unpacker's base config word. The low 20 bits are
/// the register's first value, in the field's own units (the unpacker's base is
/// in 16-byte units, so an L1 address below 1 MiB fits). The body of a loop is
/// one copy: this is how an address that moves per iteration is expressed
/// without writing the body out.
///
/// It is `WRCFG` and not `sw` to the same config word on purpose: `WRCFG` is a
/// coprocessor instruction, so it is ordered in the instruction FIFO with the
/// `UNPACR` that reads the word. A RISC-V store is not, and the unpacker read a
/// stale base one iteration in (measured: tiles 0, 0, 2, 3, ...).
pub const STEP_STORE: u32 = 0x7F50_0000;
/// A staged word the translator turns into `addi t2, t2, step`: the advance of
/// that register for the next iteration, low 12 bits, in the same units.
pub const STEP_ADVANCE: u32 = 0x7F60_0000;
/// The bits a step word is recognised by.
pub const STEP_MASK: u32 = 0xFFF0_0000;

/// `lui rd, imm20`.
pub const fn lui(rd: u32, imm20: u32) -> u32 {
    ((imm20 & 0xf_ffff) << 12) | (rd << 7) | 0x37
}

/// `sw rs2, imm(rs1)`.
pub const fn sw(rs2: u32, rs1: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    (((imm >> 5) & 0x7f) << 25) | (rs2 << 20) | (rs1 << 15) | (0b010 << 12) | ((imm & 0x1f) << 7) | 0x23
}

/// `rd = value`, for a 32-bit constant: `lui` and a signed 12-bit remainder.
pub fn load_const(rd: u32, value: u32) -> [u32; 2] {
    let hi = (value >> 12) + u32::from(value & 0x800 != 0);
    [lui(rd, hi & 0xf_ffff), addi(rd, rd, ((value & 0xfff) as i32) - if value & 0x800 != 0 { 0x1000 } else { 0 })]
}

/// `addi rd, rs1, imm`.
pub const fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
    (((imm as u32) & 0xfff) << 20) | (rs1 << 15) | (rd << 7) | 0x13
}

/// `bne rs, x0, offset` (a B-type branch; `offset` in bytes, even, within +-4 KiB).
pub const fn bne(rs: u32, offset: i32) -> u32 {
    let imm = offset as u32;
    ((imm >> 12 & 1) << 31)
        | ((imm >> 5 & 0x3f) << 25)
        | (rs << 15)
        | (0b001 << 12)
        | ((imm >> 1 & 0xf) << 8)
        | ((imm >> 11 & 1) << 7)
        | 0x63
}

fn has_opcode(opcodes: &[u32; OPCODE_WORDS], byte: u32) -> bool {
    opcodes[(byte >> 5) as usize] >> (byte & 31) & 1 != 0
}

/// Words written to a code buffer in L1, bounded by `cap`.
struct Out {
    base: u64,
    cap: usize,
    len: usize,
}

impl Out {
    fn push(&mut self, word: u32) -> Option<()> {
        if self.len == self.cap {
            return None;
        }
        // SAFETY: inside the caller's buffer, which it owns for this.
        unsafe { l1_write32(self.base + 4 * self.len as u64, word) };
        self.len += 1;
        Some(())
    }
}

/// `[lo, hi)` of the staged program at `code`, each block repeat inside it as a
/// counted loop (the walk `corpus::Pusher::span` does).
fn emit(
    out: &mut Out,
    code: u64,
    loops: &[(u32, u32, u32)],
    opcodes: Option<&[u32; OPCODE_WORDS]>,
    (lo, hi): (u32, u32),
    depth: usize,
) -> Option<()> {
    let mut at = lo;
    loop {
        let mut next: Option<(u32, u32, u32)> = None;
        for &l in loops {
            let inside = l.0 >= at && l.0 + l.1 <= hi && (l.0, l.0 + l.1) != (lo, hi);
            if inside && next.is_none_or(|n| l.0 < n.0 || (l.0 == n.0 && l.1 > n.1)) {
                next = Some(l);
            }
        }
        let Some((start, len, count)) = next else {
            return words(out, code, opcodes, at, hi);
        };
        words(out, code, opcodes, at, start)?;
        if depth >= 2 || count == 0 || count > MAX_COUNT {
            return None;
        }
        // A body that steps an address needs its Tensix GPRs loaded before the
        // loop: the base with the first value, and the stride with the advance
        // (`set_gpr` is `SETDMAREG_Immediate` twice, a coprocessor write of a
        // literal -- there is no RISC-V register involved, so the value has to
        // be known here, which the staged step words carry).
        if let Some(base) = step_base(code, start, start + len) {
            for word in tt_isa::backend::set_gpr(BASE_GPR, base).ok()? {
                out.push(word.word().rotate_left(2))?;
            }
            if let Some(step) = step_advance(code, start, start + len) {
                for word in tt_isa::backend::set_gpr(STRIDE_GPR, step).ok()? {
                    out.push(word.word().rotate_left(2))?;
                }
            }
        }
        let reg = [T0, T1][depth];
        out.push(addi(reg, ZERO, count as i32))?;
        let body = out.len;
        emit(out, code, loops, opcodes, (start, start + len), depth + 1)?;
        out.push(addi(reg, reg, -1))?;
        // The branch is the next word, at index `out.len`, back to `body`.
        let offset = -((out.len - body) as i32) * 4;
        if offset < -4096 {
            return None;
        }
        out.push(bne(reg, offset))?;
        at = start + len;
    }
}

fn words(
    out: &mut Out,
    code: u64,
    opcodes: Option<&[u32; OPCODE_WORDS]>,
    lo: u32,
    hi: u32,
) -> Option<()> {
    let n = (hi - lo) as usize;
    let src = (code + 4 * lo as u64) as *const u32;
    for k in 0..n {
        // SAFETY: inside the staged program, whose length the runner checked.
        let w = unsafe { core::ptr::read_volatile(src.add(k)) };
        // A step word is the translator's own: the address register's store and
        // advance, never an instruction of the machine.
        if w & STEP_MASK == STEP_STORE {
            let write = tt_isa::isa::generated::encode::wrcfg(BASE_GPR, 0, BASE_ADDRESS_CFG).ok()?;
            out.push(write.word().rotate_left(2))?;
            continue;
        }
        if w & STEP_MASK == STEP_ADVANCE {
            let add = tt_isa::isa::generated::encode::adddmareg(BASE_GPR, STRIDE_GPR, BASE_GPR).ok()?;
            out.push(add.word().rotate_left(2))?;
            continue;
        }
        // `.ttinsn` is only valid below 0xC000_0000: the rotated word's low two
        // bits must not both be set, or it would be an ordinary instruction.
        if w >= 0xC000_0000 {
            return None;
        }
        if let Some(opcodes) = opcodes {
            if !has_opcode(opcodes, w >> 24) {
                return None;
            }
        }
        out.push(w.rotate_left(2))?;
    }
    Some(())
}

/// The first value of a body's stepping address, if it has one: the low 20 bits
/// of its [`STEP_STORE`] word.
fn step_base(code: u64, lo: u32, hi: u32) -> Option<u32> {
    for i in lo..hi {
        // SAFETY: inside the staged program, whose extent the runner checked.
        let w = unsafe { core::ptr::read_volatile((code + 4 * i as u64) as *const u32) };
        if w & STEP_MASK == STEP_STORE {
            return Some(w & 0xf_ffff);
        }
    }
    None
}

/// The advance of a body's stepping address: the low 12 bits of its
/// [`STEP_ADVANCE`] word.
fn step_advance(code: u64, lo: u32, hi: u32) -> Option<u32> {
    for i in lo..hi {
        // SAFETY: inside the staged program, whose extent the runner checked.
        let w = unsafe { core::ptr::read_volatile((code + 4 * i as u64) as *const u32) };
        if w & STEP_MASK == STEP_ADVANCE {
            return Some(w & 0xfff);
        }
    }
    None
}

/// Translate the staged program (`code_len` words at `code`, block repeats
/// `loops`) into `.ttinsn` code at `buffer` (room for `cap` words), ending in a
/// return. Returns the words written, or `None` if it cannot be done safely.
///
/// # Safety
///
/// `code` is the runner's checked program; `buffer` is L1 this caller owns, for
/// `cap` words.
pub unsafe fn translate(
    code: u64,
    code_len: u32,
    loops: &[(u32, u32, u32)],
    opcodes: Option<&[u32; OPCODE_WORDS]>,
    buffer: u64,
    cap: usize,
) -> Option<usize> {
    let mut out = Out {
        base: buffer,
        cap,
        len: 0,
    };
    emit(&mut out, code, loops, opcodes, (0, code_len), 0)?;
    out.push(RET)?;
    Some(out.len)
}

/// Invalidate T1's instruction cache after writing code to L1 at `last_written`
/// (`InstructionCache.md`, "Cache invalidation": the write to the invalidate
/// register must be processed after the writes to L1, or the flush can happen
/// first and the prefetcher or branch predictor refill the cache with the old
/// code).
///
/// The sequence the page gives -- a read back of the last write, consumed, then
/// the mask store -- followed by the 24 instructions it says may be needed for
/// the pipeline to drain. `mask` bit 2 is RISCV T1.
///
/// # Safety
///
/// Only this core's own cache (`mask` = 4) has been exercised; the Tensix backend
/// must be out of reset.
pub unsafe fn invalidate_icache(last_written: u64, mask: u32) {
    publish();
    let register = RISCV_IC_INVALIDATE_InvalidateAll.riscv_address(ConfigBank::Bank0);
    // SAFETY: the documented register, a whole-word store (`sw`: RISC-V cannot
    // write `Config` any narrower); the word holds only this field.
    unsafe {
        core::arch::asm!(
            "lw   {t}, 0({last})",
            "andi {t}, {t}, 0",
            "sw   {mask}, 0({register})",
            last = in(reg) last_written as u32,
            mask = in(reg) mask,
            register = in(reg) register as u32,
            t = out(reg) _,
            options(nostack),
        );
    }
    for _ in 0..24 {
        // SAFETY: a no-op; not `spin_loop`, whose `pause` Blackhole lacks.
        unsafe { core::arch::asm!("nop") };
    }
}

/// Call the code at `buffer` (a function taking nothing, returning `a0`).
///
/// # Safety
///
/// `buffer` holds valid RV32IM code ending in a return, and its cache lines are
/// current (`invalidate_icache`).
pub unsafe fn call(buffer: u64) -> u32 {
    // SAFETY: the caller's contract.
    let f: extern "C" fn() -> u32 = unsafe { core::mem::transmute(buffer as usize) };
    f()
}

/// `a0 = value; ret`: the smallest code with an observable result, for the
/// invalidation self-test.
///
/// # Safety
///
/// `buffer` is L1 the caller owns, for two words.
pub unsafe fn write_return_value(buffer: u64, value: i32) {
    // SAFETY: the caller owns `buffer`.
    unsafe {
        l1_write32(buffer, addi(A0, ZERO, value));
        l1_write32(buffer + 4, RET);
    }
}
