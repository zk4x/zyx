// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Minimal RV32I emission for the role and mover programs.
//!
//! Only the instructions a compiled Tensix kernel actually needs: integer
//! loads/stores, adds and shifts for addressing, branches and jumps, `fence`.
//! No compressed encodings (their encoding space is `.ttinsn` on Blackhole), no
//! multiply/divide, no floating point, no atomics.
//!
//! Branch and jump targets are labels in [`Assembler`]; `finish` resolves them
//! and refuses an offset that does not fit the encoding. A branch reaches
//! ±4 KiB and a `jal` ±1 MiB; a program that outgrows those is refused, loudly,
//! rather than silently pointing somewhere else.
//!
//! The full instruction surface is here from the start; the callers are the
//! unit tests and the mover programs as the pipeline starts emitting them.

#![allow(dead_code)]

/// Register numbers, with their ABI names.
///
/// The program uses the caller-saved set only (`t0`-`t6`, `a0`-`a7`); see
/// "Role programs are RISC-V code" in NOTES. `SP`, `GP`, `TP`, `RA` and
/// `S0`-`S11` are named so the emitter can assert it never writes them.
pub mod reg {
    pub const ZERO: u8 = 0;
    pub const RA: u8 = 1;
    pub const SP: u8 = 2;
    pub const GP: u8 = 3;
    pub const TP: u8 = 4;
    pub const T0: u8 = 5;
    pub const T1: u8 = 6;
    pub const T2: u8 = 7;
    pub const S0: u8 = 8;
    pub const S1: u8 = 9;
    pub const A0: u8 = 10;
    pub const A1: u8 = 11;
    pub const A2: u8 = 12;
    pub const A3: u8 = 13;
    pub const A4: u8 = 14;
    pub const A5: u8 = 15;
    pub const A6: u8 = 16;
    pub const A7: u8 = 17;
    pub const S2: u8 = 18;
    pub const S3: u8 = 19;
    pub const S4: u8 = 20;
    pub const S5: u8 = 21;
    pub const S6: u8 = 22;
    pub const S7: u8 = 23;
    pub const S8: u8 = 24;
    pub const S9: u8 = 25;
    pub const S10: u8 = 26;
    pub const S11: u8 = 27;
    pub const T3: u8 = 28;
    pub const T4: u8 = 29;
    pub const T5: u8 = 30;
    pub const T6: u8 = 31;
}

/// Opcodes, as the ISA numbers them.
const OP_LOAD: u32 = 0x03;
const OP_STORE: u32 = 0x23;
const OP_IMM: u32 = 0x13;
const OP_REG: u32 = 0x33;
const OP_LUI: u32 = 0x37;
const OP_AUIPC: u32 = 0x17;
const OP_JAL: u32 = 0x6f;
const OP_JALR: u32 = 0x67;
const OP_BRANCH: u32 = 0x63;
const OP_MISC_MEM: u32 = 0x0f;

/// `fence iorw, iorw`, the whole fence on Blackhole.
pub const FENCE: u32 = 0x0ff0_000f;
/// `nop`.
pub const NOP: u32 = 0x0000_0013;
/// `ret`, `jalr zero, 0(ra)`.
pub const RET: u32 = 0x0000_8067;

const fn r_type(opcode: u32, funct3: u32, funct7: u32, rd: u8, rs1: u8, rs2: u8) -> u32 {
    (funct7 << 25) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | opcode
}

const fn i_type(opcode: u32, funct3: u32, rd: u8, rs1: u8, imm: i32) -> u32 {
    (((imm as u32) & 0xfff) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | opcode
}

const fn s_type(opcode: u32, funct3: u32, rs1: u8, rs2: u8, imm: i32) -> u32 {
    let imm = imm as u32;
    (((imm >> 5) & 0x7f) << 25) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((imm & 0x1f) << 7) | opcode
}

const fn b_type(funct3: u32, rs1: u8, rs2: u8, imm: i32) -> u32 {
    let imm = imm as u32;
    (((imm >> 12) & 1) << 31)
        | (((imm >> 5) & 0x3f) << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | (((imm >> 1) & 0xf) << 8)
        | (((imm >> 11) & 1) << 7)
        | OP_BRANCH
}

const fn j_type(rd: u8, imm: i32) -> u32 {
    let imm = imm as u32;
    (((imm >> 20) & 1) << 31)
        | (((imm >> 1) & 0x3ff) << 21)
        | (((imm >> 11) & 1) << 20)
        | (((imm >> 12) & 0xff) << 12)
        | ((rd as u32) << 7)
        | OP_JAL
}

// -- arithmetic ------------------------------------------------------------

pub fn addi(rd: u8, rs1: u8, imm: i32) -> u32 {
    assert!(fits_i12(imm), "addi immediate {imm} does not fit 12 bits");
    i_type(OP_IMM, 0, rd, rs1, imm)
}
pub fn add(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(OP_REG, 0, 0, rd, rs1, rs2)
}
pub fn sub(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(OP_REG, 0, 0x20, rd, rs1, rs2)
}
pub fn and(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(OP_REG, 7, 0, rd, rs1, rs2)
}
pub fn or(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(OP_REG, 6, 0, rd, rs1, rs2)
}
pub fn xor(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(OP_REG, 4, 0, rd, rs1, rs2)
}
pub fn slli(rd: u8, rs1: u8, sh: u32) -> u32 {
    assert!(sh < 32, "shift {sh} does not fit 5 bits");
    i_type(OP_IMM, 1, rd, rs1, sh as i32)
}
pub fn andi(rd: u8, rs1: u8, imm: i32) -> u32 {
    assert!(fits_i12(imm), "andi immediate {imm} does not fit 12 bits");
    i_type(OP_IMM, 7, rd, rs1, imm)
}
pub fn srli(rd: u8, rs1: u8, sh: u32) -> u32 {
    assert!(sh < 32, "shift {sh} does not fit 5 bits");
    i_type(OP_IMM, 5, rd, rs1, sh as i32)
}
pub fn lui(rd: u8, imm20: u32) -> u32 {
    ((imm20 & 0xfffff) << 12) | ((rd as u32) << 7) | OP_LUI
}
pub fn auipc(rd: u8, imm20: u32) -> u32 {
    ((imm20 & 0xfffff) << 12) | ((rd as u32) << 7) | OP_AUIPC
}

/// `rd = imm`, one or two instructions.
///
/// Returns up to two words; an immediate that fits 12 bits signed is one
/// `addi`, anything else is `lui` + `addi`.
pub fn li(rd: u8, imm: u32) -> ([u32; 2], usize) {
    let v = imm as i32;
    if fits_i12(v) {
        ([addi(rd, reg::ZERO, v), NOP], 1)
    } else {
        let hi = ((imm as i64 + 0x800) >> 12) as u32;
        let lo = imm.wrapping_sub(hi << 12) as i32;
        ([lui(rd, hi), addi(rd, rd, lo)], 2)
    }
}

pub const fn fits_i12(v: i32) -> bool {
    v >= -2048 && v <= 2047
}

// -- memory ----------------------------------------------------------------

pub fn lw(rd: u8, rs1: u8, imm: i32) -> u32 {
    assert!(fits_i12(imm), "lw offset {imm} does not fit 12 bits");
    i_type(OP_LOAD, 2, rd, rs1, imm)
}
pub fn lbu(rd: u8, rs1: u8, imm: i32) -> u32 {
    assert!(fits_i12(imm), "lbu offset {imm} does not fit 12 bits");
    i_type(OP_LOAD, 4, rd, rs1, imm)
}
pub fn sw(rs1: u8, rs2: u8, imm: i32) -> u32 {
    assert!(fits_i12(imm), "sw offset {imm} does not fit 12 bits");
    s_type(OP_STORE, 2, rs1, rs2, imm)
}
pub fn sb(rs1: u8, rs2: u8, imm: i32) -> u32 {
    assert!(fits_i12(imm), "sb offset {imm} does not fit 12 bits");
    s_type(OP_STORE, 0, rs1, rs2, imm)
}

// -- control ---------------------------------------------------------------

pub const fn beq(rs1: u8, rs2: u8, imm: i32) -> u32 {
    b_type(0, rs1, rs2, imm)
}
pub const fn bne(rs1: u8, rs2: u8, imm: i32) -> u32 {
    b_type(1, rs1, rs2, imm)
}
pub const fn blt(rs1: u8, rs2: u8, imm: i32) -> u32 {
    b_type(4, rs1, rs2, imm)
}
pub const fn bge(rs1: u8, rs2: u8, imm: i32) -> u32 {
    b_type(5, rs1, rs2, imm)
}
pub const fn bltu(rs1: u8, rs2: u8, imm: i32) -> u32 {
    b_type(6, rs1, rs2, imm)
}
pub const fn bgeu(rs1: u8, rs2: u8, imm: i32) -> u32 {
    b_type(7, rs1, rs2, imm)
}
pub const fn jal(rd: u8, imm: i32) -> u32 {
    j_type(rd, imm)
}
pub const fn jalr(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(OP_JALR, 0, rd, rs1, imm)
}

// -- assembler -------------------------------------------------------------

/// A forward or backward target, bound once.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Label(u32);

#[derive(Clone, Copy)]
struct Fixup {
    /// Word index of the instruction to patch.
    at: usize,
    label: u32,
    kind: Kind,
}

#[derive(Clone, Copy)]
enum Kind {
    Branch(u32, u8, u8),
    Jal(u8),
}

/// A program under construction.
#[derive(Default)]
pub struct Assembler {
    words: Vec<u32>,
    labels: Vec<Option<usize>>,
    fixups: Vec<Fixup>,
}

impl Assembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// The program so far, with no unresolved target.
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    pub fn here(&self) -> usize {
        self.words.len()
    }

    pub fn push(&mut self, word: u32) -> usize {
        self.words.push(word);
        self.words.len() - 1
    }

    /// Emit `rd = imm` (one or two words).
    pub fn li(&mut self, rd: u8, imm: u32) {
        let (words, n) = li(rd, imm);
        self.push(words[0]);
        if n == 2 {
            self.push(words[1]);
        }
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() as u32 - 1)
    }

    pub fn bind(&mut self, label: Label) {
        let slot = &mut self.labels[label.0 as usize];
        assert!(slot.is_none(), "label bound twice");
        *slot = Some(self.words.len());
    }

    pub fn beq(&mut self, rs1: u8, rs2: u8, label: Label) {
        let at = self.push(beq(rs1, rs2, 0));
        self.fixups.push(Fixup { at, label: label.0, kind: Kind::Branch(0, rs1, rs2) });
    }

    pub fn bne(&mut self, rs1: u8, rs2: u8, label: Label) {
        let at = self.push(bne(rs1, rs2, 0));
        self.fixups.push(Fixup { at, label: label.0, kind: Kind::Branch(1, rs1, rs2) });
    }

    pub fn jal(&mut self, rd: u8, label: Label) {
        let at = self.push(jal(rd, 0));
        self.fixups.push(Fixup { at, label: label.0, kind: Kind::Jal(rd) });
    }

    /// Resolve every target and return the program.
    ///
    /// Panics on an unbound label or an offset outside the encoding: a program
    /// that does not fit is a bug, not something to patch around.
    pub fn finish(mut self) -> Vec<u32> {
        for f in std::mem::take(&mut self.fixups) {
            let Some(target) = self.labels[f.label as usize] else {
                panic!("riscv: label {} is never bound", f.label);
            };
            let imm = (target as isize - f.at as isize) * 4;
            let imm = i32::try_from(imm).expect("riscv: jump offset does not fit i32");
            self.words[f.at] = match f.kind {
                Kind::Branch(f3, rs1, rs2) => {
                    assert!((-4096..=4094).contains(&imm), "riscv: branch offset {imm} does not fit ±4 KiB");
                    b_type(f3, rs1, rs2, imm)
                }
                Kind::Jal(rd) => {
                    assert!((-1_048_576..=1_048_574).contains(&imm), "riscv: jal offset {imm} does not fit ±1 MiB");
                    j_type(rd, imm)
                }
            };
        }
        self.words
    }
}

// -- NoC ------------------------------------------------------------------

/// Inline NOC requests, one initiator, one transaction id.
///
/// The register writes are the ones the firmware's issue path makes
/// (`firmware/tt-firmware/src/lib.rs:656`, `issue_dram_on`): wait until the
/// initiator is free, write the ten words, write `CMD_CTRL`, read `CMD_CTRL`
/// back so the outstanding-counter read cannot overtake the issue, then poll
/// the counter's low byte until the request has completed.
pub mod noc {
    use super::*;
    use tt_isa::noc::niu::{self, CMD_RD, CMD_WR, Niu, RESP_MARKED, STATIC_VC_1, TxnId, initiator};

    /// `NOC_CTRL` for a DRAM read.
    pub const READ: u32 = CMD_RD | RESP_MARKED | STATIC_VC_1;
    /// `NOC_CTRL` for a DRAM write.
    pub const WRITE: u32 = CMD_WR | RESP_MARKED | STATIC_VC_1;

    /// A request address: a register or a constant.
    #[derive(Copy, Clone)]
    pub enum Addr {
        Reg(u8),
        Imm(u32),
    }

    fn store(a: &mut Assembler, base: u8, off: u64, addr: Addr, scratch: u8) {
        let off = off as i32;
        match addr {
            Addr::Reg(r) => {
                a.push(sw(base, r, off));
            }
            Addr::Imm(0) => {
                a.push(sw(base, reg::ZERO, off));
            }
            Addr::Imm(v) => {
                a.li(scratch, v);
                a.push(sw(base, scratch, off));
            }
        }
    }

    /// Wait until the initiator's `CMD_CTRL` low bit is clear.
    pub fn wait_idle(a: &mut Assembler, base: u8, scratch: u8) {
        let top = a.label();
        a.bind(top);
        a.push(lw(scratch, base, initiator::CMD_CTRL as i32));
        a.push(andi(scratch, scratch, 1));
        a.bne(scratch, reg::ZERO, top);
    }

    /// Issue one request and return with it in flight.
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        a: &mut Assembler,
        base: u8,
        targ: Addr,
        targ_hi: u32,
        ret: Addr,
        ret_hi: u32,
        tag: u32,
        len: u32,
        write: bool,
        scratch: u8,
    ) {
        wait_idle(a, base, scratch);
        store(a, base, initiator::TARG_ADDR_LO, targ, scratch);
        store(a, base, initiator::TARG_ADDR_MID, Addr::Imm(0), scratch);
        store(a, base, initiator::TARG_ADDR_HI, Addr::Imm(targ_hi), scratch);
        store(a, base, initiator::RET_ADDR_LO, ret, scratch);
        store(a, base, initiator::RET_ADDR_MID, Addr::Imm(0), scratch);
        store(a, base, initiator::RET_ADDR_HI, Addr::Imm(ret_hi), scratch);
        store(a, base, initiator::PACKET_TAG, Addr::Imm(tag), scratch);
        store(a, base, initiator::CTRL, Addr::Imm(if write { WRITE } else { READ }), scratch);
        store(a, base, initiator::AT_LEN_BE, Addr::Imm(len), scratch);
        store(a, base, initiator::AT_DATA, Addr::Imm(0), scratch);
        store(a, base, initiator::CMD_CTRL, Addr::Imm(1), scratch);
        // A read-back of the command register orders the counter read after
        // the issue (`Counters.md:42-43`).
        a.push(lw(scratch, base, initiator::CMD_CTRL as i32));
    }

    /// Poll the transaction's outstanding counter until the request completes.
    pub fn drain(a: &mut Assembler, base: u8, txn: TxnId, scratch: u8) {
        let top = a.label();
        a.bind(top);
        a.push(lw(scratch, base, niu::reqs_outstanding(txn) as i32));
        a.push(andi(scratch, scratch, 0xff));
        a.bne(scratch, reg::ZERO, top);
    }

    /// Put the NOC's initiator base in `base`.
    pub fn open(a: &mut Assembler, base: u8, niu: Niu) {
        a.li(base, niu.base() as u32);
    }
}

/// A row-gather writeback: for each of `count` rows, read `row_bytes` at
/// `src + index[i] * row_bytes` into the one L1 page, then write the page to
/// `dst + i * row_bytes`.
///
/// The indices live in L1 at `idx` as `u32`; `stride_shift` is
/// `log2(row_bytes)` and `row_bytes` is `len`. One page, one CB-less loop: the
/// first writeback program (`tt_metal/programming_examples/loopback`).
#[derive(Copy, Clone, Debug)]
pub struct Gather {
    /// Which NoC issues.
    pub niu: tt_isa::noc::niu::Niu,
    /// Transaction id the outstanding counter is polled on.
    pub txn: tt_isa::noc::niu::TxnId,
    /// DRAM channel offset of the source and the destination.
    pub src: u32,
    pub dst: u32,
    /// High address words: the source/destination endpoints and this tile.
    pub src_hi: u32,
    pub dst_hi: u32,
    pub me_hi: u32,
    /// L1 page the row lands in.
    pub page: u32,
    /// L1 address of the `u32` index array.
    pub idx: u32,
    /// Number of rows.
    pub count: u32,
    /// `log2(row_bytes)`.
    pub stride_shift: u32,
    /// Bytes per row; a power of two, and a multiple of 64.
    pub row_bytes: u32,
    /// `PACKET_TAG`.
    pub tag: u32,
}

impl Gather {
    /// The program: prologue, the gather/writeback loop, `ret`.
    pub fn program(&self) -> Vec<u32> {
        assert!(
            self.row_bytes.is_power_of_two() && self.row_bytes >= 64,
            "riscv: row bytes {} must be a power of two and a multiple of 64",
            self.row_bytes
        );
        assert_eq!(1u32 << self.stride_shift, self.row_bytes, "riscv: stride shift");
        assert!(self.count > 0, "riscv: gather needs at least one row");
        assert_eq!(self.src % 64, self.page % 64, "riscv: read not congruent mod 64");
        assert_eq!(self.page % 16, self.dst % 16, "riscv: write not congruent mod 16");
        use reg::*;
        let mut a = Assembler::new();
        // n = NOC base, idx pointer, count, source, destination, page.
        noc::open(&mut a, T0, self.niu);
        a.li(T1, self.idx);
        a.li(T2, self.count);
        a.li(T3, self.src);
        a.li(T4, self.dst);
        a.li(A0, self.page);
        let top = a.label();
        a.bind(top);
        // index -> source address.
        a.push(lw(T5, T1, 0));
        a.push(slli(T5, T5, self.stride_shift));
        a.push(add(T5, T3, T5));
        // DRAM -> L1.
        noc::issue(
            &mut a,
            T0,
            noc::Addr::Reg(T5),
            self.src_hi,
            noc::Addr::Imm(self.page),
            self.me_hi,
            self.tag,
            self.row_bytes,
            false,
            T6,
        );
        noc::drain(&mut a, T0, self.txn, T6);
        // L1 -> DRAM.
        noc::issue(
            &mut a,
            T0,
            noc::Addr::Imm(self.page),
            self.me_hi,
            noc::Addr::Reg(T4),
            self.dst_hi,
            self.tag,
            self.row_bytes,
            true,
            T6,
        );
        noc::drain(&mut a, T0, self.txn, T6);
        a.push(addi(T1, T1, 4));
        a.push(addi(T4, T4, self.row_bytes as i32));
        a.push(addi(T2, T2, -1));
        a.bne(T2, ZERO, top);
        a.push(RET);
        a.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_words() {
        assert_eq!(NOP, 0x0000_0013);
        assert_eq!(RET, 0x0000_8067);
        assert_eq!(FENCE, 0x0ff0_000f);
        assert_eq!(addi(reg::SP, reg::SP, -16), 0xff01_0113);
        assert_eq!(sw(reg::A1, reg::A0, 0), 0x00a5_a023);
        assert_eq!(lw(reg::A0, reg::A1, 4), 0x0045_a503);
        assert_eq!(lui(reg::T0, 0x12), 0x0001_22b7);
        assert_eq!(jal(reg::ZERO, 0), 0x0000_006f);
        assert_eq!(beq(reg::ZERO, reg::ZERO, 0), 0x0000_0063);
        assert_eq!(jalr(reg::ZERO, reg::RA, 0), 0x0000_8067);
    }

    #[test]
    fn li_small_and_large() {
        let (w, n) = li(reg::T0, 7);
        assert_eq!(n, 1);
        assert_eq!(w[0], addi(reg::T0, reg::ZERO, 7));

        let (w, n) = li(reg::T0, 0x0012_0000);
        assert_eq!(n, 2);
        assert_eq!(w[0], lui(reg::T0, 0x120));
        assert_eq!(w[1], addi(reg::T0, reg::T0, 0));

        // The low 12 bits are signed, so 0x800 in the immediate borrows one.
        let (w, n) = li(reg::T0, 0x0000_0fff);
        assert_eq!(n, 2);
        assert_eq!(w[0], lui(reg::T0, 1));
        assert_eq!(w[1], addi(reg::T0, reg::T0, -1));
    }

    #[test]
    fn gather_program_loops_and_returns() {
        let g = Gather {
            niu: tt_isa::noc::niu::Niu::Noc0,
            txn: tt_isa::noc::niu::TxnId::new(0).unwrap(),
            src: 0x1000,
            dst: 0x2040,
            src_hi: 0x100,
            dst_hi: 0x100,
            me_hi: 0x41,
            page: 0x1300,
            idx: 0x2000,
            count: 4,
            stride_shift: 6,
            row_bytes: 64,
            tag: 0,
        };
        let words = g.program();
        assert_eq!(*words.last().unwrap(), RET);
        // One backward branch: the loop's `bne`.
        let back = words.iter().filter(|w| **w & 0x7f == 0x63 && (**w >> 31) != 0).count();
        // Five backward branches: wait-idle and drain around each of the two
        // requests, and the gather loop itself.
        assert_eq!(back, 5, "three poll loops and the gather loop");
        assert_eq!(words.iter().filter(|w| **w == RET).count(), 1);
    }

    #[test]
    #[should_panic(expected = "multiple of 64")]
    fn gather_refuses_a_misaligned_row() {
        let g = Gather {
            niu: tt_isa::noc::niu::Niu::Noc0,
            txn: tt_isa::noc::niu::TxnId::new(0).unwrap(),
            src: 0x1000,
            dst: 0x2000,
            src_hi: 0,
            dst_hi: 0,
            me_hi: 0,
            page: 0x1000,
            idx: 0,
            count: 1,
            stride_shift: 5,
            row_bytes: 32,
            tag: 0,
        };
        g.program();
    }

    #[test]
    fn backward_branch_target() {
        let mut a = Assembler::new();
        let top = a.label();
        a.bind(top);
        a.push(NOP);
        a.bne(reg::T0, reg::ZERO, top);
        let words = a.finish();
        // The branch is word 1 and the target word 0: -4 bytes.
        assert_eq!(words[1], bne(reg::T0, reg::ZERO, -4));
    }

    #[test]
    fn forward_jump_and_call() {
        let mut a = Assembler::new();
        let end = a.label();
        a.jal(reg::RA, end);
        a.push(NOP);
        a.bind(end);
        a.push(RET);
        let words = a.finish();
        assert_eq!(words[0], jal(reg::RA, 8));
    }

    #[test]
    #[should_panic(expected = "does not fit ±4 KiB")]
    fn branch_out_of_range_is_loud() {
        let mut a = Assembler::new();
        let target = a.label();
        a.bne(reg::T0, reg::ZERO, target);
        for _ in 0..2000 {
            a.push(NOP);
        }
        a.bind(target);
        a.finish();
    }
}
