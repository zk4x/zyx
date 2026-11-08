// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Disassembly of a role or mover program.
//!
//! A program is one flat word stream mixing two instruction sets (NOTES, "Role
//! programs are RISC-V code"): RV32IM words for control flow, addressing and
//! the inline CB/NOC sequences, and inline Tensix `.ttinsn` words for the
//! coprocessor. An RV32 opcode is the word's low seven bits, a Tensix opcode is
//! its top byte, so the two are told apart by decoding: the RV32 set is small
//! and fully enumerated here, and a word outside it is looked up in the
//! documented Tensix encodings. A word that is neither prints raw, never
//! guessed.
//!
//! This is the debug view behind `ZYX_DEBUG=16`, the analogue of the SPIR-V text
//! dump for the Tensix stream.

use tt_isa::isa::generated::ALL;

/// One line per word: `pc  hex  mnemonic operands`.
pub fn program(words: &[u32]) -> String {
    let mut out = String::new();
    for (pc, &word) in words.iter().enumerate() {
        out.push_str(&format!("{:6}  {:08x}  {}\n", pc, word, word_text(word)));
    }
    out
}

/// The text of one word, without the address.
pub fn word_text(word: u32) -> String {
    if let Some(rv) = riscv(word) {
        return rv;
    }
    if let Some(tensix) = tensix(word) {
        return tensix;
    }
    format!(".word {word:#010x}")
}

/// A documented Tensix encoding, named with its operands in field order.
fn tensix(word: u32) -> Option<String> {
    let def = ALL.iter().find(|d| d.matches(word))?;
    let mut s = def.mnemonic().to_string();
    for (i, field) in def.fields().iter().enumerate() {
        s.push_str(if i == 0 { " " } else { ", " });
        s.push_str(&format!("{}={}", field.name(), field.extract(word)));
    }
    Some(s)
}

const REGS: [&str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
    "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
];

fn reg(i: u32) -> &'static str {
    REGS[(i & 0x1f) as usize]
}

/// RV32IM as emitted by [`super::riscv`]. `None` for a word whose low seven
/// bits are not one of the opcodes the emitter produces.
fn riscv(w: u32) -> Option<String> {
    let opcode = w & 0x7f;
    let rd = (w >> 7) & 0x1f;
    let f3 = (w >> 12) & 7;
    let rs1 = (w >> 15) & 0x1f;
    let rs2 = (w >> 20) & 0x1f;
    let f7 = (w >> 25) & 0x7f;
    let imm_i = (w as i32) >> 20;
    let imm_s = ((((w >> 25) << 5) | ((w >> 7) & 0x1f)) as i32) - if w & 0x8000_0000 != 0 { 1 << 12 } else { 0 };
    let imm_b = {
        let b = (((w >> 31) & 1) << 12)
            | (((w >> 7) & 1) << 11)
            | (((w >> 25) & 0x3f) << 5)
            | (((w >> 8) & 0xf) << 1);
        let b = b as i32;
        if b & 0x1000 != 0 { b - 0x2000 } else { b }
    };
    let imm_j = {
        let j = (((w >> 31) & 1) << 20)
            | (((w >> 12) & 0xff) << 12)
            | (((w >> 20) & 1) << 11)
            | (((w >> 21) & 0x3ff) << 1);
        let j = j as i32;
        if j & 0x10_0000 != 0 { j - 0x20_0000 } else { j }
    };
    let upper = (w >> 12) & 0xf_ffff;
    Some(match (opcode, f3, f7) {
        (0x37, _, _) => format!("lui {}, 0x{upper:05x}", reg(rd)),
        (0x17, _, _) => format!("auipc {}, 0x{upper:05x}", reg(rd)),
        (0x6f, _, _) => format!("jal {}, {}", reg(rd), imm_j),
        (0x67, 0, _) if rd == 0 && rs1 == 1 && imm_i == 0 => "ret".to_string(),
        (0x67, 0, _) => format!("jalr {}, {}({})", reg(rd), imm_i, reg(rs1)),
        (0x63, 0, _) => format!("beq {}, {}, {}", reg(rs1), reg(rs2), imm_b),
        (0x63, 1, _) => format!("bne {}, {}, {}", reg(rs1), reg(rs2), imm_b),
        (0x63, 4, _) => format!("blt {}, {}, {}", reg(rs1), reg(rs2), imm_b),
        (0x63, 5, _) => format!("bge {}, {}, {}", reg(rs1), reg(rs2), imm_b),
        (0x63, 6, _) => format!("bltu {}, {}, {}", reg(rs1), reg(rs2), imm_b),
        (0x63, 7, _) => format!("bgeu {}, {}, {}", reg(rs1), reg(rs2), imm_b),
        (0x03, 0, _) => format!("lb {}, {}({})", reg(rd), imm_i, reg(rs1)),
        (0x03, 1, _) => format!("lh {}, {}({})", reg(rd), imm_i, reg(rs1)),
        (0x03, 2, _) => format!("lw {}, {}({})", reg(rd), imm_i, reg(rs1)),
        (0x03, 4, _) => format!("lbu {}, {}({})", reg(rd), imm_i, reg(rs1)),
        (0x03, 5, _) => format!("lhu {}, {}({})", reg(rd), imm_i, reg(rs1)),
        (0x23, 0, _) => format!("sb {}, {}({})", reg(rs2), imm_s, reg(rs1)),
        (0x23, 1, _) => format!("sh {}, {}({})", reg(rs2), imm_s, reg(rs1)),
        (0x23, 2, _) => format!("sw {}, {}({})", reg(rs2), imm_s, reg(rs1)),
        (0x13, 0, _) if w == 0x0000_0013 => "nop".to_string(),
        (0x13, 0, _) => format!("addi {}, {}, {}", reg(rd), reg(rs1), imm_i),
        (0x13, 7, _) => format!("andi {}, {}, {}", reg(rd), reg(rs1), imm_i),
        (0x13, 6, _) => format!("ori {}, {}, {}", reg(rd), reg(rs1), imm_i),
        (0x13, 4, _) => format!("xori {}, {}, {}", reg(rd), reg(rs1), imm_i),
        (0x13, 1, _) => format!("slli {}, {}, {}", reg(rd), reg(rs1), rs2),
        (0x13, 5, _) => format!("srli {}, {}, {}", reg(rd), reg(rs1), rs2),
        (0x33, 0, 0x00) => format!("add {}, {}, {}", reg(rd), reg(rs1), reg(rs2)),
        (0x33, 0, 0x20) => format!("sub {}, {}, {}", reg(rd), reg(rs1), reg(rs2)),
        (0x33, 7, 0x00) => format!("and {}, {}, {}", reg(rd), reg(rs1), reg(rs2)),
        (0x33, 6, 0x00) => format!("or {}, {}, {}", reg(rd), reg(rs1), reg(rs2)),
        (0x33, 4, 0x00) => format!("xor {}, {}, {}", reg(rd), reg(rs1), reg(rs2)),
        (0x33, 1, 0x00) => format!("sll {}, {}, {}", reg(rd), reg(rs1), reg(rs2)),
        (0x33, 5, 0x00) => format!("srl {}, {}, {}", reg(rd), reg(rs1), reg(rs2)),
        (0x33, 0, 0x01) => format!("mul {}, {}, {}", reg(rd), reg(rs1), reg(rs2)),
        (0x33, 4, 0x01) => format!("divu {}, {}, {}", reg(rd), reg(rs1), reg(rs2)),
        (0x33, 6, 0x01) => format!("remu {}, {}, {}", reg(rd), reg(rs1), reg(rs2)),
        (0x0f, _, _) => "fence".to_string(),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::tenstorrent::riscv;
    use crate::codegen::tenstorrent::riscv::reg as r;

    #[test]
    fn every_encoder_round_trips() {
        let words = [
            riscv::addi(r::T0, r::A0, -17),
            riscv::add(r::T1, r::T2, r::T3),
            riscv::sub(r::T1, r::T2, r::T3),
            riscv::and(r::T1, r::T2, r::T3),
            riscv::or(r::T1, r::T2, r::T3),
            riscv::xor(r::T1, r::T2, r::T3),
            riscv::slli(r::T1, r::T2, 7),
            riscv::andi(r::T1, r::T2, 0xff),
            riscv::srli(r::T1, r::T2, 7),
            riscv::mul(r::T1, r::T2, r::T3),
            riscv::divu(r::T1, r::T2, r::T3),
            riscv::remu(r::T1, r::T2, r::T3),
            riscv::lui(r::T0, 0x12345),
            riscv::auipc(r::T0, 0x1),
            riscv::lw(r::T0, r::A6, 32),
            riscv::lbu(r::T0, r::A6, 32),
            riscv::lhu(r::T0, r::A6, 32),
            riscv::sw(r::A6, r::T0, 32),
            riscv::sh(r::A6, r::T0, 32),
            riscv::sb(r::A6, r::T0, 32),
            riscv::beq(r::T0, r::T1, 8),
            riscv::bne(r::T0, r::T1, -8),
            riscv::blt(r::T0, r::T1, 8),
            riscv::bge(r::T0, r::T1, 8),
            riscv::bltu(r::T0, r::T1, 8),
            riscv::bgeu(r::T0, r::T1, 8),
            riscv::jal(r::RA, 16),
            riscv::jalr(r::ZERO, r::RA, 0),
            riscv::NOP,
            riscv::FENCE,
            riscv::RET,
        ];
        for w in words {
            let text = word_text(w);
            assert!(!text.starts_with(".word"), "word {w:#010x} did not decode: {text}");
        }
    }

    #[test]
    fn operands_are_named() {
        assert_eq!(word_text(riscv::addi(r::T0, r::A0, -17)), "addi t0, a0, -17");
        assert_eq!(word_text(riscv::li(r::T0, 0x1234_5678).0[0]), "lui t0, 0x12345");
        assert_eq!(word_text(riscv::RET), "ret");
        assert_eq!(word_text(riscv::NOP), "nop");
        assert_eq!(word_text(riscv::lw(r::T0, r::A6, 32)), "lw t0, 32(a6)");
        assert_eq!(word_text(riscv::bne(r::T0, r::T1, -8)), "bne t0, t1, -8");
    }
}
