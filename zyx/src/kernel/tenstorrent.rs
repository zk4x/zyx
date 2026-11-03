// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tenstorrent kernel passes: tile the IR, lower it to native ops, and
//! verify the reader/compute/writer stream shape.
//!
//! Pipeline order: `tt_tile` -> `tt_reindex` -> `tt_dataflow_select` ->
//! `tt_dst_reuse` -> `tt_reg_sync` -> `verify` -> `tt_verify`. The passes are
//! public so external pass authors can reuse or replace stages. Calling them
//! out of order is a loud panic, never silent corruption.
//!
//! `tt_dataflow_select` is the phase boundary: it selects the native instruction
//! form for every value and gives each result its own tile (no reuse), so
//! the result is a legal machine program with maximal register pressure.
//! `tt_dst_reuse` is an optimization over that form (coalescing, in-place
//! merging, store/load forwarding). Nothing generic runs after it.
//!
//! Lowering is SSA over DST: every native op writes a fresh
//! `Op::Storage { scope: Register }` and the `TTOp` carries that storage
//! OpId as its `slot`. A separate pass (`tt_dst_reuse`) assigns the 8-16
//! physical tile registers by liveness and rewrites operands, so lowering
//! never needs a refcounted slot map and codegen only maps a physical
//! storage to its DST row. Coalescing an input and output onto one row is
//! also the reuse pass's job, not a lowering special case.

use super::split::SplitMode;
use crate::DType;
use crate::Map;
use crate::Set;
use crate::dtype::Constant;
use crate::kernel::Pat;
use crate::kernel::{
    BOp, Bindings, Kernel, MemScope, Op, OpId, ParamKind, RangeKind, TTOp, TTSfpConst, TTSfpInsn, TTSfpLoad, TTSfpStore, TileDim,
    UOp,
};
use crate::shape::Dim;

/// The register hand-off's semaphores (`tt_reg_sync`) are per *edge*, assigned
/// in `tt_reg_sync` from the ordered `(producer role, consumer role)` pairs a
/// kernel needs: one counter per edge, never one per direction. `SEMWAIT` is a
/// level test, not a reservation, so two consumer roles on one counter both
/// pass on a single post and the second `SEMGET` underflows (ttsim's
/// `tensix_semget` message prints the *counter value*, `sem=0`, not the
/// index). Indices are ours, not hardware: the Sync Unit has eight flat 4-bit
/// counters and no idea what any of them means.
/// `Before` codes, as codegen's `tt_before` numbering: the pipeline classes a
/// `SEMWAIT` holds back until its semaphore was posted.
const TT_BEFORE_UNPACKER: u8 = 0;
const TT_BEFORE_PACKER: u8 = 1;
/// `Before::EVERYTHING`. T1 has two DST-writing units (Matrix and SFPU) and no
/// single `Before` names both, so a take on T1 blocks every class.
const TT_BEFORE_ALL: u8 = 7;

/// SFPU row-op operands, in the `u8` the `TTSfp*` ops carry: the eight
/// writable lane registers and the two read-only constants (`LReg.md`), and
/// the two `Dst` format codes (`tt_isa::sfpu::mod0_fmt`). Narrowed from the
/// ISA's `u32` once, here, instead of at every use.
const L0: u8 = tt_isa::sfpu::LReg::L0.index() as u8;
const L1: u8 = tt_isa::sfpu::LReg::L1.index() as u8;
const L2: u8 = tt_isa::sfpu::LReg::L2.index() as u8;
const L3: u8 = tt_isa::sfpu::LReg::L3.index() as u8;
const L4: u8 = tt_isa::sfpu::LReg::L4.index() as u8;
const L5: u8 = tt_isa::sfpu::LReg::L5.index() as u8;
const L6: u8 = tt_isa::sfpu::LReg::L6.index() as u8;
const L7: u8 = tt_isa::sfpu::LReg::L7.index() as u8;
const LREG_ZERO: u16 = tt_isa::sfpu::LREG_ZERO as u16;
const LREG_ONE: u16 = tt_isa::sfpu::LREG_ONE as u16;
const FP32: u8 = tt_isa::sfpu::mod0_fmt::FP32 as u8;
const INT32: u8 = tt_isa::sfpu::mod0_fmt::INT32 as u8;

/// The lane registers one SFPU instruction reads and writes, as 16-bit sets
/// over `LReg::index()` (`bit i` is `L[i]`).
///
/// `InstructionDef` carries field layouts but no read/write sets, so this is
/// the backend's model: a fusion pass needs it to carry a value across
/// sequences in a lane register and to rename around collisions. Fields are
/// resolved by the ISA's own names, never by position, and the same form
/// `tt_sfp_encode` picks (Blackhole's, not the superseded Wormhole layout).
///
/// A false read is harmless (it only loses a fusion opportunity); a false
/// write would corrupt a value, so the arms' own usage is what the table
/// states. `SFPSHFT2`/`SFPTRANSP` (the reduce programs only) have implicit
/// operands, so they are conservatively all-writable-registers on both sides.
fn tt_sfp_reg_fields(mnemonic: &str) -> (&'static [&'static str], &'static [&'static str]) {
    // `(read fields, written fields)`, by the ISA's own field names. This is
    // the *only* place that says which fields are register operands: an
    // instruction can have a field named `VD` that is not one (`SFPSETCC`'s
    // must stay 0, or the simulator answers `tensix_sfpsetcc: lreg_dest=1`).
    match mnemonic {
        // `d = f(c)`.
        "SFPABS" | "SFPCAST" | "SFPEXEXP" | "SFPEXMAN" | "SFPSETEXP" | "SFPSETSGN" | "SFPARECIP" | "SFPNOT" | "SFPSETMAN"
        | "SFPDIVP2" | "SFPXOR" | "SFPMOV" => (&["VC"], &["VD"]),
        // `d = a & b` / `a | b` (the `USE_VB` layout).
        "SFPAND" | "SFPOR" => (&["VB", "VC"], &["VD"]),
        // In place: `d = c +/- d`, `d = d <<</>>> c`.
        "SFPIADD" | "SFPSHFT" => (&["VC", "VD"], &["VD"]),
        // `d = a * b + c` and its two-operand forms.
        "SFPMAD" | "SFPADD" | "SFPMUL" | "SFPMUL24" => (&["VA", "VB", "VC"], &["VD"]),
        // Predicates write the lane flags, not a register; their `VD` is not
        // an operand.
        "SFPGT" | "SFPLE" => (&["VC", "VD"], &[]),
        "SFPSETCC" => (&["VC"], &[]),
        "SFPPUSHC" | "SFPPOPC" | "SFPCOMPC" | "SFPENCC" | "SFPNOP" => (&[], &[]),
        "SFPSHFT2" | "SFPTRANSP" => (&["VB", "VC", "VD"], &["VD"]),
        other => todo!("tt_sfp_reg_fields: {other} has no register model"),
    }
}

fn tt_sfp_regs(op: u8, args: &[u16; 5]) -> (Set<u16>, Set<u16>) {
    let Some(def) = crate::codegen::tenstorrent::tt_sfp_def(op) else {
        todo!("tt_sfp_regs: no ISA instruction has opcode {op:#x}");
    };
    let (reads, writes) = tt_sfp_reg_fields(def.mnemonic());
    let value = |name: &str| def.fields().iter().position(|f| f.name() == name).map(|i| args[i]);
    let mut read = Set::default();
    let mut write = Set::default();
    for name in reads {
        if let Some(v) = value(name) {
            read.insert(v);
        }
    }
    for name in writes {
        if let Some(v) = value(name) {
            write.insert(v);
        }
    }
    (read, write)
}

/// Instruction words a row op emits, counted as codegen counts them: a
/// `SFPLOADI` is one word for a BF16 immediate and two for a full FP32 one.
fn tt_row_words(op: &Op) -> u32 {
    match op {
        Op::TT(TTOp::SfpConst(c)) => {
            if c.bits & 0xffff == 0 {
                1
            } else {
                2
            }
        }
        Op::TT(TTOp::SfpLoad(_) | TTOp::SfpStore(_) | TTOp::Sfp(_)) => 1,
        other => panic!("a row loop body holds only row ops, got {other:?}"),
    }
}

/// The lane registers a row op reads and writes, as sets over the register
/// number (`u16`: a candidate cone is numbered in a temporary space that is
/// packed back into the ISA's 8-bit register field before it is emitted).
/// `tt_sfpu_fusion` uses it to prove that the producer's result register
/// survives the consumer's sequence.
fn tt_row_regs(op: &Op) -> (Set<u16>, Set<u16>) {
    match op {
        Op::TT(TTOp::Sfp(insn)) => tt_sfp_regs(insn.op, &insn.args),
        Op::TT(TTOp::SfpLoad(load)) => (Set::default(), Set::from_iter([u16::from(load.reg)])),
        Op::TT(TTOp::SfpStore(store)) => (Set::from_iter([u16::from(store.reg)]), Set::default()),
        Op::TT(TTOp::SfpConst(k)) => (Set::default(), Set::from_iter([u16::from(k.reg)])),
        other => panic!("tt_row_regs: {other:?} is not a row op"),
    }
}

/// Greedy linear scan of the eight general-purpose lane registers over a
/// straight-line body, given each instruction's `(reads, writes)` in *virtual*
/// space (register numbers `>= 16`; `0..8` are ignored, they are already
/// physical). Returns the assignment indexed by virtual name, or `None` when
/// nine values are live at once.
///
/// One implementation, used twice: `tt_fuse_cone` runs it on the body it is
/// about to build (and forms a shorter cone when it does not fit), and
/// `tt_lreg_alloc` runs the same scan on the body it emitted (where `None` is
/// a bug). Liveness is exact because a row body is one block: every virtual
/// has exactly one interval, `def -> last use`, and a register frees when its
/// interval ends.
fn tt_alloc_regs(body: &[(Set<u16>, Set<u16>)]) -> Option<Map<u16, u8>> {
    let mut def: Map<u16, i64> = Map::default();
    let mut last: Map<u16, i64> = Map::default();
    let mut first_use: Map<u16, i64> = Map::default();
    let mut any = false;
    for (k, (reads, writes)) in body.iter().enumerate() {
        for &r in writes {
            def.entry(r).or_insert(k as i64);
            last.insert(r, k as i64);
            any |= r >= 16;
        }
        for &r in reads {
            first_use.entry(r).or_insert(k as i64);
            last.insert(r, k as i64);
        }
    }
    let mut assign: Map<u16, u8> = Map::default();
    if !any {
        return Some(assign);
    }
    // Read before any write: the value is live from the body's start, and the
    // interval must cover that read.
    for (&r, &use_at) in &first_use {
        if def.get(&r).is_none_or(|&d| use_at < d) {
            def.insert(r, -1);
        }
    }
    let mut order: Vec<u16> = last.keys().copied().filter(|r| *r >= 16).collect();
    order.sort_by_key(|r| (def.get(r).copied().unwrap_or(-1), *r));
    let mut active: Vec<(u16, u8, i64)> = Vec::new(); // (vreg, phys, last)
    for r in order {
        let start = def.get(&r).copied().unwrap_or(-1);
        active.retain(|&(_, _, l)| l >= start);
        let used: u16 = active.iter().fold(0, |m, &(_, p, _)| m | 1u16 << p);
        let p = (0..8u8).find(|p| used & (1u16 << p) == 0)?;
        assign.insert(r, p);
        active.push((r, p, last.get(&r).copied().unwrap_or(start)));
    }
    Some(assign)
}

/// Pack a straight-line body's temporary register names (`>= 16`) into the
/// ISA's 8-bit register field by liveness. Two names share a final name when
/// their `def -> last use` intervals are disjoint, so a body uses as many names
/// as it has values live at once, not as many as it defines across its units.
/// Returns the map indexed by temporary name and the first unused final name.
fn tt_pack_reg_names(body: &[(Set<u16>, Set<u16>)], names: u16) -> (Vec<u16>, u16) {
    let n = usize::from(names);
    let mut pack = vec![u16::MAX; n];
    let mut def = vec![-1i64; n];
    let mut last = vec![-1i64; n];
    let mut first_use = vec![-1i64; n];
    for (k, (reads, writes)) in body.iter().enumerate() {
        for &r in writes {
            let r = usize::from(r);
            if r < 16 {
                continue;
            }
            if def[r] < 0 {
                def[r] = k as i64;
            }
            last[r] = k as i64;
        }
        for &r in reads {
            let r = usize::from(r);
            if r < 16 {
                continue;
            }
            if first_use[r] < 0 {
                first_use[r] = k as i64;
            }
            if last[r] < k as i64 {
                last[r] = k as i64;
            }
        }
    }
    // Read before any write: live from the body's start.
    for r in 16..n {
        if first_use[r] >= 0 && (def[r] < 0 || first_use[r] < def[r]) {
            def[r] = -1;
        }
    }
    let mut order: Vec<usize> = (16..n).filter(|r| last[*r] >= 0).collect();
    order.sort_by_key(|r| (def[*r], *r));
    let mut active: Vec<(u16, i64)> = Vec::new(); // (final name, last use)
    let mut next = 16u16;
    for r in order {
        let start = def[r];
        let reuse = active.iter().enumerate().filter(|(_, (_, l))| *l < start).min_by_key(|(_, (name, _))| *name).map(|(i, _)| i);
        let name = match reuse {
            Some(i) => active.remove(i).0,
            None => {
                let name = next;
                next += 1;
                name
            }
        };
        pack[r] = name;
        active.push((name, last[r]));
    }
    (pack, next)
}

/// Rename lane register `from` to `to` in one row op's register operands.
/// Only `VA`/`VB`/`VC`/`VD` are registers; `Imm12`/`Imm1`/`Mod1` are not, so
/// the fields are matched by the ISA's own names.
fn tt_row_rename(op: &mut Op, from: u8, to: u8) {
    match op {
        Op::TT(TTOp::Sfp(insn)) => {
            let Some(def) = crate::codegen::tenstorrent::tt_sfp_def(insn.op) else {
                todo!("tt_row_rename: no ISA instruction has opcode {:#x}", insn.op);
            };
            let (reads, writes) = tt_sfp_reg_fields(def.mnemonic());
            for name in reads.iter().chain(writes.iter()) {
                let Some(i) = def.fields().iter().position(|f| f.name() == *name) else {
                    continue;
                };
                if insn.args[i] == u16::from(from) {
                    insn.args[i] = u16::from(to);
                }
            }
        }
        Op::TT(TTOp::SfpLoad(load)) => {
            if load.reg == from {
                load.reg = to;
            }
        }
        Op::TT(TTOp::SfpStore(store)) => {
            if store.reg == from {
                store.reg = to;
            }
        }
        Op::TT(TTOp::SfpConst(k)) => {
            if k.reg == from {
                k.reg = to;
            }
        }
        other => todo!("tt_row_rename: {other:?} is not a row op"),
    }
}

/// Rename every lane register of one row op through `map`, looking each
/// operand's *current* value up exactly once. Unlike repeated [`tt_row_rename`]
/// calls, a source that is itself a rename target cannot chain: the whole body
/// is remapped in one pass.
fn tt_row_remap(op: &mut Op, map: &[u16]) {
    let lookup = |v: u16| map.get(usize::from(v)).copied().unwrap_or(u16::MAX);
    match op {
        Op::TT(TTOp::Sfp(insn)) => {
            let Some(def) = crate::codegen::tenstorrent::tt_sfp_def(insn.op) else {
                todo!("tt_row_remap: no ISA instruction has opcode {:#x}", insn.op);
            };
            let (reads, writes) = tt_sfp_reg_fields(def.mnemonic());
            for name in reads.iter().chain(writes.iter()) {
                let Some(i) = def.fields().iter().position(|f| f.name() == *name) else {
                    continue;
                };
                let to = lookup(insn.args[i]);
                if to != u16::MAX {
                    insn.args[i] = to;
                }
            }
        }
        Op::TT(TTOp::SfpLoad(load)) => {
            let to = lookup(u16::from(load.reg));
            if to != u16::MAX {
                load.reg = to as u8;
            }
        }
        Op::TT(TTOp::SfpStore(store)) => {
            let to = lookup(u16::from(store.reg));
            if to != u16::MAX {
                store.reg = to as u8;
            }
        }
        Op::TT(TTOp::SfpConst(k)) => {
            let to = lookup(u16::from(k.reg));
            if to != u16::MAX {
                k.reg = to as u8;
            }
        }
        other => todo!("tt_row_remap: {other:?} is not a row op"),
    }
}

/// One phase-2 pattern and the native lowering it emits. The list is
/// ordered biggest-first; the recursive walk tries each pattern at a value
/// root and stops at the first emitter that succeeds. The SFPU variant is
/// data, not a per-op arm: adding an elementwise op is a table entry.
#[derive(Clone, Copy)]
enum TtEmit {
    /// Elementwise SFPU: `arity` 1 or 2 DST inputs, one DST result.
    Sfpu { kind: u32, arity: u8 },
    /// Elementwise SFPU with one scalar-constant operand: the constant is
    /// filled into a tile first, then the same arm as [`TtEmit::Sfpu`] runs.
    /// `tt_unfold`'s coefficients reach the SFPU this way; a later pass can
    /// fold the fill into the arm's lane-register immediate.
    SfpuConst { kind: u32 },
    /// Hardware Matrix Unit: CB inputs, DST result.
    Matmul,
    /// Structural tile reduce: DST input, DST result.
    Reduce(TileDim),
}

/// What a tile `Copy` moves between, and hence which stream runs it. The
/// movers (`Publish`, `Drain`) are not part of the register hand-off: their
/// traffic is a flat list issued around the compute program, not a role.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TtCopy {
    /// `Param -> Circular`: the reader (B mover).
    Publish,
    /// `Circular -> Register`: the unpacker (T0).
    Unpack,
    /// `Register -> Circular`: the packer (T2).
    Pack,
    /// `Circular -> Param`: the writer (NC mover).
    Drain,
}

/// zyx unary op -> metalium SFPU kind. `None` means no native kind yet;
/// the value stays high-level instead of being mis-lowered.
fn uop_kind(uop: UOp) -> Option<u32> {
    use crate::backend::tenstorrent::kern::sfpu::ops::kind_sfpu as sf;
    Some(match uop {
        UOp::Neg => sf::NEG,
        UOp::Not => sf::BOOL_NOT,
        UOp::BitNot => sf::INT_NOT,
        // `Exp2`, `Log2`, `Sin` and `Cos` are IR from `tt_unfold` on, so they
        // never reach the pattern table and name no kind.
        UOp::Exp2 | UOp::Log2 | UOp::Sin | UOp::Cos => return None,
        UOp::Reciprocal => sf::RECIP,
        UOp::Sqrt => sf::SQRT,
        UOp::Rsqrt => sf::RSQRT,
        UOp::Floor => sf::FLOOR,
        UOp::Trunc => sf::TRUNC,
        UOp::Abs => sf::ABS,
    })
}

/// zyx binary op -> metalium SFPU kind. The same table covers float,
/// integer, and comparison kinds; `None` is "no native kind yet".
fn bop_kind(bop: BOp) -> Option<u32> {
    use crate::backend::tenstorrent::kern::kind;
    use crate::backend::tenstorrent::kern::sfpu::ops::kind_sfpu as sf;
    Some(match bop {
        BOp::Add => kind::ADD,
        BOp::Sub => kind::SUB,
        BOp::Mul => kind::MUL,
        BOp::Div => sf::DIV,
        BOp::Pow => sf::POW,
        BOp::Mod => sf::REM,
        BOp::Cmplt => sf::LT,
        BOp::Cmpgt => sf::GT,
        BOp::Cmpge => sf::GE,
        // Tensor-tensor max has no direct SFPU kind. A reduce by max is not
        // an elementwise op at all: it is the `TileDim` pattern's fold.
        BOp::Max => return None,
        BOp::Or | BOp::BitOr => sf::INT_OR,
        BOp::And | BOp::BitAnd => sf::INT_AND,
        BOp::BitXor => sf::INT_XOR,
        BOp::BitShiftLeft => sf::INT_SHL,
        BOp::BitShiftRight => sf::INT_SHR,
        BOp::NotEq => sf::NE,
        BOp::Eq => sf::EQ,
    })
}

/// Every zyx unary op. `uop_kind` filters out the ones without a native
/// SFPU kind, so the table stays data-only.
const UOPS: &[UOp] = &[
    UOp::Neg,
    UOp::Not,
    UOp::BitNot,
    UOp::Exp2,
    UOp::Log2,
    UOp::Reciprocal,
    UOp::Sqrt,
    UOp::Rsqrt,
    UOp::Sin,
    UOp::Cos,
    UOp::Floor,
    UOp::Trunc,
    UOp::Abs,
];

/// Every zyx binary op. `bop_kind` filters out the ones without a native
/// SFPU kind.
const BOPS: &[BOp] = &[
    BOp::Add,
    BOp::Sub,
    BOp::Mul,
    BOp::Div,
    BOp::Pow,
    BOp::Mod,
    BOp::Cmplt,
    BOp::Cmpgt,
    BOp::Cmpge,
    BOp::Max,
    BOp::Or,
    BOp::And,
    BOp::BitXor,
    BOp::BitOr,
    BOp::BitAnd,
    BOp::BitShiftLeft,
    BOp::BitShiftRight,
    BOp::NotEq,
    BOp::Eq,
];

/// State for the recursive phase-2 lowering. `rcs` is the original
/// high-level use count: a computed value with more than one use is packed
/// to a Circular CB, because DST is single-use scratch and multi-use values
/// live in L1 (the Tensix dataflow model). `materialized` maps that value
/// to its CB; every consumer unpacks from it. The redundant first unpack
/// is deliberate — a later pass can coalesce it with the producer's DST
/// row without complicating lowering.
struct TtLowering<'a> {
    pats: &'a [(Pat, TtEmit)],
    /// The pattern each value matched, and its bindings, decided over the
    /// tiled IR before anything lowers. Matching cannot be redone later: a
    /// value's `Op` becomes its last row op when it lowers, and `shape` then
    /// reads the slot's buffer length, not the tile's.
    matched: &'a Map<OpId, (usize, Bindings)>,
    prologue: OpId,
    reader_cb_of: &'a Map<OpId, OpId>,
    rcs: &'a Map<OpId, u32>,
    materialized: Map<OpId, OpId>,
    /// One DST tile per reader CB, unpacked at its first use: otherwise every
    /// use of the same value copies it in again, one `UNPACKED` hand-off each,
    /// and a 4-bit semaphore cannot carry that many.
    unpacked: Map<OpId, OpId>,
    /// Matmul feeder index patterns (A plain, B K-scaled).
    pat_a: Pat,
    pat_b: Pat,
}

impl Kernel {
    /// Section-split constructor: end of the reader section. Section
    /// assignment is a placement decision, so markers are builder
    /// ops — render only splits output at them, never derives them.
    pub fn tt_end_reader(&mut self) -> OpId {
        self.push_back(Op::TT(TTOp::EndReader))
    }

    /// Section-split constructor: end of the compute section. Same
    /// placement rules as [`Kernel::tt_end_reader`].
    pub fn tt_end_compute(&mut self) -> OpId {
        self.push_back(Op::TT(TTOp::EndCompute))
    }

    /// Tenstorrent tiling: pad to multiples of 32 and unroll loops
    /// and group ranges to multiples of 32. No special tiling or
    /// vectorization passes: the 32-wide `Unroll` lanes plus composed
    /// indices are the tile IR the backend matchers read.
    pub fn tt_tile(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        self.default_epilogue();
        self.default_epilogue();
        // Gather loops, pad to 32, split unroll-32 FIRST: split's
        // loop arm allows exactly one acc store per body, so it must
        // run before coarsening replicates the body. pad_to pads
        // lengths only on TT (no guards); already-multiples are
        // untouched. Non-reduce loops are a no-op inside (no acc
        // found); anything else keeps its shape for later cuts.
        let mut pad_loops: Vec<OpId> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if matches!(x, Op::Loop { .. }) {
                pad_loops.push(op);
            }
        }
        for l in pad_loops {
            self.pad_to(l, 32);
            let len = match self.ops[l].op {
                Op::Loop { len, .. } => len,
                _ => unreachable!("pad_to preserves the Loop variant"),
            };
            // resolve_const evaluates const-expr lengths on demand, so
            // freshly padded (unfolded) round-ups resolve directly.
            if self.resolve_const(len).and_then(|c| c.as_dim()).is_some_and(|l| l % 32 == 0) {
                self.split(l, SplitMode::Unroll, 32);
            }
        }

        self.default_epilogue();
        self.default_epilogue();
        // Gather group ranges, pad to 32, coarsen LAST: duplicate
        // the vectorized body, one lane per row. Anything else
        // (symbolic, indivisible) is skipped, never fatal. The device
        // grid is 2D: one group range is pre-split into two nested
        // groups below, two proceed as-is, more than two is a later
        // increment.
        //
        // The contract (NOTES, "Tiling: the search explores, `tt_tile`
        // completes"): the search is free to tile, `opt_split` never splits
        // past what the device can hold, and this pass **fills the remaining
        // gaps** — it adds the tile axes the search did not, and touches
        // nothing that is already tiled. An access's index rank is the number
        // of tile axes already present.
        let elementwise = !self.iter_unordered().any(|(_, op)| matches!(op, Op::Reduce { .. }));
        let axes = self.device_info().tile.len();
        // `Op::Reduce` is not a "reduce kernel": `linearize` turns a reduce
        // into a `Loop` with a scalar accumulator, and a `Reduce` only exists
        // here once `split` has unrolled such a loop into lanes. That case
        // keeps its extra axes — the rank-3 cube is deliberate, and
        // `tt_reindex` rewrites it for the movers — so it splits every group.
        let rank = if elementwise {
            self.iter_unordered().fold(0, |rank, (_, op)| {
                let idx = match *op {
                    Op::Load { src_idx, .. } => src_idx,
                    Op::Store { dst_idx, .. } => dst_idx,
                    Op::Copy { src_idx, .. } => src_idx,
                    _ => return rank,
                };
                if idx.is_null() {
                    rank
                } else {
                    rank.max(self.shape(idx).len())
                }
            })
        } else {
            0
        };
        let mut pad_groups: Vec<OpId> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if matches!(x, Op::Range { kind: RangeKind::Group(_), .. }) {
                pad_groups.push(op);
            }
        }
        if pad_groups.len() > 2 {
            todo!("tt_tile: more than two group ranges");
        }
        if pad_groups.len() == 1 && rank == 0 {
            // Only a **fully scalar** access needs a 2-D grid out of one group
            // range: split it into two nested groups of roughly equal size
            // (user spec — no loop check, no length threshold), then pad and
            // split each normally below. An access that already carries a lane
            // axis must not be pre-split: the group then is the *other*
            // dimension, and pre-splitting it double-counts that dimension
            // (`[32, 32]` unary: one 32-lane unroll over rows plus one group
            // over columns; the pre-split made a second grid axis, then a
            // rank-3 load `tt_dataflow_select` bails on).
            let g = pad_groups[0];
            self.pad_to(g, 32);
            let len = match self.ops[g].op {
                Op::Range { kind: RangeKind::Group(len), .. } => self.resolve_const(len).and_then(|c| c.as_dim()),
                _ => unreachable!("pad_to preserves the Group range"),
            };
            if let Some(len) = len {
                // Largest divisor at or below sqrt(len): the two
                // factors are as equal as possible.
                let mut f = (len as f64).sqrt() as Dim;
                while f > 1 && len % f != 0 {
                    f -= 1;
                }
                if f > 1 {
                    let factor = u16::try_from(f).expect("tt_tile: group split factor overflows u16");
                    // The inner group takes a new axis: highest group
                    // axis + 1 (user spec; verify requires unique group
                    // axes, and split reuses the axis). TT-local: split
                    // itself is generic across backends.
                    let mut max_group_axis = 0u32;
                    for (_, x) in self.iter_unordered() {
                        if let Op::Range { axis, kind: RangeKind::Group(_) } = x {
                            max_group_axis = max_group_axis.max(*axis);
                        }
                    }
                    let inner = self.split(g, SplitMode::Group, factor);
                    if !inner.is_null() {
                        let new_axis = max_group_axis.checked_add(1).expect("tt_tile: group axis overflows u32");
                        match &mut self.ops[inner].op {
                            Op::Range { axis, kind: RangeKind::Group(_) } => *axis = new_axis,
                            _ => unreachable!("split Group returns the inner group range"),
                        }
                    }
                }
            }
            pad_groups.clear();
            for (op, x) in self.iter_unordered() {
                if matches!(x, Op::Range { kind: RangeKind::Group(_), .. }) {
                    pad_groups.push(op);
                }
            }
        }
        // A group already counted in **tiles** is the one the search vectorized
        // along: its term is `32 * g + lane`, with `lane` an `Unroll` of 32.
        // That group is the dimension's grid axis and must be left as it is
        // (padding 8 column tiles to 32 would turn them into a lane axis and
        // leave the row group an element grid: `[10, 256]` unary, grid
        // `[10, 1]`, one 4 KB page per row). The other group counts
        // **elements**; it is the one padded to 32 and split into the axis
        // being added. A bare `g * 32` is not the test — 32 is also a legal
        // row stride (`[10, 32]`'s rows are `g * 32`), and what marks a
        // vectorized dimension is the lane **added** to the scaled group.
        let mut chunked: Vec<OpId> = Vec::new();
        if elementwise {
            for (_, op) in self.iter_unordered() {
                let Op::Binary { x, y, bop: BOp::Add } = *op else { continue };
                for (scaled, lane) in [(x, y), (y, x)] {
                    if !matches!(self.ops[lane].op, Op::Unroll { .. }) {
                        continue;
                    }
                    let Op::Binary { x: m, y: n, bop: BOp::Mul } = self.ops[scaled].op else {
                        continue;
                    };
                    for (g, c) in [(m, n), (n, m)] {
                        if matches!(self.ops[g].op, Op::Range { kind: RangeKind::Group(_), .. })
                            && self.resolve_const(c).and_then(Constant::as_dim) == Some(32)
                        {
                            chunked.push(g);
                        }
                    }
                }
            }
        }
        pad_groups.sort_by_key(|g| match self.ops[*g].op {
            Op::Range { axis, .. } => axis,
            _ => unreachable!("pad_groups holds group ranges"),
        });
        let mut need = if elementwise {
            axes.saturating_sub(rank)
        } else {
            pad_groups.len()
        };
        for g in pad_groups {
            if need == 0 {
                break;
            }
            if chunked.contains(&g) {
                continue;
            }
            self.pad_to(g, 32);
            let len = match self.ops[g].op {
                Op::Range { kind: RangeKind::Group(len), .. } => len,
                _ => unreachable!("pad_to preserves the Group range"),
            };
            if self.resolve_const(len).and_then(|c| c.as_dim()).is_some_and(|l| l % 32 == 0) {
                self.split(g, SplitMode::Unroll, 32);
                need -= 1;
            }
        }

        self.default_epilogue();
        self.default_epilogue();
    }

    /// General TT lowering + register allocation (matmul, reduce,
    /// sigmoid/silu carried). Traffic is part of emission: the pass
    /// matches compute, creates the CBs (DRAM-backed reader/writer CBs
    /// plus interior CBs), emits the DRAM↔CB movement as `TTOp::Copy`
    /// (explicit bytes, scalar base indices — unrolls stripped), places
    /// `EndReader`/`EndCompute`, lowers each match to a structured LLK
    /// op over storages only, and allocates the Register storages in
    /// the same step. Render only substitutes numbers; stripped unrolls
    /// dangle into DCE.
    ///
    /// Lowered values chain through DST: a reduce reads its feeder's
    /// storage and writes a new one; sigmoid/silu run in place on their
    /// feeder's storage; matmul reads CBs and writes a fresh DST row.
    /// Only a value that reaches a DRAM output is packed (phase 3).
    /// Anything unmatched is left in place for later increments.
    /// Canonicalize a single-tile reduce's traffic to the physical tile
    /// pitch.
    ///
    /// `tt_tile` builds the reader index from the *logical* layout: a
    /// reduce over a contiguous axis loads a rank-3 cube (`[32, 32, 32]`
    /// for a whole tile) and a reduce over a tiled rank-2 value loads at
    /// the logical pitch. Neither is what the NOC moves: the reader reads
    /// one 32x32 tile from its base, and the reduce wants the reduced axis
    /// as the innermost tile lane.
    ///
    /// This pass recognizes the one template zyx emits today — exactly one
    /// `Reduce` fed by exactly one DRAM load and drained by exactly one
    /// DRAM store — and rewrites both indices to `row * 32 + lane`, so the
    /// load publishes one tile and the store drains one row. The rewrite is
    /// traffic-only: it does not care whether the fold is a sum or a maximum,
    /// and `tt_dataflow_select`'s pattern table has both, so gating on `Add`
    /// here would leave every other reduce at the logical pitch and unsuited
    /// to the movers. Anything else is left untouched (multi-reduce, loops,
    /// groups, matmul cubes).
    pub fn tt_reindex(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        // Single-tile kernels only: loops and groups mean extents or
        // multi-tile traffic this rewrite does not model.
        for (_, x) in self.iter_unordered() {
            match x {
                Op::Loop { .. } => return,
                Op::Range { kind: RangeKind::Group(_), .. } => return,
                _ => {}
            }
        }
        // Exactly one reduce, one DRAM load, one DRAM store.
        let mut reduces: Vec<OpId> = Vec::new();
        let mut loads: Vec<OpId> = Vec::new();
        let mut stores: Vec<OpId> = Vec::new();
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() {
                break;
            }
            match self.ops[id].op {
                Op::Reduce { .. } => reduces.push(id),
                Op::Load { src: base, .. } => {
                    if matches!(self.ops[base].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. }) {
                        loads.push(id);
                    }
                }
                Op::Store { dst: base, .. } => {
                    if matches!(self.ops[base].op, Op::Param { kind: ParamKind::GlobalMut, .. }) {
                        stores.push(id);
                    }
                }
                _ => {}
            }
            id = self.ops[id].next;
        }
        if reduces.len() != 1 || loads.len() != 1 || stores.len() != 1 {
            return;
        }
        let root = reduces[0];
        let load = loads[0];
        let store = stores[0];
        // A reduce over the DRAM load; the store drains the reduce. The fold
        // is not this pass's business — both `Add` and `Max` have a native
        // pattern downstream, and the traffic rewrite is identical.
        let (x, k) = match &self.ops[root].op {
            Op::Reduce { x, reduce_axis, rop } if matches!(rop, BOp::Add | BOp::Max) => (*x, *reduce_axis),
            _ => return,
        };
        if x != load {
            return;
        }
        let ssrc = match &self.ops[store].op {
            Op::Store { src, .. } => *src,
            _ => return,
        };
        if ssrc != root {
            return;
        }
        // Rank-3 cube to a rank-2 result (axis drop), or a
        // logical-pitch rank-2 tile to a row (pitch substitution).
        // Anything else (matmul cubes, scalars) is not this template.
        let drop3 = self.shape(load) == vec![32, 32, 32] && self.shape(root) == vec![32, 32];
        let row2 = self.shape(load) == vec![32, 32] && self.shape(root) == vec![32];
        if !drop3 && !row2 {
            return;
        }
        let lidx = match self.ops[load].op {
            Op::Load { src_idx, .. } => src_idx,
            _ => unreachable!("tt_reindex: load lost its index"),
        };
        // Stride anatomy of the load index: the reduce lane at stride
        // 1, one row axis at stride W, and (cube only) dropped axes at
        // multiples of W. A const offset must be zero (a nonzero
        // logical offset is not a physical one); anything else bails.
        let Some(strides) = self.get_strides(lidx) else {
            return;
        };
        if !matches!(self.ops[k].op, Op::Unroll { len: 32, .. }) {
            return;
        }
        match strides.get(&k) {
            Some((32, 1)) => {}
            _ => return,
        }
        let mut rows: Vec<(OpId, Dim)> = Vec::new();
        for (var, (len, stride)) in strides.iter() {
            if *var == OpId::NULL {
                if *stride != 0 {
                    return;
                }
                continue;
            }
            if *var == k {
                continue;
            }
            if !matches!(self.ops[*var].op, Op::Unroll { len: 32, .. }) {
                return;
            }
            if *len != 32 || *stride <= 0 {
                return;
            }
            rows.push((*var, *stride));
        }
        // Keep the smallest stride above 1 as the row axis; every
        // other axis must stride a multiple of it (pure upward row
        // shift, never a new column structure).
        rows.sort_by_key(|&(_, s)| s);
        let row = match rows.as_slice() {
            [(r, w)] if row2 && *w != 32 && *w > 1 => *r,
            [(r, w), (_d, dw)] if drop3 && *w > 1 && *dw > *w && *dw % *w == 0 => *r,
            _ => return,
        };
        // Canonical traffic indices at the physical tile pitch.
        let phys = self.insert_const_idx_before(load, 32i64);
        let rk = self.insert_before(load, Op::Binary { x: row, y: phys, bop: BOp::Mul });
        let new_idx = self.insert_before(load, Op::Binary { x: rk, y: k, bop: BOp::Add });
        match &mut self.ops[load].op {
            Op::Load { src_idx, .. } => *src_idx = new_idx,
            _ => unreachable!("tt_reindex: load lost its index"),
        }
        match &mut self.ops[store].op {
            Op::Store { dst_idx, .. } => *dst_idx = row,
            _ => unreachable!("tt_reindex: store lost its index"),
        }
        self.verify();
    }

    /// The transcendentals of the Tensix Vector Unit, written in the IR
    /// instead of as row-op sequences. Runs after `tt_tile` and before
    /// `tt_dataflow_select`, while the kernel is still ordinary `Op`s, and
    /// rewrites each leaf transcendental into the arithmetic of its metalium
    /// program -- one IR op per instruction -- so the native lowering only
    /// ever sees primitives (`mul`/`add`/`sub`/`div`, the bit ops,
    /// `Bitcast`/`Cast`, comparisons and `Select`):
    ///
    /// - `Exp2`: the `exp_program` reduction (`n = round(x)` by the
    ///   `1.5 * 2^23` trick, `r = x - n`), its `e^(r ln 2)` polynomial, and
    ///   the exponent-field assembly. `x^y = exp2(y log2 x)` (`unfold_pows`)
    ///   turns `Pow` into this pair.
    /// - `Log2`: the `log_program` reduction (`m` in `[1/sqrt 2, sqrt 2)`, the
    ///   exponent by `SFPSETEXP`'s bit form), its `2 atanh(s)` series and
    ///   `e ln 2`, times `log2 e`.
    ///
    /// Both carry `exp_program`/`log_program`'s special values (overflow to
    /// `+0` or `+inf`, `log2` of a zero or denormal `-inf`, of a negative
    /// `NaN`, a NaN staying one). `Sin`/`Cos` are not unfolded yet.
    ///
    /// The IR has no `SFPMAD`, so a Horner step is a multiply and an add; the
    /// programs' error bounds still hold to within a rounding or two. A scalar
    /// constant stays an `Op::Const` operand; the native lowering fills it into
    /// a tile (`TtEmit::SfpuConst`, `tt_fill_const`), and a later pass can fold
    /// that fill into the consumer's lane-register immediate.
    pub fn tt_unfold(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        let mut ids: Vec<OpId> = Vec::new();
        for (id, x) in self.iter_unordered() {
            if matches!(x, Op::Unary { uop: UOp::Exp2 | UOp::Log2 | UOp::Sin | UOp::Cos, .. }) {
                ids.push(id);
            }
        }
        for id in ids {
            let Op::Unary { x, uop } = self.ops[id].op else {
                unreachable!("tt_unfold collected a non-`Unary` op");
            };
            let out = match uop {
                UOp::Exp2 => self.tt_exp2(id, x),
                UOp::Log2 => self.tt_log2(id, x),
                UOp::Sin => self.tt_sin_cos(id, x, false),
                UOp::Cos => self.tt_sin_cos(id, x, true),
                _ => unreachable!("tt_unfold collected {uop:?}"),
            };
            self.remap(id, out);
        }
    }

    /// `2^x` by `exp_program`: the reduction and polynomial of `e^r`, the
    /// exponent assembled into the result's bit pattern. `x < -126` is `+0`
    /// (a denormal, which the unit's arithmetic flushes), `x >= 128` is
    /// `+inf`, a NaN stays one.
    fn tt_exp2(&mut self, at: OpId, x: OpId) -> OpId {
        let konst = |k: &mut Kernel, v: f32| k.insert_before(at, Op::Const(Constant::new(v)));
        let bin = |k: &mut Kernel, x: OpId, y: OpId, bop: BOp| k.insert_before(at, Op::Binary { x, y, bop });
        let sel = |k: &mut Kernel, cond: OpId, if_true: OpId, if_false: OpId| {
            k.insert_before(at, Op::Select { cond, if_true, if_false })
        };
        let magic = konst(self, 12_582_912.0); // 1.5 * 2^23
        let lo = konst(self, -126.0);
        let hi = konst(self, 128.0);
        // Clamp only the reduction's input; the selects below restore the
        // specials, so the clamped path's value never escapes.
        let over = bin(self, x, hi, BOp::Cmpgt);
        let xc = sel(self, over, hi, x);
        let under = bin(self, xc, lo, BOp::Cmplt);
        let xc = sel(self, under, lo, xc);
        // `n = round(xc)` by the magic constant, `r = xc - n` exactly.
        let t = bin(self, xc, magic, BOp::Add);
        let nf = bin(self, t, magic, BOp::Sub);
        let r = bin(self, xc, nf, BOp::Sub);
        // `2^r = e^(r ln 2)`, `exp_program`'s Horner form.
        let ln2 = konst(self, std::f32::consts::LN_2);
        let s = bin(self, r, ln2, BOp::Mul);
        let mut d = konst(self, 1.0 / 5040.0);
        for c in [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5] {
            let cc = konst(self, c);
            let p = bin(self, d, s, BOp::Mul);
            d = bin(self, p, cc, BOp::Add);
        }
        for _ in 0..2 {
            let one = konst(self, 1.0);
            let p = bin(self, d, s, BOp::Mul);
            d = bin(self, p, one, BOp::Add);
        }
        // `bits(d) + (n << 23)`: `SFPEXEXP`/`SFPSETEXP`'s exponent add, in the
        // IR's own bit form.
        let bits = self.insert_before(at, Op::Bitcast { x: d, dtype: DType::I32 });
        let n = self.insert_before(at, Op::Cast { x: nf, dtype: DType::I32 });
        let c23 = self.insert_before(at, Op::Const(Constant::new(23i32)));
        let n23 = bin(self, n, c23, BOp::BitShiftLeft);
        let scale = bin(self, bits, n23, BOp::Add);
        let base = self.insert_before(at, Op::Bitcast { x: scale, dtype: DType::F32 });
        // The specials, each its own lanes.
        let zero = konst(self, 0.0);
        let small = bin(self, x, lo, BOp::Cmplt);
        let out = sel(self, small, zero, base);
        let inf = konst(self, f32::INFINITY);
        let large = bin(self, x, hi, BOp::Cmpge);
        let out = sel(self, large, inf, out);
        // A NaN input stays that NaN (its own payload and sign).
        let is_nan = bin(self, x, x, BOp::NotEq);
        sel(self, is_nan, x, out)
    }

    /// `log2 x` by `log_program`: the exponent and mantissa by the bit form of
    /// `SFPEXEXP`/`SFPSETEXP`, `ln m = 2 atanh(s)` with `s = f / (2 + f)`, then
    /// `e ln 2 + ln m`, times `log2 e`. `±0` and a denormal are `-inf`, a
    /// negative is `NaN`, `+inf` is `+inf`, a NaN stays one.
    fn tt_log2(&mut self, at: OpId, x: OpId) -> OpId {
        let konst = |k: &mut Kernel, v: f32| k.insert_before(at, Op::Const(Constant::new(v)));
        let icons = |k: &mut Kernel, v: i32| k.insert_before(at, Op::Const(Constant::new(v)));
        let bin = |k: &mut Kernel, x: OpId, y: OpId, bop: BOp| k.insert_before(at, Op::Binary { x, y, bop });
        let sel = |k: &mut Kernel, cond: OpId, if_true: OpId, if_false: OpId| {
            k.insert_before(at, Op::Select { cond, if_true, if_false })
        };
        let bits = self.insert_before(at, Op::Bitcast { x, dtype: DType::I32 });
        let signless = icons(self, 0x7fff_ffffu32 as i32);
        let signless = bin(self, bits, signless, BOp::BitAnd);
        // `m` in `[1, 2)` with the exponent set to 127.
        let mask = icons(self, 0x807f_ffffu32 as i32);
        let mant = bin(self, bits, mask, BOp::BitAnd);
        let exp127 = icons(self, 0x3f80_0000);
        let mant = bin(self, mant, exp127, BOp::BitOr);
        let mant = self.insert_before(at, Op::Bitcast { x: mant, dtype: DType::F32 });
        // `e = (bits >> 23) - 127`.
        let c23 = icons(self, 23);
        let e = bin(self, signless, c23, BOp::BitShiftRight);
        let c127 = icons(self, 127);
        let e = bin(self, e, c127, BOp::Sub);
        // Bring `m` below `sqrt 2`, carrying the halving into `e`.
        let sqrt2 = konst(self, std::f32::consts::SQRT_2);
        let big = bin(self, mant, sqrt2, BOp::Cmpgt);
        let half = konst(self, 0.5);
        let halved = bin(self, mant, half, BOp::Mul);
        let m = sel(self, big, halved, mant);
        let one_i = icons(self, 1);
        let e1 = bin(self, e, one_i, BOp::Add);
        let e = sel(self, big, e1, e);
        // `s = f / (2 + f)`, `f = m - 1`.
        let one = konst(self, 1.0);
        let two = konst(self, 2.0);
        let f = bin(self, m, one, BOp::Sub);
        let den = bin(self, f, two, BOp::Add);
        let s = bin(self, f, den, BOp::Div);
        // `ln m = 2(s + s*(s^2*(1/3 ...)))`.
        let s2 = bin(self, s, s, BOp::Mul);
        let mut ser = konst(self, 1.0 / 9.0);
        for c in [1.0 / 7.0, 1.0 / 5.0, 1.0 / 3.0] {
            let cc = konst(self, c);
            let p = bin(self, ser, s2, BOp::Mul);
            ser = bin(self, p, cc, BOp::Add);
        }
        let t1 = bin(self, s2, ser, BOp::Mul);
        let t2 = bin(self, s, t1, BOp::Mul);
        let t3 = bin(self, t2, s, BOp::Add);
        let lnm = bin(self, t3, t3, BOp::Add);
        // `e ln 2 + ln m`.
        let ef = self.insert_before(at, Op::Cast { x: e, dtype: DType::F32 });
        let ln2 = konst(self, std::f32::consts::LN_2);
        let eln2 = bin(self, ef, ln2, BOp::Mul);
        let ln = bin(self, lnm, eln2, BOp::Add);
        let log2e = konst(self, std::f32::consts::LOG2_E);
        let mut out = bin(self, ln, log2e, BOp::Mul);
        // The specials, in `log_program`'s order.
        let absx = self.insert_before(at, Op::Unary { x, uop: UOp::Abs });
        let min_normal = konst(self, f32::from_bits(0x0080_0000));
        let tiny = bin(self, absx, min_normal, BOp::Cmplt);
        let neg_inf = konst(self, f32::NEG_INFINITY);
        out = sel(self, tiny, neg_inf, out);
        let zero = konst(self, 0.0);
        let neg = bin(self, x, zero, BOp::Cmplt);
        // No NaN constant (`verify` refuses one, and constant folding would
        // make one of a bitcast): `sqrt` of a negative is the NaN.
        let nan = self.insert_before(at, Op::Unary { x, uop: UOp::Sqrt });
        out = sel(self, neg, nan, out);
        let inf = konst(self, f32::INFINITY);
        let is_inf = bin(self, x, inf, BOp::Eq);
        out = sel(self, is_inf, inf, out);
        // A NaN input stays that NaN (its own payload and sign).
        let is_nan = bin(self, x, x, BOp::NotEq);
        sel(self, is_nan, x, out)
    }

    /// `sin x` / `cos x` as IR: argument reduction plus a Taylor core.
    ///
    /// The builder's `sin_cos_program` is Payne–Hanek — five limbs of 2/π
    /// shifted into integer registers, three `SFPMUL24` pairs, a carry chain,
    /// about 180 instructions — and it exists to stay exact for arguments
    /// anywhere in the exponent range. ML math does not need that (nothing
    /// bit-matches `__sinf`, a TF32 matmul rounds to ten mantissa bits), and
    /// `SFPMUL24` is not expressible in IR anyway, so this is the classical
    /// float reduction:
    ///
    /// - `n = round(x * 2/π)` by the magic constant, the same trick `tt_exp2`
    ///   uses, kept as a float for the reduction and an `i32` for the quadrant;
    /// - `r = x - n*PIO2_HI - n*PIO2_LO` with a two-limb π/2, so `|r| <= π/4`
    ///   and `n*PIO2_HI` is exact while `|n| < 2^12`;
    /// - the Taylor polynomial of `sin r` / `cos r`, five terms, error under
    ///   1e-9 over that interval;
    /// - the quadrant from `q = n & 3`: its low bit chooses `sin` or `cos`, and
    ///   `q >> 1` (for `cos`, `(q + 1) >> 1 & 1`) flips the sign.
    ///
    /// Exact to about 6400 rad; past that the two-limb reduction loses
    /// precision, which is the trade every float library makes. `±0` falls out
    /// of the polynomial, and `±inf`/NaN through `n*PIO2_HI` being infinite, so
    /// `r` is a NaN.
    fn tt_sin_cos(&mut self, at: OpId, x: OpId, cos: bool) -> OpId {
        let konst = |k: &mut Kernel, v: f32| k.insert_before(at, Op::Const(Constant::new(v)));
        let iconst = |k: &mut Kernel, v: i32| k.insert_before(at, Op::Const(Constant::new(v)));
        let bin = |k: &mut Kernel, x: OpId, y: OpId, bop: BOp| k.insert_before(at, Op::Binary { x, y, bop });
        let sel = |k: &mut Kernel, cond: OpId, if_true: OpId, if_false: OpId| {
            k.insert_before(at, Op::Select { cond, if_true, if_false })
        };
        // Two-limb π/2: the high part keeps 12 mantissa bits, so `n * PIO2_HI`
        // is exact for every `|n| < 2^12`; the low part carries the rest.
        let pio2_hi = f32::from_bits((std::f64::consts::FRAC_PI_2 as f32).to_bits() & !0xfff);
        let pio2_lo = (std::f64::consts::FRAC_PI_2 - f64::from(pio2_hi)) as f32;
        let magic = konst(self, 12_582_912.0); // 1.5 * 2^23
        let inv_pio2 = konst(self, std::f32::consts::FRAC_2_PI);
        let pio2_hi = konst(self, pio2_hi);
        let pio2_lo = konst(self, pio2_lo);
        let t = bin(self, x, inv_pio2, BOp::Mul);
        let t = bin(self, t, magic, BOp::Add);
        let nf = bin(self, t, magic, BOp::Sub);
        let n = self.insert_before(at, Op::Cast { x: nf, dtype: DType::I32 });
        let a = bin(self, nf, pio2_hi, BOp::Mul);
        let r = bin(self, x, a, BOp::Sub);
        let b = bin(self, nf, pio2_lo, BOp::Mul);
        let r = bin(self, r, b, BOp::Sub);
        let s = bin(self, r, r, BOp::Mul);
        let sin_coeffs = [-1.0 / 6.0, 1.0 / 120.0, -1.0 / 5040.0, 1.0 / 362_880.0, -1.0 / 39_916_800.0];
        let cos_coeffs = [-0.5, 1.0 / 24.0, -1.0 / 720.0, 1.0 / 40_320.0, -1.0 / 3_628_800.0];
        let sin_core = self.tt_horner(at, s, &sin_coeffs);
        let cos_core = self.tt_horner(at, s, &cos_coeffs);
        let sin_r = bin(self, r, sin_core, BOp::Mul);
        // `q`'s low bit picks the core, `q >> 1` the sign (see the doc above).
        // Both conditions are comparisons: `Select` wants a `Bool` tile.
        let three = iconst(self, 3);
        let one_i = iconst(self, 1);
        let zero_i = iconst(self, 0);
        let q = bin(self, n, three, BOp::BitAnd);
        let low = bin(self, q, one_i, BOp::BitAnd);
        let odd = bin(self, low, zero_i, BOp::NotEq);
        let neg = if cos {
            let q1 = bin(self, q, one_i, BOp::Add);
            let half = bin(self, q1, one_i, BOp::BitShiftRight);
            let bit = bin(self, half, one_i, BOp::BitAnd);
            bin(self, bit, zero_i, BOp::NotEq)
        } else {
            let half = bin(self, q, one_i, BOp::BitShiftRight);
            bin(self, half, zero_i, BOp::NotEq)
        };
        let (odd_core, even_core) = if cos { (sin_r, cos_core) } else { (cos_core, sin_r) };
        let base = sel(self, odd, odd_core, even_core);
        let minus_one = konst(self, -1.0);
        let flipped = bin(self, base, minus_one, BOp::Mul);
        sel(self, neg, flipped, base)
    }

    /// `1 + s*(c1 + s*(c2 + ...))`: the trig cores as IR adds and multiplies,
    /// with `coeffs` the non-constant terms in increasing power of `s`.
    fn tt_horner(&mut self, at: OpId, s: OpId, coeffs: &[f32]) -> OpId {
        let konst = |k: &mut Kernel, v: f32| k.insert_before(at, Op::Const(Constant::new(v)));
        let bin = |k: &mut Kernel, x: OpId, y: OpId, bop: BOp| k.insert_before(at, Op::Binary { x, y, bop });
        let mut p = konst(self, coeffs[coeffs.len() - 1]);
        for c in coeffs[..coeffs.len() - 1].iter().rev() {
            let cc = konst(self, *c);
            let q = bin(self, p, s, BOp::Mul);
            p = bin(self, q, cc, BOp::Add);
        }
        let one = konst(self, 1.0);
        let q = bin(self, p, s, BOp::Mul);
        bin(self, q, one, BOp::Add)
    }

    pub fn tt_dataflow_select(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        for (_, x) in self.iter_unordered() {
            if let Op::Loop { .. } = x {
                return;
            }
        }
        // CB Storage definitions live in the prologue (past leading
        // params/storages, before everything else): inserting before
        // `prologue` keeps it stable.
        let mut prologue = self.head;
        for _ in 0..250_000 {
            if prologue.is_null() {
                break;
            }
            match self.ops[prologue].op {
                Op::Param { .. } | Op::Storage { .. } => prologue = self.ops[prologue].next,
                _ => break,
            }
        }
        // Phase-2 pattern table, biggest first. The recursive lowering
        // tries each pattern at a value root and stops at the first
        // emitter that succeeds; a failed emitter falls through.
        let tile = Pat::bind('t').shape([32i64, 32]);
        // Single-tile matmul: Reduce(Add) over a tiled Mul to a
        // tiled root. Both orders (matmul is elementwise-commutative
        // in IR; the A/B roles resolve structurally at install).
        let ma = Pat::bind('a').shape([32i64, 32]);
        let mb = Pat::bind('b').shape([32i64, 32]);
        let mmul = Pat::sum((ma.clone() * mb.clone()).shape([32i64, 32, 32])).shape([32i64, 32]);
        let mmul_swap = Pat::sum((mb.clone() * ma.clone()).shape([32i64, 32, 32])).shape([32i64, 32]);
        // Feeder index shapes: K-plain (A) vs K-scaled (B). Add/Mul
        // commutativity covers operand orders; each match_pat call
        // binds at most 3 (fresh budget per call).
        let pat_a = Pat::bind('k') + (Pat::bind('d') * Pat::bind('e'));
        let pat_b = Pat::bind('f') + (Pat::bind('g') * Pat::bind('k'));
        // A rank-1 operand: one tile row (see the vector reduce patterns).
        let vec1 = Pat::bind('t').shape([32i64]);
        let mut pats: Vec<(Pat, TtEmit)> = vec![
            (mmul, TtEmit::Matmul),
            (mmul_swap, TtEmit::Matmul),
            (Pat::sum(tile.clone()).shape([32i64]), TtEmit::Reduce(TileDim::Row)),
            (Pat::reduce(tile.clone(), BOp::Max).shape([32i64]), TtEmit::Reduce(TileDim::Row)),
            (Pat::sum(tile.clone()).shape([] as [i64; 0]), TtEmit::Reduce(TileDim::Scalar)),
            (Pat::reduce(tile, BOp::Max).shape([] as [i64; 0]), TtEmit::Reduce(TileDim::Scalar)),
            // A `[32]` vector occupies one tile **column**: element `i` is
            // row `i`, column 0 (the tiled layout's rank-1 view — the reader
            // moves the whole tile and the transfer untilizes by taking
            // column 0 of each row). So folding it is a reduce over the
            // tile's rows: `TileDim::Col`. That is the second kernel of a
            // whole-tensor reduce, whose only axis the loop split consumed.
            (Pat::sum(vec1.clone()).shape([] as [i64; 0]), TtEmit::Reduce(TileDim::Col)),
            (Pat::reduce(vec1, BOp::Max).shape([] as [i64; 0]), TtEmit::Reduce(TileDim::Col)),
        ];
        // Generic elementwise table: every UOp/BOp with a native kind.
        // This is data, not per-op lowering code. Silu needs no entry:
        // it is `Mul(x, sigmoid(x))`, so the Mul entry recurses into the
        // fused sigmoid entry, which allocates its own DST result.
        for &uop in UOPS {
            let Some(kind) = uop_kind(uop) else { continue };
            let p2 = Pat::unary(Pat::bind('x').shape([32i64, 32]), uop).shape([32i64, 32]);
            let p1 = Pat::unary(Pat::bind('x').shape([32i64]), uop).shape([32i64]);
            pats.push((Pat::Any(Box::new([p2, p1])), TtEmit::Sfpu { kind, arity: 1 }));
        }
        for &bop in BOPS {
            let Some(kind) = bop_kind(bop) else { continue };
            let p2 = Pat::binary(bop, Pat::bind('x').shape([32i64, 32]), Pat::bind('y').shape([32i64, 32])).shape([32i64, 32]);
            let p1 = Pat::binary(bop, Pat::bind('x').shape([32i64]), Pat::bind('y').shape([32i64])).shape([32i64]);
            pats.push((Pat::Any(Box::new([p2, p1])), TtEmit::Sfpu { kind, arity: 2 }));
            // A scalar-constant operand (`tt_unfold`'s coefficients): only the
            // other side's shape is pinned; the emitter tells the sides apart
            // by which operand is a `Const`.
            let right2 = Pat::binary(bop, Pat::bind('x').shape([32i64, 32]), Pat::Const);
            let left2 = Pat::binary(bop, Pat::Const, Pat::bind('x').shape([32i64, 32]));
            pats.push((Pat::Any(Box::new([right2, left2])), TtEmit::SfpuConst { kind }));
            let right1 = Pat::binary(bop, Pat::bind('x').shape([32i64]), Pat::Const);
            let left1 = Pat::binary(bop, Pat::Const, Pat::bind('x').shape([32i64]));
            pats.push((Pat::Any(Box::new([right1, left1])), TtEmit::SfpuConst { kind }));
        }
        // Phase 1 — reader publishes: every tile `Load` over a DRAM
        // param gets a `TTOp::Copy` publishing the tile to the param's
        // CB, base index stripped of lanes, explicit byte size. Loads
        // stay in place and dangle once compute is lowered (DCE
        // removes them with their index math). Tiled DRAM traffic
        // only — anything else bails the whole pass, since a
        // half-rewritten kernel is worse than none.
        let mut loads: Vec<(OpId, OpId, OpId)> = Vec::new();
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() {
                break;
            }
            if let Op::Load { src: base, src_idx: off, .. } = self.ops[id].op {
                if matches!(self.ops[base].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. }) {
                    // A `[32, 32]` tile, or a `[32]` vector — one tile row.
                    // The reader moves a whole tile in both cases (the DRAM
                    // buffers are page-sized), and row 0 of the unpacked tile
                    // is the vector, which is what the scalar-reduce kernel's
                    // second pass reads.
                    let shape = self.shape(id);
                    if shape != vec![32, 32] && shape != vec![32] {
                        return;
                    }
                    loads.push((id, base, off));
                }
            }
            id = self.ops[id].next;
        }
        // CB per DRAM param, in first-load order; slots counted up front.
        // `reader_cb_of` survives for phase 2 (feeder → CB resolution).
        let mut reader_cb_of: Map<OpId, OpId> = Map::default();
        if !loads.is_empty() {
            for (_, _, off) in loads.iter() {
                let mut stack = vec![*off];
                for _ in 0..250_000 {
                    let Some(n) = stack.pop() else { break };
                    match self.ops[n].op {
                        Op::Const(..) | Op::Range { .. } | Op::Param { .. } | Op::Unroll { .. } => {}
                        Op::Binary { x, y, .. } => {
                            stack.push(x);
                            stack.push(y);
                        }
                        Op::Unary { x, .. } => stack.push(x),
                        Op::Cast { x, .. } => stack.push(x),
                        _ => return,
                    }
                }
            }
            let anchor = loads[0].0;
            // One CB per DRAM param, in first-load order; slots counted up front.
            let mut cbs: Vec<(OpId, DType, i64)> = Vec::new();
            for (_, base, _) in loads.iter() {
                if let Some(entry) = cbs.iter_mut().find(|(b, _, _)| *b == *base) {
                    entry.2 += 1;
                } else {
                    let Op::Param { dtype, .. } = self.ops[*base].op else { return };
                    cbs.push((*base, dtype, 1));
                }
            }
            let mut cb_of: Map<OpId, OpId> = Map::default();
            let mut slot_of: Map<OpId, i64> = Map::default();
            for (base, dtype, ntiles) in cbs.iter() {
                let cb = if prologue.is_null() {
                    self.push_back(Op::Storage { dtype: *dtype, scope: MemScope::Circular, len: ntiles * 1024, id: 0 })
                } else {
                    self.insert_before(
                        prologue,
                        Op::Storage { dtype: *dtype, scope: MemScope::Circular, len: ntiles * 1024, id: 0 },
                    )
                };
                cb_of.insert(*base, cb);
                reader_cb_of.insert(*base, cb);
                slot_of.insert(*base, 0);
            }
            for (_load, base, off) in loads.iter() {
                let slot = slot_of[base];
                slot_of.insert(*base, slot + 1);
                let cb = cb_of[base];
                let dtype = self.dtype(*base);
                // CB indices are elementwise (uniform with vector traffic):
                // tile k lives at element offset k*1024.
                let slot_c = self.insert_const_idx_before(anchor, slot * 1024);
                // Base address with lanes stripped; the transfer size is
                // explicit, so no lane math rides along.
                let Some(base_idx) = self.strip_unrolls(*off, anchor) else {
                    return;
                };
                let bytes = dtype.bit_size() as u32 / 8 * 1024;
                self.insert_before(anchor, Op::TT(TTOp::Copy { src: *base, src_idx: base_idx, dst: cb, dst_idx: slot_c, bytes }));
            }
            self.insert_before(anchor, Op::TT(TTOp::EndReader));
        }
        // Phase 2 — native lowering. Values are lowered recursively from
        // the DRAM output stores. Every native op writes a fresh logical
        // Register storage. A computed value with more than one use is
        // packed to a Circular CB (DST is single-use scratch; multi-use
        // values live in L1) and every consumer unpacks from that CB. The
        // first unpack is redundant with the producer's DST row on purpose:
        // a later pass can coalesce it, keeping lowering simple.
        let (_, rcs) = self.compute_dtypes_and_rcs();
        // Every pattern decision is taken here, before anything lowers, and
        // recorded with its bindings: matching again later would read the
        // buffer length of an already-lowered operand as its shape.
        let mut matched: Map<OpId, (usize, Bindings)> = Map::default();
        let mut scan = self.head;
        for _ in 0..250_000 {
            if scan.is_null() {
                break;
            }
            if matches!(
                self.ops[scan].op,
                Op::Unary { .. }
                    | Op::Binary { .. }
                    | Op::Reduce { .. }
                    | Op::Select { .. }
                    | Op::Bitcast { .. }
                    | Op::Cast { .. }
            ) {
                for (i, (pat, _)) in pats.iter().enumerate() {
                    if let Some(m) = self.match_pat(scan, pat) {
                        matched.insert(scan, (i, m));
                        break;
                    }
                }
            }
            scan = self.next_op(scan);
        }
        let mut lower = TtLowering {
            pats: &pats,
            matched: &matched,
            prologue,
            reader_cb_of: &reader_cb_of,
            rcs: &rcs,
            materialized: Map::default(),
            unpacked: Map::default(),
            pat_a,
            pat_b,
        };
        let mut roots: Vec<OpId> = Vec::new();
        let mut scan = self.head;
        for _ in 0..250_000 {
            if scan.is_null() {
                break;
            }
            if let Op::Store { dst, src, .. } = self.ops[scan].op {
                if matches!(self.ops[dst].op, Op::Param { kind: ParamKind::GlobalMut, .. }) {
                    roots.push(src);
                }
            }
            scan = self.next_op(scan);
        }
        for root in roots {
            let _ = self.tt_lower_value(root, &mut lower);
        }
        self.common_subexpression_elimination();
        // The high-level cones replaced by native ops are dead now. Drop
        // them before phase 3 so codegen sees only native compute.
        self.dead_code_elimination();

        // The prologue anchor was the first non-param/storage op when the
        // pass started; the DCE above can take exactly that op (a rewritten
        // index cone leaves nothing after the params). Phase 3 inserts CB
        // storages before it, so re-resolve it first.
        prologue = self.head;
        for _ in 0..250_000 {
            if prologue.is_null() {
                break;
            }
            match self.ops[prologue].op {
                Op::Param { .. } | Op::Storage { .. } => prologue = self.ops[prologue].next,
                _ => break,
            }
        }

        // Phase 3 — writer drains: every DRAM store of a lowered value
        // becomes a `TTOp::Copy` in a writer section closed by
        // `EndCompute`. A value already materialized in a CB is drained
        // directly; a DST result is packed to a writer CB first.
        let mut stores: Vec<(OpId, OpId, OpId, OpId)> = Vec::new();
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() {
                break;
            }
            match self.ops[id].op {
                Op::Load { src: base, .. } => {
                    if matches!(self.ops[base].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. })
                        && self.shape(id) != vec![32, 32]
                    {
                        return;
                    }
                }
                Op::Store { dst: base, dst_idx: off, src, .. } => {
                    if matches!(self.ops[base].op, Op::Storage { .. }) {
                        // Already lowered; nothing owned here.
                    } else if matches!(self.ops[base].op, Op::Param { kind: ParamKind::GlobalMut, .. }) {
                        // Tiled stores, row stores (a reduce row is 32
                        // elements) and scalar stores (a whole-tile reduce's
                        // first lane); anything else bails the pass.
                        match self.shape(id).as_slice() {
                            [32, 32] | [32] | [1024] | [] => stores.push((id, base, off, src)),
                            _ => return,
                        }
                    } else {
                        return;
                    }
                }
                _ => {}
            }
            id = self.ops[id].next;
        }
        if !stores.is_empty() {
            let first = stores[0].0;
            // Empty reader section for store-only kernels.
            let mut scan = self.head;
            let mut has_reader = false;
            for _ in 0..250_000 {
                if scan.is_null() {
                    break;
                }
                if matches!(self.ops[scan].op, Op::TT(TTOp::EndReader)) {
                    has_reader = true;
                    break;
                }
                scan = self.ops[scan].next;
            }
            if !has_reader {
                // A tiled DRAM load without a reader section means phase 1
                // bailed: don't paper over it.
                let mut scan = self.head;
                for _ in 0..250_000 {
                    if scan.is_null() {
                        break;
                    }
                    if let Op::Load { src: base, .. } = self.ops[scan].op {
                        let shape = self.shape(scan);
                        if matches!(self.ops[base].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. })
                            && (shape == vec![32, 32] || shape == vec![32])
                        {
                            return;
                        }
                    }
                    scan = self.ops[scan].next;
                }
                self.insert_before(first, Op::TT(TTOp::EndReader));
            }
            // Compute-side rewrites in head order; drains append at the tail.
            self.push_back(Op::TT(TTOp::EndCompute));
            let slot_zero = self.push_back(Op::Const(Constant::idx(0i64)));
            for (store, base, off, src) in stores.iter() {
                let Some((dtype, storage)) = self.tt_lowered_storage(*src, &lower) else {
                    // Unlowered value: stays for later increments.
                    continue;
                };
                // Single-tile shapes only; anything else bails the pass.
                match self.shape(*store).as_slice() {
                    [32, 32] | [32] | [1024] | [] => {}
                    _ => return,
                };
                if self.dtype(*store) != dtype {
                    return;
                }
                let bytes = match (dtype.bit_size() as u32 / 8).checked_mul(1024) {
                    Some(b) if b > 0 => b,
                    _ => return,
                };
                let tail = self.tail;
                let Some(base_idx) = self.strip_unrolls(*off, tail) else {
                    return;
                };
                if matches!(self.ops[storage].op, Op::Storage { scope: MemScope::Circular, .. }) {
                    // Multi-use value already in L1: drain it directly.
                    let zero = self.insert_const_idx_before(*store, 0i64);
                    self.push_back(Op::TT(TTOp::Copy { src: storage, src_idx: zero, dst: *base, dst_idx: base_idx, bytes }));
                } else {
                    // Output boundary: pack the DST row into a writer CB,
                    // then drain that CB to DRAM. The writer CB is FP32 (the
                    // packer writes FP32 out of `Dst`); when the param is
                    // 16-bit, the host boundary narrows on the way back
                    // (`TTDmMove::narrow`). Packing BF16 straight from `Dst`
                    // is the packer's F32->BF16 conversion, which the pinned
                    // ttsim refuses.
                    let pack_bytes = match (DType::F32.bit_size() as u32 / 8).checked_mul(1024) {
                        Some(b) if b > 0 => b,
                        _ => return,
                    };
                    let writer_cb = self.tt_new_cb(DType::F32, prologue);
                    let zero_pack = self.insert_const_idx_before(*store, 0i64);
                    // The pack is a tile move (DST -> L1), not an SSA store:
                    // after allocation there is no SSA, so it goes out as
                    // `Copy` like every other memory-level move. It keeps the
                    // storage as its source, which is what the reuse pass
                    // needs to know the row is still live.
                    let pack = Op::TT(TTOp::Copy {
                        src: storage,
                        src_idx: zero_pack,
                        dst: writer_cb,
                        dst_idx: zero_pack,
                        bytes: pack_bytes,
                    });
                    // The pack itself is a plain tile move; the hand-off that
                    // orders it against the math that filled the tile is
                    // `tt_reg_sync`'s, from the tile's lifetimes.
                    self.insert_before(*store, pack);
                    self.push_back(Op::TT(TTOp::Copy {
                        src: writer_cb,
                        src_idx: slot_zero,
                        dst: *base,
                        dst_idx: base_idx,
                        bytes: pack_bytes,
                    }));
                }
                self.remove_op(*store);
            }
        }
        self.verify();
    }

    /// Fresh logical DST storage, inserted in the prologue so it dominates
    /// every compute op. The reuse pass later assigns it a physical row.
    fn tt_new_dst(&mut self, dtype: DType, prologue: OpId) -> OpId {
        let op = Op::Storage { dtype, scope: MemScope::Register, len: 1024, id: 0 };
        if prologue.is_null() {
            self.push_back(op)
        } else {
            self.insert_before(prologue, op)
        }
    }

    /// One row loop of a ported kind: `body` writes one four-row group's
    /// column half at offset `o`; the emitted sequence is that body once
    /// inside `TTOp::Replay` (32 iterations, the row address modifier stepping
    /// `Dst` by two rows). Codegen picks the mechanism from the length: a
    /// body that fits the replay buffer is a hardware `REPLAY`, a longer one
    /// is a block repeat the runner pushes. The body must end with a `Dst`
    /// access.
    fn tt_row_loop(&mut self, body: impl Fn(&mut Vec<Op>, u16)) -> Vec<Op> {
        let mut once: Vec<Op> = Vec::new();
        body(&mut once, 0);
        let words: u32 = once.iter().map(tt_row_words).sum();
        let mut ops: Vec<Op> = Vec::with_capacity(once.len() + 1);
        ops.push(Op::TT(TTOp::Replay { len: words as u16, count: 32 }));
        ops.extend(once);
        ops
    }

    /// A scalar constant as a DST tile: one lane-register immediate, stored to
    /// every row group, inserted before `consumer`. A scalar operand of an
    /// elementwise op has no tile of its own in the IR (`tt_unfold`'s
    /// coefficients), so this is where it gets one.
    fn tt_fill_const(&mut self, konst: OpId, consumer: OpId, prologue: OpId) -> Option<(DType, OpId)> {
        let Op::Const(c) = self.ops[konst].op else {
            return None;
        };
        let (bits, fmt) = match c {
            Constant::F32(b) => (u32::from_le_bytes(b), FP32),
            Constant::I32(v) => (v as u32, INT32),
            other => todo!("a fill of a {other:?} constant has no native form"),
        };
        let dtype = c.dtype();
        let slot = self.tt_new_dst(dtype, prologue);
        self.insert_before(consumer, Op::TT(TTOp::SfpConst(Box::new(TTSfpConst { reg: L0, bits }))));
        for o in (0..64u16).step_by(2) {
            self.insert_before(consumer, Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L0, fmt }))));
        }
        Some((dtype, slot))
    }

    /// Fresh Circular CB, inserted in the prologue. Used for multi-use
    /// values and for writer packs.
    fn tt_new_cb(&mut self, dtype: DType, prologue: OpId) -> OpId {
        let op = Op::Storage { dtype, scope: MemScope::Circular, len: 1024, id: 0 };
        if prologue.is_null() {
            self.push_back(op)
        } else {
            self.insert_before(prologue, op)
        }
    }

    /// Resolve a value to its logical storage. A computed value with one
    /// use is its DST row; a multi-use value is its CB; a DRAM load is its
    /// reader CB. The caller decides whether to unpack.
    fn tt_lower_value(&mut self, id: OpId, lower: &mut TtLowering) -> Option<(DType, OpId)> {
        if let Some(&cb) = lower.materialized.get(&id) {
            return Some((self.dtype(id), cb));
        }
        match self.ops[id].op {
            // A lowered accumulator (`Setrwc`) carries its tile in `dst`:
            // the matmul's value is the DST storage the Matrix Unit filled.
            Op::TT(TTOp::Sfpu { slot, .. }) => {
                return Some((self.dtype(id), slot));
            }
            Op::TT(TTOp::Setrwc { dst }) => return Some((self.dtype(id), dst)),
            Op::Storage { dtype, .. } => return Some((dtype, id)),
            Op::Load { .. } => return self.tt_lower_load(id, lower),
            // The three value-preserving/reinterpreting ops are `Op`s, not
            // `UOp`/`BOp` kinds, so they lower here rather than through the
            // kind-keyed pattern table.
            Op::Bitcast { x, dtype } => {
                let (_, a) = self.tt_resolve_dst(x, id, lower)?;
                let slot = self.tt_new_dst(dtype, lower.prologue);
                let mut ops = self.tt_sfp_bitcast(a, slot);
                let last = ops.pop().expect("the sequence is never empty");
                for op in ops {
                    self.insert_before(id, op);
                }
                self.ops[id].op = last;
                return self.tt_materialize(id, dtype, slot, lower);
            }
            Op::Cast { x, dtype } => {
                let (from, a) = self.tt_resolve_dst(x, id, lower)?;
                let slot = self.tt_new_dst(dtype, lower.prologue);
                let mut ops = match (from, dtype) {
                    (DType::I32, DType::F32) => self.tt_sfp_cast_i32_f32(a, slot),
                    (DType::F32, DType::I32) => self.tt_sfp_cast_f32_i32(a, slot),
                    _ => todo!("cast {from:?} -> {dtype:?} has no native row-op program"),
                };
                let last = ops.pop().expect("the sequence is never empty");
                for op in ops {
                    self.insert_before(id, op);
                }
                self.ops[id].op = last;
                return self.tt_materialize(id, dtype, slot, lower);
            }
            // An integer `Add`/`Sub` is `SFPIADD`, never the float arm:
            // `SFPMAD` would add the two words as F32 and drop the integer
            // carry, which silently loses `tt_exp2`'s `bits(d) + (n << 23)`
            // (every |x| >= 0.5 came out exactly half of 2^x), and the float
            // arm's unpack refuses an `I32` tile outright (`tt_log2`'s
            // `e - 127`). The pattern table keys on the `BOp` alone, so the
            // dtype has to be checked here.
            Op::Binary { x, y, bop } if matches!(bop, BOp::Add | BOp::Sub) && self.dtype(id) == DType::I32 => {
                // A scalar constant operand has no tile; fill one, as the
                // float arms' `SfpuConst` emitter does.
                let (dtype, a, b) = if matches!(self.ops[x].op, Op::Const(_)) {
                    let (dtype, a) = self.tt_resolve_dst(y, id, lower)?;
                    let (_, kb) = self.tt_fill_const(x, id, lower.prologue)?;
                    (dtype, kb, a)
                } else if matches!(self.ops[y].op, Op::Const(_)) {
                    let (dtype, a) = self.tt_resolve_dst(x, id, lower)?;
                    let (_, kb) = self.tt_fill_const(y, id, lower.prologue)?;
                    (dtype, a, kb)
                } else {
                    let (dtype, a) = self.tt_resolve_dst(x, id, lower)?;
                    let (_, b) = self.tt_resolve_dst(y, id, lower)?;
                    (dtype, a, b)
                };
                let slot = self.tt_new_dst(dtype, lower.prologue);
                let mut ops = match bop {
                    BOp::Add => self.tt_sfp_int_add(a, b, slot),
                    BOp::Sub => self.tt_sfp_int_sub(a, b, slot),
                    other => unreachable!("the arm collected {other:?}"),
                };
                let last = ops.pop().expect("the sequence is never empty");
                for op in ops {
                    self.insert_before(id, op);
                }
                self.ops[id].op = last;
                return self.tt_materialize(id, dtype, slot, lower);
            }
            Op::Select { cond, if_true, if_false } => {
                let (_, c) = self.tt_resolve_dst(cond, id, lower)?;
                // A branch may be a scalar constant (`tt_unfold`'s specials):
                // it has no storage, so it is filled into a tile here.
                let (dtype, t) = if matches!(self.ops[if_true].op, Op::Const(_)) {
                    self.tt_fill_const(if_true, id, lower.prologue)?
                } else {
                    self.tt_resolve_dst(if_true, id, lower)?
                };
                let (_, f) = if matches!(self.ops[if_false].op, Op::Const(_)) {
                    self.tt_fill_const(if_false, id, lower.prologue)?
                } else {
                    self.tt_resolve_dst(if_false, id, lower)?
                };
                let slot = self.tt_new_dst(dtype, lower.prologue);
                let mut ops = self.tt_sfp_select(c, t, f, slot);
                let last = ops.pop().expect("the sequence is never empty");
                for op in ops {
                    self.insert_before(id, op);
                }
                self.ops[id].op = last;
                return self.tt_materialize(id, dtype, slot, lower);
            }
            _ => {}
        }
        // The decision recorded before lowering, then (for an emitter that
        // turned out not to apply) the rest of the table.
        let recorded = lower.matched.get(&id).map(|(i, m)| (*i, m.clone()));
        if let Some((i, m)) = recorded {
            let emit = lower.pats[i].1;
            if let Some(v) = self.tt_emit(id, emit, &m, lower) {
                return Some(v);
            }
        }
        for i in 0..lower.pats.len() {
            let Some(m) = self.match_pat(id, &lower.pats[i].0) else {
                continue;
            };
            let emit = lower.pats[i].1;
            if let Some(v) = self.tt_emit(id, emit, &m, lower) {
                return Some(v);
            }
        }
        None
    }

    /// A DRAM load is already published to its reader CB by phase 1; a
    /// load from another storage is that storage. No DST traffic yet.
    fn tt_lower_load(&self, id: OpId, lower: &TtLowering) -> Option<(DType, OpId)> {
        let Op::Load { src: base, .. } = self.ops[id].op else {
            return None;
        };
        match self.ops[base].op {
            Op::Param { dtype, kind: ParamKind::Global | ParamKind::GlobalMut, .. } => {
                let cb = lower.reader_cb_of.get(&base).copied()?;
                Some((dtype, cb))
            }
            Op::Storage { dtype, .. } => Some((dtype, base)),
            _ => None,
        }
    }

    /// Resolve a value to a DST row, unpacking a CB if needed. This is the
    /// only CB -> DST conversion in compute; the copy is single-use and the
    /// reuse pass can coalesce its row with the producer.
    fn tt_resolve_dst(&mut self, id: OpId, consumer: OpId, lower: &mut TtLowering) -> Option<(DType, OpId)> {
        let (dtype, storage) = self.tt_lower_value(id, lower)?;
        match self.ops[storage].op {
            Op::Storage { scope: MemScope::Register, .. } => Some((dtype, storage)),
            Op::Storage { scope: MemScope::Circular, .. } => {
                // One tile per CB: a kernel phase's CB holds one tile, and
                // the unpack always reads slot 0, so every use of the same CB
                // may share one unpacked DST tile.
                if let Some(&dst) = lower.unpacked.get(&storage) {
                    return Some((dtype, dst));
                }
                let dst = self.tt_new_dst(dtype, lower.prologue);
                let zero = self.insert_const_idx_before(consumer, 0i64);
                let bytes = (dtype.bit_size() as u32 / 8).checked_mul(1024)?;
                self.insert_before(consumer, Op::TT(TTOp::Copy { src: storage, src_idx: zero, dst, dst_idx: zero, bytes }));
                lower.unpacked.insert(storage, dst);
                Some((dtype, dst))
            }
            _ => None,
        }
    }

    /// A computed value with more than one use must live in L1: DST is
    /// single-use scratch. Pack it after the producer and return the CB;
    /// every consumer unpacks. The extra first unpack is deliberate.
    fn tt_materialize(&mut self, id: OpId, dtype: DType, slot: OpId, lower: &mut TtLowering) -> Option<(DType, OpId)> {
        let uses = lower.rcs.get(&id).copied().unwrap_or(0);
        if uses <= 1 {
            return Some((dtype, slot));
        }
        let cb = self.tt_new_cb(dtype, lower.prologue);
        let zero = self.insert_const_idx_before(id, 0i64);
        let bytes = (dtype.bit_size() as u32 / 8).checked_mul(1024)?;
        // A tile move, not an SSA store: the materialized value is in L1 now,
        // and every consumer reads it back with a `Copy` (unpack).
        let pack = Op::TT(TTOp::Copy { src: slot, src_idx: zero, dst: cb, dst_idx: zero, bytes });
        self.insert_after(id, pack);
        lower.materialized.insert(id, cb);
        Some((dtype, cb))
    }

    /// Emit one SFPU kind as row-level ops in place of `Sfpu { kind }`.
    ///
    /// The kinds are grouped by the builder program they transcribe, one
    /// function per program; this only dispatches. Every sequence is
    /// **unrolled**: `for_each_row_group(64)` runs 32 iterations at offsets
    /// `0, 2, …, 62` (an `SFPLOAD` address is a four-row group plus a column
    /// half), and every instruction becomes one `TTOp`. The last op takes
    /// `id`'s place so `tt_materialize`'s pack lands after the whole sequence
    /// rather than inside it. Returns `false` for a kind that is not ported
    /// yet; the caller then keeps `Sfpu { kind }`.
    fn tt_sfp_sequence(&mut self, id: OpId, kind: u32, a: OpId, b: OpId, slot: OpId) -> bool {
        use crate::backend::tenstorrent::kern::kind;
        use crate::backend::tenstorrent::kern::sfpu::ops::kind_sfpu as sf;
        let mut ops: Vec<Op> = if kind == sf::BOOL_NOT && b.is_null() {
            self.tt_sfp_bool_not(a, slot)
        } else if kind == kind::ADD && !b.is_null() {
            self.tt_sfp_add(a, b, slot)
        } else if kind == kind::SUB && !b.is_null() {
            self.tt_sfp_sub(a, b, slot)
        } else if kind == kind::MUL && !b.is_null() {
            self.tt_sfp_mul(a, b, slot)
        } else if kind == sf::NEG && b.is_null() {
            self.tt_sfp_neg(a, slot)
        } else if kind == sf::ABS && b.is_null() {
            self.tt_sfp_abs(a, slot)
        } else if kind == sf::INT_NOT && b.is_null() {
            self.tt_sfp_int_not(a, slot)
        } else if (kind == sf::SQRT || kind == sf::RSQRT) && b.is_null() {
            self.tt_sfp_sqrt(kind, a, slot)
        } else if kind == sf::RECIP && b.is_null() {
            self.tt_sfp_recip(a, slot)
        } else if (kind == sf::FLOOR || kind == sf::TRUNC) && b.is_null() {
            self.tt_sfp_round(kind, a, slot)
        } else if kind == sf::DIV && !b.is_null() {
            self.tt_sfp_div(a, b, slot)
        } else if (sf::EQ..=sf::LE).contains(&kind) && !b.is_null() {
            self.tt_sfp_compare(kind, a, b, slot)
        } else if (sf::INT_AND..=sf::INT_SHR).contains(&kind) && !b.is_null() {
            self.tt_sfp_integer(kind, a, b, slot)
        } else {
            return false;
        };
        let last = ops.pop().expect("the sequence is never empty");
        for op in ops {
            self.insert_before(id, op);
        }
        self.ops[id].op = last;
        true
    }

    /// `exact_program`'s `BOOL_NOT`, on raw bits.
    fn tt_sfp_bool_not(&mut self, a: OpId, slot: OpId) -> Vec<Op> {
        let mut ops: Vec<Op> = Vec::new();
        // `0`/`1` either way: `x ^ 1`, on raw bits.
        let xor = tt_isa::isa::generated::defs::SFPXOR.opcode();
        ops.push(Op::TT(TTOp::SfpConst(Box::new(TTSfpConst { reg: L3, bits: 1 }))));
        let body = self.tt_row_loop(|mut ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L2, fmt: INT32 }))));
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op: xor, args: [u16::from(L3), u16::from(L2), 0, 0, 0] })));
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L2, fmt: INT32 }))));
        });
        ops.extend(body);
        ops
    }

    /// `d = 1 * a + b` (`SFPMAD` with the constant-one register), the same
    /// words `Program::add` emits.
    fn tt_sfp_add(&mut self, a: OpId, b: OpId, slot: OpId) -> Vec<Op> {
        // `d = 1 * a + b` (`SFPMAD` with the constant-one register), the
        // same words `Program::add` emits.
        let mad = tt_isa::isa::generated::defs::SFPMAD.opcode();
        self.tt_row_loop(|mut ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: FP32 }))));
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: b, at: o, reg: L1, fmt: FP32 }))));
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op: mad, args: [LREG_ONE, u16::from(L0), u16::from(L1), u16::from(L2), 0] })));
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L2, fmt: FP32 }))));
        })
    }

    /// `d = 1 * a - b`: `SFPMAD` with the constant-one register and a
    /// negated addend, the same words `Program::sub` emits.
    fn tt_sfp_sub(&mut self, a: OpId, b: OpId, slot: OpId) -> Vec<Op> {
        // `d = 1 * a - b`: `SFPMAD` with the constant-one register and a
        // negated addend, the same words `Program::sub` emits.
        let mad = tt_isa::isa::generated::defs::SFPMAD.opcode();
        self.tt_row_loop(|mut ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: FP32 }))));
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: b, at: o, reg: L1, fmt: FP32 }))));
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op: mad, args: [LREG_ONE, u16::from(L0), u16::from(L1), u16::from(L2), 2] })));
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L2, fmt: FP32 }))));
        })
    }

    /// `SFPMAD` with the constant-zero register as the addend and that zero
    /// negated (`Program::mul`): a plain `SFPMUL` adds a `+0`, which turns a
    /// `-0` product into `+0`.
    fn tt_sfp_mul(&mut self, a: OpId, b: OpId, slot: OpId) -> Vec<Op> {
        // `SFPMAD` with the constant-zero register as the addend and that
        // zero negated (`Program::mul`): a plain `SFPMUL` adds a `+0`,
        // which turns a `-0` product into `+0`.
        let mad = tt_isa::isa::generated::defs::SFPMAD.opcode();
        self.tt_row_loop(|mut ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: FP32 }))));
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: b, at: o, reg: L1, fmt: FP32 }))));
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op: mad, args: [u16::from(L0), u16::from(L1), LREG_ZERO, u16::from(L2), 2] })));
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L2, fmt: FP32 }))));
        })
    }

    /// `exact_program`'s `NEG`: a sign flip on the raw bits, so a NaN stays
    /// one (`SFPMOV` with `Mod1 = 1`). Loaded and stored as `Int32`: an FP32
    /// store would flush a denormal.
    fn tt_sfp_neg(&mut self, a: OpId, slot: OpId) -> Vec<Op> {
        // `exact_program`'s `NEG`: a sign flip on the raw bits, so a NaN
        // stays one (`SFPMOV` with `Mod1 = 1`). Loaded and stored as
        // `Int32`: an FP32 store would flush a denormal.
        let mov = tt_isa::isa::generated::defs::SFPMOV.opcode();
        self.tt_row_loop(|mut ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: INT32 }))));
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op: mov, args: [u16::from(L0), u16::from(L2), 1, 0, 0] })));
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L2, fmt: INT32 }))));
        })
    }

    /// `exact_program`'s `ABS`: the sign bit cleared, a negative NaN staying
    /// one (`SFPABS` with `Mod1 = 1`), raw bits both ways.
    /// `exact_program`'s `ABS`: the sign bit cleared on raw bits, as
    /// `x & 0x7fff_ffff` — a negative NaN comes out `+NaN`, where `SFPABS`
    /// would leave its sign set.
    fn tt_sfp_abs(&mut self, a: OpId, slot: OpId) -> Vec<Op> {
        let mut ops: Vec<Op> = Vec::new();
        let and = tt_isa::isa::generated::defs::SFPAND.opcode();
        ops.push(Op::TT(TTOp::SfpConst(Box::new(TTSfpConst { reg: L3, bits: 0x7fff_ffff }))));
        let body = self.tt_row_loop(|mut ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: INT32 }))));
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op: and, args: [u16::from(L3), u16::from(L0), u16::from(L2), 1, 0] })));
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L2, fmt: INT32 }))));
        });
        ops.extend(body);
        ops
    }

    /// `integer::body`'s `NOT`: bitwise, raw bits.
    fn tt_sfp_int_not(&mut self, a: OpId, slot: OpId) -> Vec<Op> {
        // `integer::body`'s `NOT`: bitwise, raw bits.
        let not = tt_isa::isa::generated::defs::SFPNOT.opcode();
        self.tt_row_loop(|mut ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: INT32 }))));
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op: not, args: [u16::from(L0), u16::from(L2), 0, 0, 0] })));
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L2, fmt: INT32 }))));
        })
    }

    /// `ops::sqrt_program`, `RSQRT` when `kind` says so.
    fn tt_sfp_sqrt(&mut self, kind: u32, a: OpId, slot: OpId) -> Vec<Op> {
        use crate::backend::tenstorrent::kern::sfpu::ops::kind_sfpu as sf;
        let mut ops: Vec<Op> = Vec::new();
        // `ops::sqrt_program`: the seed `0x5f3759df - (bits(x) >> 1)`,
        // three Newton steps (`h = x y`, `t = 1 - h y`, `y += (y/2) t`),
        // one more step for `sqrt` (`s = x y; s += (y/2)(x - s^2)`), then
        // the special lanes in the total order. `x = L0`, `d = L2`; the
        // program's `y, t, h, c` land in `L1, L3, L4, L5`.
        use tt_isa::isa::generated::defs;
        let (mad, mov, abs, and, shft, iadd, gt, ple, setsgn) = (
            defs::SFPMAD.opcode(),
            defs::SFPMOV.opcode(),
            defs::SFPABS.opcode(),
            defs::SFPAND.opcode(),
            defs::SFPSHFT.opcode(),
            defs::SFPIADD.opcode(),
            defs::SFPGT.opcode(),
            defs::SFPLE.opcode(),
            defs::SFPSETSGN.opcode(),
        );
        let (pushc, popc) = (defs::SFPPUSHC.opcode(), defs::SFPPOPC.opcode());
        let sfp = |ops: &mut Vec<Op>, op: u8, args: [u16; 5]| {
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op, args })));
        };
        let konst = |ops: &mut Vec<Op>, reg: u8, bits: u32| {
            ops.push(Op::TT(TTOp::SfpConst(Box::new(TTSfpConst { reg, bits }))));
        };
        // Lane-flag predication on, all lanes enabled, as `Program::new`
        // does: without `SFPENCC` every scope below is a no-op.
        sfp(&mut ops, defs::SFPENCC.opcode(), [3, 0, 10, 0, 0]);
        let inverse = kind == sf::RSQRT;
        let body = self.tt_row_loop(|mut ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: FP32 }))));
            // `mov(x, y)` then `y >>= 31`, `y = c - y`.
            sfp(&mut ops, mov, [u16::from(L0), u16::from(L1), 0, 0, 0]);
            konst(&mut ops, L5, 0xffff_ffff);
            sfp(&mut ops, shft, [0, u16::from(L5), u16::from(L1), 0, 0]);
            konst(&mut ops, L5, 0x5f37_59df);
            sfp(&mut ops, iadd, [0, u16::from(L5), u16::from(L1), 6, 0]);
            konst(&mut ops, L5, 0x3f00_0000); // 0.5
            for _ in 0..3 {
                // `h = x y`, `t = 1 - h y`, `h = y c`, `y += h t`.
                sfp(&mut ops, mad, [u16::from(L0), u16::from(L1), LREG_ZERO, u16::from(L4), 2]);
                sfp(&mut ops, mad, [u16::from(L4), u16::from(L1), LREG_ONE, u16::from(L3), 1]);
                sfp(&mut ops, mad, [u16::from(L1), u16::from(L5), LREG_ZERO, u16::from(L4), 2]);
                sfp(&mut ops, mad, [u16::from(L4), u16::from(L3), u16::from(L1), u16::from(L1), 0]);
            }
            if inverse {
                sfp(&mut ops, mov, [u16::from(L1), u16::from(L2), 0, 0, 0]);
            } else {
                // `d = x y`, `t = x - d d`, `h = y c`, `d += h t`.
                sfp(&mut ops, mad, [u16::from(L0), u16::from(L1), LREG_ZERO, u16::from(L2), 2]);
                sfp(&mut ops, mad, [u16::from(L2), u16::from(L2), u16::from(L0), u16::from(L3), 1]);
                sfp(&mut ops, mad, [u16::from(L1), u16::from(L5), LREG_ZERO, u16::from(L4), 2]);
                sfp(&mut ops, mad, [u16::from(L4), u16::from(L3), u16::from(L2), u16::from(L2), 0]);
            }
            // The special lanes, in the total order.
            sfp(&mut ops, abs, [u16::from(L0), u16::from(L3), 1, 0, 0]);
            konst(&mut ops, L5, 0x0080_0000);
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, gt, [u16::from(L3), u16::from(L5), 1, 0, 0]);
            if inverse {
                konst(&mut ops, L4, 0x7f80_0000);
                sfp(&mut ops, mov, [u16::from(L0), u16::from(L2), 0, 0, 0]);
                sfp(&mut ops, setsgn, [0, u16::from(L4), u16::from(L2), 0, 0]);
            } else {
                konst(&mut ops, L4, 0x8000_0000);
                sfp(&mut ops, and, [u16::from(L4), u16::from(L0), u16::from(L2), 1, 0]);
            }
            sfp(&mut ops, popc, [0; 5]);
            konst(&mut ops, L5, 0x7f80_0000);
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, ple, [u16::from(L0), u16::from(L5), 1, 0, 0]);
            if inverse {
                sfp(&mut ops, mov, [LREG_ZERO, u16::from(L2), 0, 0, 0]);
            } else {
                sfp(&mut ops, mov, [u16::from(L5), u16::from(L2), 0, 0, 0]);
            }
            sfp(&mut ops, popc, [0; 5]);
            konst(&mut ops, L5, 0x8000_0000);
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, gt, [u16::from(L0), u16::from(L5), 1, 0, 0]);
            konst(&mut ops, L2, 0x7fc0_0000);
            sfp(&mut ops, popc, [0; 5]);
            konst(&mut ops, L5, 0x7f80_0000);
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, gt, [u16::from(L5), u16::from(L0), 1, 0, 0]);
            konst(&mut ops, L2, 0x7fc0_0000);
            sfp(&mut ops, popc, [0; 5]);
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L2, fmt: FP32 }))));
        });
        ops.extend(body);
        ops
    }

    /// `exact_program`'s `RECIP`: `Program::recip`.
    fn tt_sfp_recip(&mut self, a: OpId, slot: OpId) -> Vec<Op> {
        let mut ops: Vec<Op> = Vec::new();
        // `exact_program`'s `RECIP`: `Program::recip` — the seed, two
        // Newton steps, then the four special cases (a `0`/denormal, an
        // infinity, `±0`'s sign, and the `±inf` sign).
        use tt_isa::isa::generated::defs;
        let (mad, mov, abs, and, arecip, setsgn, gt) = (
            defs::SFPMAD.opcode(),
            defs::SFPMOV.opcode(),
            defs::SFPABS.opcode(),
            defs::SFPAND.opcode(),
            defs::SFPARECIP.opcode(),
            defs::SFPSETSGN.opcode(),
            defs::SFPGT.opcode(),
        );
        let (pushc, popc, compc) = (defs::SFPPUSHC.opcode(), defs::SFPPOPC.opcode(), defs::SFPCOMPC.opcode());
        let sfp = |ops: &mut Vec<Op>, op: u8, args: [u16; 5]| {
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op, args })));
        };
        let konst = |ops: &mut Vec<Op>, reg: u8, bits: u32| {
            ops.push(Op::TT(TTOp::SfpConst(Box::new(TTSfpConst { reg, bits }))));
        };
        // Lane-flag predication on, all lanes enabled, as `Program::new`
        // does: without `SFPENCC` every scope below is a no-op.
        sfp(&mut ops, defs::SFPENCC.opcode(), [3, 0, 10, 0, 0]);
        let body = self.tt_row_loop(|mut ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: FP32 }))));
            // `max = L6` holds `f32::MAX` for the tail. Inside the body, so a
            // fused body writes every register it reads.
            konst(&mut ops, L6, 0x7f7f_ffff);
            // `recip(x = L0, d = L2, t0 = L3, t1 = L4, max = L6)`.
            sfp(&mut ops, abs, [u16::from(L0), u16::from(L4), 1, 0, 0]);
            konst(&mut ops, L3, 0x7180_0000); // 2^100
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, gt, [u16::from(L3), u16::from(L4), 1, 0, 0]);
            konst(&mut ops, L6, 0x1f80_0000); // 2^-64
            sfp(&mut ops, mad, [u16::from(L0), u16::from(L6), LREG_ZERO, u16::from(L4), 2]);
            sfp(&mut ops, arecip, [0, u16::from(L4), u16::from(L2), 0, 0]);
            for _ in 0..2 {
                sfp(&mut ops, mad, [u16::from(L4), u16::from(L2), LREG_ONE, u16::from(L3), 1]);
                sfp(&mut ops, mad, [u16::from(L3), u16::from(L2), u16::from(L2), u16::from(L2), 0]);
            }
            sfp(&mut ops, mad, [u16::from(L2), u16::from(L6), LREG_ZERO, u16::from(L2), 2]);
            konst(&mut ops, L6, 0x7f7f_ffff);
            sfp(&mut ops, compc, [0; 5]);
            sfp(&mut ops, arecip, [0, u16::from(L0), u16::from(L2), 0, 0]);
            for _ in 0..2 {
                sfp(&mut ops, mad, [u16::from(L0), u16::from(L2), LREG_ONE, u16::from(L3), 1]);
                sfp(&mut ops, mad, [u16::from(L3), u16::from(L2), u16::from(L2), u16::from(L2), 0]);
            }
            sfp(&mut ops, popc, [0; 5]);
            // `1/±0` and a denormal: the seed's infinity met `0 * inf`.
            konst(&mut ops, L3, 0x7fff_ffff);
            sfp(&mut ops, and, [u16::from(L3), u16::from(L0), u16::from(L4), 1, 0]);
            konst(&mut ops, L3, 0x0080_0000);
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, gt, [u16::from(L4), u16::from(L3), 1, 0, 0]);
            konst(&mut ops, L3, 0x7f80_0000);
            sfp(&mut ops, mov, [u16::from(L0), u16::from(L2), 0, 0, 0]);
            sfp(&mut ops, setsgn, [0, u16::from(L3), u16::from(L2), 0, 0]);
            sfp(&mut ops, popc, [0; 5]);
            // `1/±inf`: the seed's zero met `inf * 0`.
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, gt, [u16::from(L6), u16::from(L4), 1, 0, 0]);
            konst(&mut ops, L3, 0x7f80_0001);
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, gt, [u16::from(L4), u16::from(L3), 1, 0, 0]);
            sfp(&mut ops, mov, [u16::from(L0), u16::from(L2), 0, 0, 0]);
            sfp(&mut ops, setsgn, [0, LREG_ZERO, u16::from(L2), 0, 0]);
            sfp(&mut ops, popc, [0; 5]);
            sfp(&mut ops, popc, [0; 5]);
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L2, fmt: FP32 }))));
        });
        ops.extend(body);
        ops
    }

    /// `round::body`, `FLOOR` when `kind` says so, else `TRUNC`.
    fn tt_sfp_round(&mut self, kind: u32, a: OpId, slot: OpId) -> Vec<Op> {
        use crate::backend::tenstorrent::kern::sfpu::ops::kind_sfpu as sf;
        let mut ops: Vec<Op> = Vec::new();
        // `round::body`: raw bits; `|x| >= 2^23` is integral, `|x| < 1`
        // truncates to a signed zero (`floor` then takes `-1` for a
        // negative non-zero), else the mantissa below one integer ULP is
        // dropped and `floor` adds that ULP back.
        use tt_isa::isa::generated::defs;
        let (mov, and, shft, iadd, not, exexp, gt, setcc) = (
            defs::SFPMOV.opcode(),
            defs::SFPAND.opcode(),
            defs::SFPSHFT.opcode(),
            defs::SFPIADD.opcode(),
            defs::SFPNOT.opcode(),
            defs::SFPEXEXP.opcode(),
            defs::SFPGT.opcode(),
            defs::SFPSETCC.opcode(),
        );
        let (pushc, popc, compc) = (defs::SFPPUSHC.opcode(), defs::SFPPOPC.opcode(), defs::SFPCOMPC.opcode());
        let sfp = |ops: &mut Vec<Op>, op: u8, args: [u16; 5]| {
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op, args })));
        };
        let konst = |ops: &mut Vec<Op>, reg: u8, bits: u32| {
            ops.push(Op::TT(TTOp::SfpConst(Box::new(TTSfpConst { reg, bits }))));
        };
        // Lane-flag predication on, all lanes enabled, as `Program::new`
        // does: without `SFPENCC` every scope below is a no-op.
        sfp(&mut ops, defs::SFPENCC.opcode(), [3, 0, 10, 0, 0]);
        let floor = kind == sf::FLOOR;
        let body = self.tt_row_loop(|mut ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: INT32 }))));
            konst(&mut ops, L3, 0x7fff_ffff);
            sfp(&mut ops, and, [u16::from(L3), u16::from(L0), u16::from(L1), 1, 0]);
            sfp(&mut ops, mov, [u16::from(L0), u16::from(L2), 0, 0, 0]);
            konst(&mut ops, L3, 0x4b00_0000); // 2^23
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, gt, [u16::from(L1), u16::from(L3), 1, 0, 0]);
            konst(&mut ops, L3, 0x3f80_0000); // 1.0
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, gt, [u16::from(L1), u16::from(L3), 1, 0, 0]);
            konst(&mut ops, L4, 0x8000_0000);
            sfp(&mut ops, and, [u16::from(L4), u16::from(L0), u16::from(L2), 1, 0]);
            if floor {
                sfp(&mut ops, pushc, [0; 5]);
                sfp(&mut ops, setcc, [0, u16::from(L0), 0, 0, 0]);
                sfp(&mut ops, pushc, [0; 5]);
                sfp(&mut ops, setcc, [0, u16::from(L1), 0, 2, 0]);
                konst(&mut ops, L2, 0xbf80_0000); // -1.0
                sfp(&mut ops, popc, [0; 5]);
                sfp(&mut ops, popc, [0; 5]);
            }
            sfp(&mut ops, compc, [0; 5]);
            sfp(&mut ops, exexp, [u16::from(L0), u16::from(L3), 0, 0, 0]);
            konst(&mut ops, L4, 23);
            sfp(&mut ops, iadd, [0, u16::from(L4), u16::from(L3), 6, 0]);
            konst(&mut ops, L4, 1);
            sfp(&mut ops, shft, [0, u16::from(L3), u16::from(L4), 0, 0]);
            sfp(&mut ops, iadd, [0xfff, u16::from(L4), u16::from(L5), 5, 0]);
            sfp(&mut ops, and, [u16::from(L5), u16::from(L0), u16::from(L6), 1, 0]);
            sfp(&mut ops, not, [u16::from(L5), u16::from(L5), 0, 0, 0]);
            sfp(&mut ops, and, [u16::from(L5), u16::from(L0), u16::from(L2), 1, 0]);
            if floor {
                sfp(&mut ops, pushc, [0; 5]);
                sfp(&mut ops, setcc, [0, u16::from(L0), 0, 0, 0]);
                sfp(&mut ops, pushc, [0; 5]);
                sfp(&mut ops, setcc, [0, u16::from(L6), 0, 2, 0]);
                sfp(&mut ops, iadd, [0, u16::from(L4), u16::from(L2), 4, 0]);
                sfp(&mut ops, popc, [0; 5]);
                sfp(&mut ops, popc, [0; 5]);
            }
            sfp(&mut ops, popc, [0; 5]);
            sfp(&mut ops, popc, [0; 5]);
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L2, fmt: INT32 }))));
        });
        ops.extend(body);
        ops
    }

    /// `binary_body`'s `DIV`: `scale_large_divisor`, `recip`, `divide`.
    fn tt_sfp_div(&mut self, a: OpId, b: OpId, slot: OpId) -> Vec<Op> {
        let mut ops: Vec<Op> = Vec::new();
        // `binary_body`'s `DIV`: scale a divisor above 2^100 by 2^-64,
        // `recip` it, then one correction step. `a = L0`, `b = L1`,
        // `y = L2`, `q = L5`; `L6` holds `f32::MAX`, `L7` `+inf`.
        use tt_isa::isa::generated::defs;
        let (mov, abs, and, arecip, setsgn, gt, mad) = (
            defs::SFPMOV.opcode(),
            defs::SFPABS.opcode(),
            defs::SFPAND.opcode(),
            defs::SFPARECIP.opcode(),
            defs::SFPSETSGN.opcode(),
            defs::SFPGT.opcode(),
            defs::SFPMAD.opcode(),
        );
        let (pushc, popc, compc) = (defs::SFPPUSHC.opcode(), defs::SFPPOPC.opcode(), defs::SFPCOMPC.opcode());
        let sfp = |ops: &mut Vec<Op>, op: u8, args: [u16; 5]| {
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op, args })));
        };
        let konst = |ops: &mut Vec<Op>, reg: u8, bits: u32| {
            ops.push(Op::TT(TTOp::SfpConst(Box::new(TTSfpConst { reg, bits }))));
        };
        // Lane-flag predication on, all lanes enabled, as `Program::new`
        // does: without `SFPENCC` every scope below is a no-op.
        sfp(&mut ops, defs::SFPENCC.opcode(), [3, 0, 10, 0, 0]);
        let body = self.tt_row_loop(|mut ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: FP32 }))));
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: b, at: o, reg: L1, fmt: FP32 }))));
            // The special cases' two constants; inside the body, so a fused
            // body writes every register it reads.
            konst(&mut ops, L6, 0x7f7f_ffff); // f32::MAX
            konst(&mut ops, L7, 0x7f80_0000); // +inf
            // `scale_large_divisor(a, b, t = L3, c = L4)`.
            sfp(&mut ops, abs, [u16::from(L1), u16::from(L3), 1, 0, 0]);
            konst(&mut ops, L4, 0x7180_0000); // 2^100
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, gt, [u16::from(L4), u16::from(L3), 1, 0, 0]);
            konst(&mut ops, L4, 0x1f80_0000); // 2^-64
            sfp(&mut ops, mad, [u16::from(L0), u16::from(L4), LREG_ZERO, u16::from(L0), 2]);
            sfp(&mut ops, mad, [u16::from(L1), u16::from(L4), LREG_ZERO, u16::from(L1), 2]);
            sfp(&mut ops, popc, [0; 5]);
            // `recip(x = L1, d = L2, t0 = L3, t1 = L4, max = L6)`.
            sfp(&mut ops, abs, [u16::from(L1), u16::from(L4), 1, 0, 0]);
            konst(&mut ops, L3, 0x7180_0000);
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, gt, [u16::from(L3), u16::from(L4), 1, 0, 0]);
            konst(&mut ops, L6, 0x1f80_0000);
            sfp(&mut ops, mad, [u16::from(L1), u16::from(L6), LREG_ZERO, u16::from(L4), 2]);
            sfp(&mut ops, arecip, [0, u16::from(L4), u16::from(L2), 0, 0]);
            for _ in 0..2 {
                sfp(&mut ops, mad, [u16::from(L4), u16::from(L2), LREG_ONE, u16::from(L3), 1]);
                sfp(&mut ops, mad, [u16::from(L3), u16::from(L2), u16::from(L2), u16::from(L2), 0]);
            }
            sfp(&mut ops, mad, [u16::from(L2), u16::from(L6), LREG_ZERO, u16::from(L2), 2]);
            konst(&mut ops, L6, 0x7f7f_ffff);
            sfp(&mut ops, compc, [0; 5]);
            sfp(&mut ops, arecip, [0, u16::from(L1), u16::from(L2), 0, 0]);
            for _ in 0..2 {
                sfp(&mut ops, mad, [u16::from(L1), u16::from(L2), LREG_ONE, u16::from(L3), 1]);
                sfp(&mut ops, mad, [u16::from(L3), u16::from(L2), u16::from(L2), u16::from(L2), 0]);
            }
            sfp(&mut ops, popc, [0; 5]);
            konst(&mut ops, L3, 0x7fff_ffff);
            sfp(&mut ops, and, [u16::from(L3), u16::from(L1), u16::from(L4), 1, 0]);
            konst(&mut ops, L3, 0x0080_0000);
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, gt, [u16::from(L4), u16::from(L3), 1, 0, 0]);
            konst(&mut ops, L3, 0x7f80_0000);
            sfp(&mut ops, mov, [u16::from(L1), u16::from(L2), 0, 0, 0]);
            sfp(&mut ops, setsgn, [0, u16::from(L3), u16::from(L2), 0, 0]);
            sfp(&mut ops, popc, [0; 5]);
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, gt, [u16::from(L6), u16::from(L4), 1, 0, 0]);
            konst(&mut ops, L3, 0x7f80_0001);
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, gt, [u16::from(L4), u16::from(L3), 1, 0, 0]);
            sfp(&mut ops, mov, [u16::from(L1), u16::from(L2), 0, 0, 0]);
            sfp(&mut ops, setsgn, [0, LREG_ZERO, u16::from(L2), 0, 0]);
            sfp(&mut ops, popc, [0; 5]);
            sfp(&mut ops, popc, [0; 5]);
            // `divide(a, b, y, q = L5, r = L3, t = L4, inf = L7)`.
            sfp(&mut ops, mad, [u16::from(L0), u16::from(L2), LREG_ZERO, u16::from(L5), 2]);
            sfp(&mut ops, abs, [u16::from(L5), u16::from(L4), 1, 0, 0]);
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, gt, [LREG_ZERO, u16::from(L4), 1, 0, 0]);
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, gt, [u16::from(L4), u16::from(L7), 1, 0, 0]);
            sfp(&mut ops, mad, [u16::from(L1), u16::from(L5), u16::from(L0), u16::from(L3), 1]);
            sfp(&mut ops, mad, [u16::from(L3), u16::from(L2), u16::from(L5), u16::from(L5), 0]);
            sfp(&mut ops, popc, [0; 5]);
            sfp(&mut ops, popc, [0; 5]);
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L5, fmt: FP32 }))));
        });
        ops.extend(body);
        ops
    }

    /// `Op::Bitcast`: the same 32 bits in a new tile. A lane register holds
    /// one word whatever its type, so this is a raw copy, `INT32` in and out,
    /// and no SFPU instruction: the copy gives the value its own storage and
    /// target dtype, and a later pass can coalesce it away.
    fn tt_sfp_bitcast(&mut self, a: OpId, slot: OpId) -> Vec<Op> {
        self.tt_row_loop(|mut ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: INT32 }))));
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L0, fmt: INT32 }))));
        })
    }

    /// `Op::Cast` from `I32` to `F32`: `ops::i32_to_float` — the two's
    /// complement made sign-magnitude for `SFPCAST` (which rounds to nearest
    /// even), the sign put back; `i32::MIN`, whose magnitude has no 31-bit
    /// form, by name. `L0` in and out, `L1` and `L2` scratch.
    fn tt_sfp_cast_i32_f32(&mut self, a: OpId, slot: OpId) -> Vec<Op> {
        use tt_isa::isa::generated::defs;
        let (mov, cast, xor, iadd, setcc) = (
            defs::SFPMOV.opcode(),
            defs::SFPCAST.opcode(),
            defs::SFPXOR.opcode(),
            defs::SFPIADD.opcode(),
            defs::SFPSETCC.opcode(),
        );
        let (pushc, popc) = (defs::SFPPUSHC.opcode(), defs::SFPPOPC.opcode());
        let sfp = |ops: &mut Vec<Op>, op: u8, args: [u16; 5]| {
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op, args })));
        };
        let konst = |ops: &mut Vec<Op>, reg: u8, bits: u32| {
            ops.push(Op::TT(TTOp::SfpConst(Box::new(TTSfpConst { reg, bits }))));
        };
        let mut ops: Vec<Op> = Vec::new();
        sfp(&mut ops, defs::SFPENCC.opcode(), [3, 0, 10, 0, 0]);
        let body = self.tt_row_loop(|mut ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: INT32 }))));
            sfp(ops, mov, [u16::from(L0), u16::from(L1), 0, 0, 0]);
            sfp(ops, pushc, [0; 5]);
            sfp(ops, setcc, [0, u16::from(L0), 0, 0, 0]);
            sfp(ops, iadd, [0, LREG_ZERO, u16::from(L1), 6, 0]);
            sfp(ops, popc, [0; 5]);
            sfp(ops, cast, [u16::from(L1), u16::from(L1), 0, 0, 0]);
            sfp(ops, pushc, [0; 5]);
            sfp(ops, setcc, [0, u16::from(L0), 0, 0, 0]);
            sfp(ops, mov, [u16::from(L1), u16::from(L1), 1, 0, 0]);
            sfp(ops, popc, [0; 5]);
            konst(ops, L2, 0x8000_0000);
            sfp(ops, xor, [u16::from(L0), u16::from(L2), 0, 0, 0]);
            sfp(ops, pushc, [0; 5]);
            sfp(ops, setcc, [0, u16::from(L2), 0, 6, 0]);
            konst(ops, L1, 0xcf00_0000); // -2^31
            sfp(ops, popc, [0; 5]);
            sfp(ops, mov, [u16::from(L1), u16::from(L0), 0, 0, 0]);
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L0, fmt: FP32 }))));
        });
        ops.extend(body);
        ops
    }

    /// `Op::Cast` from `F32` to `I32`: `round::to_i32` — Rust's saturating
    /// `as i32`, truncating toward zero, NaN to zero, infinities and
    /// overflow clamped. `L0` in, `L2` out, `L1` and `L3` scratch.
    fn tt_sfp_cast_f32_i32(&mut self, a: OpId, slot: OpId) -> Vec<Op> {
        use tt_isa::isa::generated::defs;
        let (and, mov, gt, le, setcc, pushc, popc, compc) = (
            defs::SFPAND.opcode(),
            defs::SFPMOV.opcode(),
            defs::SFPGT.opcode(),
            defs::SFPLE.opcode(),
            defs::SFPSETCC.opcode(),
            defs::SFPPUSHC.opcode(),
            defs::SFPPOPC.opcode(),
            defs::SFPCOMPC.opcode(),
        );
        let (exexp, exman, shft, iadd) =
            (defs::SFPEXEXP.opcode(), defs::SFPEXMAN.opcode(), defs::SFPSHFT.opcode(), defs::SFPIADD.opcode());
        let sfp = |ops: &mut Vec<Op>, op: u8, args: [u16; 5]| {
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op, args })));
        };
        let konst = |ops: &mut Vec<Op>, reg: u8, bits: u32| {
            ops.push(Op::TT(TTOp::SfpConst(Box::new(TTSfpConst { reg, bits }))));
        };
        let mut ops: Vec<Op> = Vec::new();
        sfp(&mut ops, defs::SFPENCC.opcode(), [3, 0, 10, 0, 0]);
        let body = self.tt_row_loop(|mut ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: FP32 }))));
            konst(&mut ops, L3, 0x7fff_ffff);
            sfp(&mut ops, and, [u16::from(L3), u16::from(L0), u16::from(L1), 1, 0]);
            sfp(&mut ops, mov, [LREG_ZERO, u16::from(L2), 0, 0, 0]);
            konst(&mut ops, L3, 0x3f80_0000); // 1.0
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, le, [u16::from(L1), u16::from(L3), 1, 0, 0]);
            konst(&mut ops, L3, 0x4f00_0000); // 2^31
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, gt, [u16::from(L1), u16::from(L3), 1, 0, 0]);
            sfp(&mut ops, exexp, [u16::from(L0), u16::from(L3), 0, 0, 0]);
            sfp(&mut ops, iadd, [(-23i32) as u16 & 0xfff, u16::from(L3), u16::from(L3), 5, 0]);
            sfp(&mut ops, exman, [u16::from(L0), u16::from(L2), 0, 0, 0]);
            sfp(&mut ops, shft, [0, u16::from(L3), u16::from(L2), 0, 0]);
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, setcc, [0, u16::from(L0), 0, 0, 0]);
            sfp(&mut ops, iadd, [0, LREG_ZERO, u16::from(L2), 6, 0]);
            sfp(&mut ops, popc, [0; 5]);
            sfp(&mut ops, compc, [0; 5]);
            konst(&mut ops, L3, 0x7f80_0000); // +inf
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, le, [u16::from(L3), u16::from(L1), 1, 0, 0]);
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, setcc, [0, u16::from(L0), 0, 0, 0]);
            konst(&mut ops, L2, 0x8000_0000);
            sfp(&mut ops, compc, [0; 5]);
            konst(&mut ops, L2, 0x7fff_ffff);
            sfp(&mut ops, popc, [0; 5]);
            sfp(&mut ops, popc, [0; 5]);
            sfp(&mut ops, popc, [0; 5]);
            sfp(&mut ops, popc, [0; 5]);
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L2, fmt: INT32 }))));
        });
        ops.extend(body);
        ops
    }

    /// `Op::Select`: `cond != 0` picks `t`, else `f`, on the raw bits. The
    /// condition is a comparison's `Bool` tile, which its arm stores as
    /// `INT32` 0/1; `t` and `f` share a dtype, so the answer is a bitwise
    /// choice. `L0` cond, `L1` true, `L2` false, `L3` the answer.
    fn tt_sfp_select(&mut self, cond: OpId, t: OpId, f: OpId, slot: OpId) -> Vec<Op> {
        use tt_isa::isa::generated::defs;
        let (mov, setcc) = (defs::SFPMOV.opcode(), defs::SFPSETCC.opcode());
        let (pushc, popc) = (defs::SFPPUSHC.opcode(), defs::SFPPOPC.opcode());
        let sfp = |ops: &mut Vec<Op>, op: u8, args: [u16; 5]| {
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op, args })));
        };
        let mut ops: Vec<Op> = Vec::new();
        sfp(&mut ops, defs::SFPENCC.opcode(), [3, 0, 10, 0, 0]);
        let body = self.tt_row_loop(|mut ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: cond, at: o, reg: L0, fmt: INT32 }))));
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: t, at: o, reg: L1, fmt: INT32 }))));
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: f, at: o, reg: L2, fmt: INT32 }))));
            sfp(ops, mov, [u16::from(L2), u16::from(L3), 0, 0, 0]);
            sfp(ops, pushc, [0; 5]);
            sfp(ops, setcc, [0, u16::from(L0), 0, 2, 0]);
            sfp(ops, mov, [u16::from(L1), u16::from(L3), 0, 0, 0]);
            sfp(ops, popc, [0; 5]);
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L3, fmt: INT32 }))));
        });
        ops.extend(body);
        ops
    }

    fn tt_sfp_compare(&mut self, kind: u32, a: OpId, b: OpId, slot: OpId) -> Vec<Op> {
        use crate::backend::tenstorrent::kern::sfpu::ops::kind_sfpu as sf;
        let mut ops: Vec<Op> = Vec::new();
        // `binary_body`'s comparison: `compare_constants`' magnitude mask,
        // `+inf` and `1` live in `L4..L6`; `x = L0`, `y = L1`, the answer
        // `L2` as raw bits. A zero of either sign is made `+0`, a lane
        // with a NaN takes the unordered answer (`1` for `NE`, else `0`),
        // and the ordered test is `SFPGT`/`SFPLE`'s total order.
        use tt_isa::isa::generated::defs;
        let (and, mov, gt, ple, setcc, pushc, popc) = (
            defs::SFPAND.opcode(),
            defs::SFPMOV.opcode(),
            defs::SFPGT.opcode(),
            defs::SFPLE.opcode(),
            defs::SFPSETCC.opcode(),
            defs::SFPPUSHC.opcode(),
            defs::SFPPOPC.opcode(),
        );
        let sfp = |ops: &mut Vec<Op>, op: u8, args: [u16; 5]| {
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op, args })));
        };
        let konst = |ops: &mut Vec<Op>, reg: u8, bits: u32| {
            ops.push(Op::TT(TTOp::SfpConst(Box::new(TTSfpConst { reg, bits }))));
        };
        // Lane-flag predication on, all lanes enabled, as `Program::new`
        // does: without `SFPENCC` the unit's `UseLaneFlagsForLaneEnable`
        // is whatever the previous kernel left, and `SFPSETCC` is a no-op.
        sfp(&mut ops, defs::SFPENCC.opcode(), [3, 0, 10, 0, 0]);
        let (unordered, holds) = if kind == sf::NE {
            (u16::from(L6), LREG_ZERO)
        } else {
            (LREG_ZERO, u16::from(L6))
        };
        // The ordered test, one scope per condition (a nested `if_`):
        // `SFPLE` takes its first operand as `VD`, so `LessEq(a, b)` is
        // `SFPLE [b, a]` and `Less(a, b)` is `SFPGT [a, b]`.
        let ordered = |ops: &mut Vec<Op>, sfp: &dyn Fn(&mut Vec<Op>, u8, [u16; 5])| match kind {
            sf::EQ | sf::NE => {
                sfp(ops, pushc, [0; 5]);
                sfp(ops, ple, [u16::from(L1), u16::from(L0), 1, 0, 0]);
                sfp(ops, pushc, [0; 5]);
                sfp(ops, ple, [u16::from(L0), u16::from(L1), 1, 0, 0]);
                sfp(ops, mov, [holds, u16::from(L2), 0, 0, 0]);
                sfp(ops, popc, [0; 5]);
                sfp(ops, popc, [0; 5]);
            }
            sf::GT => {
                sfp(ops, pushc, [0; 5]);
                sfp(ops, gt, [u16::from(L1), u16::from(L0), 1, 0, 0]);
                sfp(ops, mov, [holds, u16::from(L2), 0, 0, 0]);
                sfp(ops, popc, [0; 5]);
            }
            sf::GE => {
                sfp(ops, pushc, [0; 5]);
                sfp(ops, ple, [u16::from(L0), u16::from(L1), 1, 0, 0]);
                sfp(ops, mov, [holds, u16::from(L2), 0, 0, 0]);
                sfp(ops, popc, [0; 5]);
            }
            sf::LT => {
                sfp(ops, pushc, [0; 5]);
                sfp(ops, gt, [u16::from(L0), u16::from(L1), 1, 0, 0]);
                sfp(ops, mov, [holds, u16::from(L2), 0, 0, 0]);
                sfp(ops, popc, [0; 5]);
            }
            _ => {
                sfp(ops, pushc, [0; 5]);
                sfp(ops, ple, [u16::from(L1), u16::from(L0), 1, 0, 0]);
                sfp(ops, mov, [holds, u16::from(L2), 0, 0, 0]);
                sfp(ops, popc, [0; 5]);
            }
        };
        let body = self.tt_row_loop(|mut ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: INT32 }))));
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: b, at: o, reg: L1, fmt: INT32 }))));
            // The masks the comparison needs; inside the body, so a fused body
            // writes every register it reads.
            konst(ops, L4, 0x7fff_ffff);
            konst(ops, L5, 0x7f80_0000);
            konst(ops, L6, 1);
            // `ax = x & mask`, `ay = y & mask`; a zero of either sign is
            // `+0`.
            sfp(&mut ops, and, [u16::from(L4), u16::from(L0), u16::from(L3), 1, 0]);
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, setcc, [0, u16::from(L3), 0, 6, 0]);
            sfp(&mut ops, mov, [LREG_ZERO, u16::from(L0), 0, 0, 0]);
            sfp(&mut ops, popc, [0; 5]);
            sfp(&mut ops, and, [u16::from(L4), u16::from(L1), u16::from(L7), 1, 0]);
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, setcc, [0, u16::from(L7), 0, 6, 0]);
            sfp(&mut ops, mov, [LREG_ZERO, u16::from(L1), 0, 0, 0]);
            sfp(&mut ops, popc, [0; 5]);
            sfp(&mut ops, mov, [unordered, u16::from(L2), 0, 0, 0]);
            // Both magnitudes at most `+inf`; a NaN fails the first test
            // (the ordered answer stands for it).
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, ple, [u16::from(L5), u16::from(L3), 1, 0, 0]);
            sfp(&mut ops, pushc, [0; 5]);
            sfp(&mut ops, ple, [u16::from(L5), u16::from(L7), 1, 0, 0]);
            ordered(&mut ops, &sfp);
            sfp(&mut ops, popc, [0; 5]);
            sfp(&mut ops, popc, [0; 5]);
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L2, fmt: INT32 }))));
        });
        ops.extend(body);
        ops
    }

    /// Integer `Add` on raw 32-bit words: `SFPIADD` adds the integer in `VC`
    /// into `VD` (plus an immediate the form does not need here), so the copy
    /// into `L2` first makes it `L2 = a + b`. `Mod1 = 4` leaves the lane flags
    /// alone: with it clear the instruction rewrites them from the sum's sign,
    /// which would predicate everything after it.
    fn tt_sfp_int_add(&mut self, a: OpId, b: OpId, slot: OpId) -> Vec<Op> {
        use tt_isa::isa::generated::defs;
        let (mov, iadd) = (defs::SFPMOV.opcode(), defs::SFPIADD.opcode());
        let mut ops: Vec<Op> = Vec::new();
        // Lane-flag predication on, all lanes enabled: without `SFPENCC` the
        // mask is whatever the previous program left, and a flagged-off lane
        // would keep its stale word.
        ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op: defs::SFPENCC.opcode(), args: [3, 0, 10, 0, 0] })));
        let body = self.tt_row_loop(|ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: INT32 }))));
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: b, at: o, reg: L1, fmt: INT32 }))));
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op: mov, args: [u16::from(L0), u16::from(L2), 0, 0, 0] })));
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op: iadd, args: [0, u16::from(L1), u16::from(L2), 4, 0] })));
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L2, fmt: INT32 }))));
        });
        ops.extend(body);
        ops
    }

    /// Integer `Sub` on raw 32-bit words: `SFPIADD`'s two's-complement form
    /// (`SFPIADD_MOD1_ARG_2SCOMP_LREG_DST`) computes `VD = VC - VD`, so the
    /// copy of `b` into `L2` first makes it `L2 = a - b`. `Mod1 = 2 | 4`: the
    /// two's-complement bit plus `CC_NONE`, which leaves the lane flags alone.
    fn tt_sfp_int_sub(&mut self, a: OpId, b: OpId, slot: OpId) -> Vec<Op> {
        use tt_isa::isa::generated::defs;
        let (mov, isub) = (defs::SFPMOV.opcode(), defs::SFPIADD.opcode());
        let mut ops: Vec<Op> = Vec::new();
        // Lane-flag predication on, all lanes enabled: without `SFPENCC` the
        // mask is whatever the previous program left, and a flagged-off lane
        // would keep its stale word.
        ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op: defs::SFPENCC.opcode(), args: [3, 0, 10, 0, 0] })));
        let body = self.tt_row_loop(|ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: INT32 }))));
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: b, at: o, reg: L1, fmt: INT32 }))));
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op: mov, args: [u16::from(L1), u16::from(L2), 0, 0, 0] })));
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op: isub, args: [0, u16::from(L0), u16::from(L2), 2 | 4, 0] })));
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L2, fmt: INT32 }))));
        });
        ops.extend(body);
        ops
    }

    /// `integer::body` ops 3..7 (`AND`/`OR`/`XOR`/`SHL`/`SHR`).
    fn tt_sfp_integer(&mut self, kind: u32, a: OpId, b: OpId, slot: OpId) -> Vec<Op> {
        use crate::backend::tenstorrent::kern::sfpu::ops::kind_sfpu as sf;
        let mut ops: Vec<Op> = Vec::new();
        // `integer::body` ops 3..7: `AND`/`OR`/`XOR` on the words, and
        // `SHL`/`SHR` with the shift amount taken modulo 32 (`SHR`
        // arithmetic: the sign's ones shifted in). `a = L0`, `b = L1`,
        // the result `L2`, `L3` scratch, raw bits both ways.
        use tt_isa::isa::generated::defs;
        let (and, or, xor, not, mov, shft, iadd, setcc, pushc, popc) = (
            defs::SFPAND.opcode(),
            defs::SFPOR.opcode(),
            defs::SFPXOR.opcode(),
            defs::SFPNOT.opcode(),
            defs::SFPMOV.opcode(),
            defs::SFPSHFT.opcode(),
            defs::SFPIADD.opcode(),
            defs::SFPSETCC.opcode(),
            defs::SFPPUSHC.opcode(),
            defs::SFPPOPC.opcode(),
        );
        let sfp = |ops: &mut Vec<Op>, op: u8, args: [u16; 5]| {
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op, args })));
        };
        let konst = |ops: &mut Vec<Op>, reg: u8, bits: u32| {
            ops.push(Op::TT(TTOp::SfpConst(Box::new(TTSfpConst { reg, bits }))));
        };
        // `SHR` predicates on the sign; without `SFPENCC` its `SFPSETCC`
        // is a no-op.
        sfp(&mut ops, defs::SFPENCC.opcode(), [3, 0, 10, 0, 0]);
        let op = kind - sf::INT_AND;
        let body = self.tt_row_loop(|mut ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: INT32 }))));
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: b, at: o, reg: L1, fmt: INT32 }))));
            match op {
                0 => sfp(&mut ops, and, [u16::from(L1), u16::from(L0), u16::from(L2), 1, 0]),
                1 => sfp(&mut ops, or, [u16::from(L1), u16::from(L0), u16::from(L2), 1, 0]),
                2 => {
                    sfp(&mut ops, mov, [u16::from(L0), u16::from(L2), 0, 0, 0]);
                    sfp(&mut ops, xor, [u16::from(L1), u16::from(L2), 0, 0, 0]);
                }
                // The amount's five low bits; `SFPIADD` or `SFPSHFT` would
                // wrap it mod 32 for an unsigned six-bit field otherwise.
                3 => {
                    konst(&mut ops, L3, 31);
                    sfp(&mut ops, and, [u16::from(L3), u16::from(L1), u16::from(L1), 1, 0]);
                    sfp(&mut ops, mov, [u16::from(L0), u16::from(L2), 0, 0, 0]);
                    sfp(&mut ops, shft, [0, u16::from(L1), u16::from(L2), 0, 0]);
                }
                _ => {
                    konst(&mut ops, L3, 31);
                    sfp(&mut ops, and, [u16::from(L3), u16::from(L1), u16::from(L1), 1, 0]);
                    sfp(&mut ops, mov, [u16::from(L0), u16::from(L2), 0, 0, 0]);
                    // `L1 = -amount`, so `SFPSHFT` shifts right.
                    sfp(&mut ops, iadd, [0, u16::from(LREG_ZERO), u16::from(L1), 6, 0]);
                    sfp(&mut ops, shft, [0, u16::from(L1), u16::from(L2), 0, 0]);
                    // A negative value: the ones `not(ZERO)` shifted in.
                    sfp(&mut ops, pushc, [0; 5]);
                    sfp(&mut ops, setcc, [0, u16::from(L0), 0, 0, 0]);
                    sfp(&mut ops, not, [u16::from(LREG_ZERO), u16::from(L3), 0, 0, 0]);
                    sfp(&mut ops, shft, [0, u16::from(L1), u16::from(L3), 0, 0]);
                    sfp(&mut ops, not, [u16::from(L3), u16::from(L3), 0, 0, 0]);
                    sfp(&mut ops, or, [u16::from(L2), u16::from(L3), u16::from(L2), 1, 0]);
                    sfp(&mut ops, popc, [0; 5]);
                }
            }
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L2, fmt: INT32 }))));
        });
        ops.extend(body);
        ops
    }

    /// Emit the first matching pattern at `id`. The pattern table is data;
    /// this only knows the three emitter shapes (SFPU, Matmul, Reduce).
    fn tt_emit(&mut self, id: OpId, emit: TtEmit, m: &Bindings, lower: &mut TtLowering) -> Option<(DType, OpId)> {
        match emit {
            TtEmit::Sfpu { kind, arity } => {
                let (dtype, a) = self.tt_resolve_dst(m.op('x'), id, lower)?;
                let b = if arity == 2 {
                    let (_, b) = self.tt_resolve_dst(m.op('y'), id, lower)?;
                    b
                } else {
                    OpId::NULL
                };
                let slot = self.tt_new_dst(dtype, lower.prologue);
                if !self.tt_sfp_sequence(id, kind, a, b, slot) {
                    self.ops[id].op = Op::TT(TTOp::Sfpu { kind, a, b, slot });
                }
                self.tt_materialize(id, dtype, slot, lower)
            }
            TtEmit::SfpuConst { kind } => {
                let Op::Binary { x, y, .. } = self.ops[id].op else {
                    return None;
                };
                let (tile, konst) = if matches!(self.ops[x].op, Op::Const(_)) {
                    (y, x)
                } else {
                    (x, y)
                };
                let (dtype, a) = self.tt_resolve_dst(tile, id, lower)?;
                let (_, kb) = self.tt_fill_const(konst, id, lower.prologue)?;
                let slot = self.tt_new_dst(dtype, lower.prologue);
                let (a, b) = if tile == x { (a, kb) } else { (kb, a) };
                if !self.tt_sfp_sequence(id, kind, a, b, slot) {
                    self.ops[id].op = Op::TT(TTOp::Sfpu { kind, a, b, slot });
                }
                self.tt_materialize(id, dtype, slot, lower)
            }
            TtEmit::Reduce(dim) => {
                let Op::Reduce { rop, .. } = self.ops[id].op else {
                    return None;
                };
                let (dtype, input) = self.tt_resolve_dst(m.op('t'), id, lower)?;
                let slot = self.tt_new_dst(dtype, lower.prologue);
                // `reduce.rs`'s programs as row ops: a tile-row reduce is
                // `accumulate(op, Cols, first)` then `finish(op, Cols)`; one
                // over the tile's columns is the in-order/transpose fold; a
                // whole-tile one runs the first pass, then a second over rows
                // through a scratch tile, and copies the scalar back.
                if !matches!(
                    (rop, dim),
                    (BOp::Add, TileDim::Row)
                        | (BOp::Max, TileDim::Row)
                        | (BOp::Add, TileDim::Col)
                        | (BOp::Max, TileDim::Col)
                        | (BOp::Add, TileDim::Scalar)
                        | (BOp::Max, TileDim::Scalar)
                ) {
                    todo!("reduce {rop:?}/{dim:?} has no native row-op program");
                }
                // The scalar reduce's second pass reads the first pass's whole
                // result, so it needs a tile of its own. Declaring it here (and
                // not in codegen, as `fresh_slot` did) makes it a register the
                // allocator numbers and can place; liveness keeps it off the
                // accumulator's row, which `accumulate_in_order` requires (it
                // zeroes its output before reading its input).
                let scratch = if dim == TileDim::Scalar {
                    Some(self.tt_new_dst(dtype, lower.prologue))
                } else {
                    None
                };
                {
                    use tt_isa::isa::generated::defs;
                    let (mad, mov, gt, trans, shft2, pushc, popc, nop) = (
                        defs::SFPMAD.opcode(),
                        defs::SFPMOV.opcode(),
                        defs::SFPGT.opcode(),
                        defs::SFPTRANSP.opcode(),
                        defs::SFPSHFT2.opcode(),
                        defs::SFPPUSHC.opcode(),
                        defs::SFPPOPC.opcode(),
                        defs::SFPNOP.opcode(),
                    );
                    let sum = rop == BOp::Add;
                    let fmt = if sum { FP32 } else { INT32 };
                    let mut ops: Vec<Op> = Vec::new();
                    let sfp = |ops: &mut Vec<Op>, op: u8, args: [u16; 5]| {
                        ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op, args })));
                    };
                    let ld = |ops: &mut Vec<Op>, slot: OpId, at: u16, reg: u8| {
                        ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot, at, reg, fmt }))));
                    };
                    let st = |ops: &mut Vec<Op>, slot: OpId, at: u16, reg: u8| {
                        ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at, reg, fmt }))));
                    };
                    // `into = op(other, into)`, lanewise (`combine`).
                    let combine = |ops: &mut Vec<Op>, other: u8, into: u8| {
                        if sum {
                            sfp(ops, mad, [LREG_ONE, u16::from(other), u16::from(into), u16::from(into), 0]);
                        } else {
                            // `if_(Less(into, other)) mov(other, into)`, in
                            // the sign-magnitude order of `SFPGT`.
                            sfp(ops, pushc, [0; 5]);
                            sfp(ops, gt, [u16::from(into), u16::from(other), 1, 0, 0]);
                            sfp(ops, mov, [u16::from(other), u16::from(into), 0, 0, 0]);
                            sfp(ops, popc, [0; 5]);
                        }
                    };
                    // `accumulate(op, Cols, first)`: the first tile *is* the
                    // accumulator, so it is a copy.
                    let accumulate = |ops: &mut Vec<Op>, input: OpId, out: OpId| {
                        for o in (0..64u16).step_by(2) {
                            ld(ops, input, o, L0);
                            st(ops, out, o, L0);
                        }
                    };
                    // `finish(op, Cols)`: each tile row's four faces lanewise,
                    // then the eight lanes' seven rotations. `rotate_row` reads
                    // `VD` in a mode `SFPMAD`'s stall does not see, so every
                    // rotation after an add needs an `SFPNOP` (`mod.rs`'s
                    // `stalls_automatically_after_mad`); a max is compare+move
                    // and needs none.
                    let finish_cols = |ops: &mut Vec<Op>, tile: OpId| {
                        for lower in [0u16, 32] {
                            for g in 0..4u16 {
                                let (left, right) = (lower + 4 * g, lower + 16 + 4 * g);
                                let at = [left, left | 2, right, right | 2];
                                // The fold is always through L0; the three
                                // other faces come in through L1.
                                ld(ops, tile, at[0], L0);
                                for &a in &at[1..] {
                                    ld(ops, tile, a, L1);
                                    combine(ops, L1, L0);
                                }
                                sfp(ops, mov, [u16::from(L0), u16::from(L2), 0, 0, 0]);
                                for i in 0..7 {
                                    if i > 0 && sum {
                                        sfp(ops, nop, [0; 5]);
                                    }
                                    sfp(ops, shft2, [0, u16::from(L2), u16::from(L2), 3, 0]);
                                    combine(ops, L2, L0);
                                }
                                for &a in &at {
                                    st(ops, tile, a, L0);
                                }
                            }
                        }
                    };
                    // `accumulate(Sum, Rows, first)` — `accumulate_in_order`:
                    // zero the accumulator, then add each column's rows top to
                    // bottom. A load holds four rows of eight columns; one
                    // `SFPTRANSP` puts row `i` of the four into row 0 of `L[i]`,
                    // and the running sum lives in row 0 of `L4`, which is on
                    // the transpose's diagonal and so survives it.
                    let accumulate_in_order = |ops: &mut Vec<Op>, input: OpId, out: OpId| {
                        for o in (0..64u16).step_by(2) {
                            st(ops, out, o, LREG_ZERO as u8);
                        }
                        for (top, bottom) in [(0u16, 32u16), (16, 48)] {
                            for half in [0u16, 2] {
                                sfp(ops, mov, [u16::from(LREG_ZERO), u16::from(L4), 0, 0, 0]);
                                for g in 0..8u16 {
                                    let r = if g < 4 { top + 4 * g } else { bottom + 4 * (g - 4) };
                                    ld(ops, input, r + half, L0);
                                    sfp(ops, trans, [0; 5]);
                                    for row in [L0, L1, L2, L3] {
                                        sfp(ops, mad, [LREG_ONE, u16::from(L4), u16::from(row), u16::from(L4), 0]);
                                    }
                                }
                                // Row 0 the sums, rows 1..4 zero: `L5..L7`
                                // cleared and transposed into them.
                                for l in [L5, L6, L7] {
                                    sfp(ops, mov, [u16::from(LREG_ZERO), u16::from(l), 0, 0, 0]);
                                }
                                sfp(ops, trans, [0; 5]);
                                st(ops, out, top + half, L4);
                            }
                        }
                    };
                    // `finish(Max, Rows)`: each column's eight row groups
                    // lanewise, then the four lane rows by `SFPTRANSP`.
                    let finish_rows = |ops: &mut Vec<Op>, tile: OpId| {
                        for (top, bottom) in [(0u16, 32u16), (16, 48)] {
                            for half in [0u16, 2] {
                                ld(ops, tile, top + half, L0);
                                for g in 0..8u16 {
                                    let r = if g < 4 { top + 4 * g } else { bottom + 4 * (g - 4) };
                                    if r != top {
                                        ld(ops, tile, r + half, L1);
                                        combine(ops, L1, L0);
                                    }
                                }
                                sfp(ops, trans, [0; 5]);
                                combine(ops, L1, L0);
                                combine(ops, L3, L2);
                                combine(ops, L2, L0);
                                st(ops, tile, top + half, L0);
                            }
                        }
                    };
                    // A whole-tile copy (`tt_dst_copy`): raw bits, one address
                    // at a time.
                    let copy_tile = |ops: &mut Vec<Op>, src: OpId, dst: OpId| {
                        for o in 0..64u16 {
                            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: src, at: o, reg: L0, fmt: INT32 }))));
                            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot: dst, at: o, reg: L0, fmt: INT32 }))));
                        }
                    };
                    if dim == TileDim::Col {
                        // Over the tile's rows: an in-order sum, or a copy
                        // plus the transpose fold for a maximum.
                        if sum {
                            accumulate_in_order(&mut ops, input, slot);
                        } else {
                            accumulate(&mut ops, input, slot);
                            finish_rows(&mut ops, slot);
                        }
                    } else {
                        accumulate(&mut ops, input, slot);
                        finish_cols(&mut ops, slot);
                        if let Some(scratch) = scratch {
                            if sum {
                                accumulate_in_order(&mut ops, slot, scratch);
                            } else {
                                accumulate(&mut ops, slot, scratch);
                                finish_rows(&mut ops, scratch);
                            }
                            copy_tile(&mut ops, scratch, slot);
                        }
                    }
                    let last = ops.pop().expect("the reduce sequence is never empty");
                    for op in ops {
                        self.insert_before(id, op);
                    }
                    self.ops[id].op = last;
                }
                self.tt_materialize(id, dtype, slot, lower)
            }
            TtEmit::Matmul => {
                let root = id;
                let (a, b) = (m.op('a'), m.op('b'));
                let (ta, tb) = match (&self.ops[a].op, &self.ops[b].op) {
                    (Op::Load { src: pa, src_idx: la }, Op::Load { src: pb, src_idx: lb }) => {
                        match (&self.ops[*pa].op, &self.ops[*pb].op) {
                            (Op::Param { dtype: da, .. }, Op::Param { dtype: db, .. }) => ((*pa, *la, *da), (*pb, *lb, *db)),
                            _ => return None,
                        }
                    }
                    _ => return None,
                };
                // K-plain feeder is A, K-scaled is B; roles are the order
                // binding the same K lane node. Params ride the roles, so
                // the CB lookup follows A/B, not operand order.
                let mut fab = None;
                for ((pa, la, da), (pb, lb, db)) in [(ta, tb), (tb, ta)] {
                    let kx = self.match_pat(la, &lower.pat_a).map(|m| m.op('k'));
                    let ky = self.match_pat(lb, &lower.pat_b).map(|m| m.op('k'));
                    if let (Some(x), Some(y)) = (kx, ky) {
                        if x == y && matches!(self.ops[x].op, Op::Unroll { .. }) {
                            fab = Some(((pa, da), (pb, db), x));
                            break;
                        }
                    }
                }
                let Some(((pa, dtype_a), (pb, dtype_b), k)) = fab else {
                    return None;
                };
                // The reduced axis must be the shared K lane: contracting
                // any other axis is not a matmul.
                if !matches!(&self.ops[root].op, Op::Reduce { reduce_axis, .. } if *reduce_axis == k) {
                    return None;
                }
                let (Some(&cb_a), Some(&cb_b)) = (lower.reader_cb_of.get(&pa), lower.reader_cb_of.get(&pb)) else {
                    return None;
                };
                // The native matmul is the UNPACR/MVMUL sequence, not a
                // structured op: `SrcA`/`SrcB` are two small banks the unpack
                // and math roles hand back and forth (`flip` on the unpack,
                // `flip_a`/`flip_b` on the `MVMUL` that releases one). One
                // output tile is 64 DST rows: `fi` 0 accumulates rows 0..31,
                // `MATH_AM_FI` moves the base on 32 rows for `fi` 1, so the
                // accumulator is the tile's slot directly.
                use crate::backend::tenstorrent::kern::matmul::{MATH_AM_FI, MATH_AM_GROUP, MATH_AM_HALF, MATH_AM_PHASE};
                use tt_isa::backend::ThreadConfigEntry;
                use tt_isa::cfg::generated::thread;
                // `math_prelude`: one `SETC16` per single-field entry, and a
                // whole-word `Setcfg` for SEC2..SEC4, which carry several
                // fields of one word (`math_addr_mods`).
                // Every `ThreadConfig` word goes out *composed*: a `SETC16`
                // carries the whole word, so a field's value has to be shifted
                // into its place here (`FidelityIncr = 1` is not the word `1`,
                // and the raw `1` lands on `DestIncr`, which then walks the
                // accumulator one row per multiply).
                let one = |f: tt_isa::cfg::ThreadConfigField, v: u16| -> TTOp {
                    let e = ThreadConfigEntry::zeroed(f.addr32()).set(f, v).expect("matmul thread entry is encodable");
                    TTOp::Setc16 { field: u32::from(e.addr32()), value: e.value() }
                };
                let word = |fields: &[(tt_isa::cfg::ThreadConfigField, u16)]| -> TTOp {
                    let mut e = ThreadConfigEntry::zeroed(fields[0].0.addr32());
                    for &(f, v) in fields {
                        e = e.set(f, v).expect("matmul address modifier is encodable");
                    }
                    // These are `ThreadConfig` words, so they go out as
                    // `SETC16` with the whole word — NOT as a `Config` write:
                    // the same index in the `Config` space is a packer/unpacker
                    // register (writing address modifiers there is how the
                    // packer got its `PCK0_ADDR_CTRL_XY_REG_1` fields poked).
                    TTOp::Setc16 { field: u32::from(e.addr32()), value: e.value() }
                };
                let mut before = vec![
                    one(thread::ADDR_MOD_AB_SEC0_SrcAIncr, 0),
                    one(thread::ADDR_MOD_DST_SEC0_DestIncr, 0),
                    one(thread::ADDR_MOD_AB_SEC1_SrcAIncr, 0),
                    one(thread::ADDR_MOD_DST_SEC1_FidelityIncr, 1),
                    // 2: end of a first half: `SrcB` on eight rows, phase 0.
                    word(&[
                        (thread::ADDR_MOD_AB_SEC2_SrcBIncr, 8),
                        (thread::ADDR_MOD_AB_SEC2_SrcBClear, 0),
                    ]),
                    word(&[
                        (thread::ADDR_MOD_DST_SEC2_DestIncr, 0),
                        (thread::ADDR_MOD_DST_SEC2_DestCR, 0),
                        (thread::ADDR_MOD_DST_SEC2_FidelityClear, 1),
                    ]),
                    // 3: end of a group (its last `MVMUL` is not the `fi`'s):
                    // phase and `SrcB` back to zero.
                    word(&[
                        (thread::ADDR_MOD_AB_SEC3_SrcBIncr, 0),
                        (thread::ADDR_MOD_AB_SEC3_SrcBClear, 1),
                    ]),
                    word(&[
                        (thread::ADDR_MOD_DST_SEC3_DestIncr, 0),
                        (thread::ADDR_MOD_DST_SEC3_DestCR, 0),
                        (thread::ADDR_MOD_DST_SEC3_FidelityClear, 1),
                    ]),
                    // 4: end of an `fi`: as 3, and the base on 32 rows.
                    word(&[
                        (thread::ADDR_MOD_AB_SEC4_SrcBIncr, 0),
                        (thread::ADDR_MOD_AB_SEC4_SrcBClear, 1),
                    ]),
                    word(&[
                        (thread::ADDR_MOD_DST_SEC4_DestIncr, 32),
                        (thread::ADDR_MOD_DST_SEC4_DestCR, 0),
                        (thread::ADDR_MOD_DST_SEC4_FidelityClear, 1),
                    ]),
                    one(thread::FIDELITY_BASE_Phase, 0),
                ];
                let mut after = vec![
                    // T1 clears the whole register file before the first
                    // multiply. `tt_reg_sync` owns the role hand-off, so the
                    // accumulator's produce/consume edge is added there from
                    // the `MVMUL`s themselves, not spelled here.
                    TTOp::Zeroacc { mode: 3, imm10: 0 },
                ];
                // One `MVMUL` group into rows `base..base + 16` of the `fi`:
                // both `SrcB` halves, every phase, all but the group's last
                // multiply, which the caller issues with its flips and end.
                const PHASES: u32 = 4;
                let group = |after: &mut Vec<TTOp>, base: u32| {
                    for half in 0..2u32 {
                        let phases = if half == 0 { PHASES } else { PHASES - 1 };
                        for phase in 0..phases {
                            let am = if half == 0 && phase + 1 == PHASES {
                                MATH_AM_HALF
                            } else {
                                MATH_AM_PHASE
                            };
                            after.push(TTOp::Mvmul {
                                dst_row: base + 8 * half,
                                addr_mod: am as u8,
                                flip_a: false,
                                flip_b: false,
                                broadcast_b: false,
                            });
                        }
                    }
                };
                // `matmul_items`: `fi`, then `k`, then `fj`. `A` is in0 and
                // goes to `SrcB` (face `(fi, k)`); `B` is in1 and goes to
                // `SrcA` (faces `(k, 0)` and `(k, 1)`). Each face grid step
                // unpacks a whole operand face, so every unpack flips.
                for fi in 0..2u32 {
                    for k in 0..2u32 {
                        after.push(TTOp::Setadczw { unit: 1, face: 2 * fi + k });
                        after.push(TTOp::Unpacr { src: cb_a, unit: 1, count: 1024, flip: true });
                        after.push(TTOp::Setadczw { unit: 0, face: 2 * k });
                        after.push(TTOp::Unpacr { src: cb_b, unit: 0, count: 1024, flip: true });
                        group(&mut after, 0);
                        // The group's last multiply hands `SrcA` back: the
                        // `B` face it was read from is done.
                        after.push(TTOp::Mvmul {
                            dst_row: 8,
                            addr_mod: MATH_AM_GROUP as u8,
                            flip_a: true,
                            flip_b: false,
                            broadcast_b: false,
                        });
                        after.push(TTOp::Setadczw { unit: 0, face: 2 * k + 1 });
                        after.push(TTOp::Unpacr { src: cb_b, unit: 0, count: 1024, flip: true });
                        group(&mut after, 16);
                        let end = if k == 0 { MATH_AM_GROUP } else { MATH_AM_FI };
                        after.push(TTOp::Mvmul {
                            dst_row: 24,
                            addr_mod: end as u8,
                            flip_a: true,
                            flip_b: true,
                            broadcast_b: false,
                        });
                    }
                }
                let slot = self.tt_new_dst(dtype_a, lower.prologue);
                self.ops[root].op = Op::TT(TTOp::Setrwc { dst: slot });
                for op in before.drain(..) {
                    self.insert_before(root, Op::TT(op));
                }
                let mut cur = root;
                for op in after {
                    cur = self.insert_after(cur, Op::TT(op));
                }
                // The Matrix Unit advances the thread's `Dst` row counter as it
                // walks the accumulator (`MVMUL`'s address modifiers), and an
                // `SFPLOAD`/`SFPSTORE` address is `Imm10 + RWC.Dst`: a fused
                // SFPU op after the matmul would write one tile too far. This
                // restores the counter to the accumulator's own base, which is
                // 0 for the only placement that runs today (the accumulator is
                // tile 0). A nonzero accumulator would need a true zero here,
                // and no op expresses "RWC.Dst = 0" without naming a tile.
                self.insert_after(cur, Op::TT(TTOp::Setrwc { dst: slot }));
                // The accumulator tile is not materialized, even when it has
                // several uses: its DST rows are protected by the hand-off
                // `tt_reg_sync` derives, and a pack emitted here would land
                // *before* the Matrix Unit's work, while its unpack would be T0
                // reading a T2 pack with no seam between them. So every
                // consumer reads the accumulator's DST tile in place.
                Some((dtype_a, slot))
            }
        }
    }

    /// Storage of an already-lowered value: its materialized CB, its DST
    /// row, a storage itself, or the reader CB of a load.
    fn tt_lowered_storage(&self, id: OpId, lower: &TtLowering) -> Option<(DType, OpId)> {
        if let Some(&cb) = lower.materialized.get(&id) {
            return Some((self.dtype(id), cb));
        }
        match self.ops[id].op {
            Op::TT(TTOp::Sfpu { slot, .. }) => Some((self.dtype(id), slot)),
            Op::TT(TTOp::SfpStore(ref store)) => Some((self.dtype(id), store.slot)),
            Op::TT(TTOp::Setrwc { dst }) => Some((self.dtype(id), dst)),
            Op::TT(TTOp::Copy { dst, .. }) => match self.ops[dst].op {
                Op::Storage { dtype, scope: MemScope::Register, .. } => Some((dtype, dst)),
                _ => None,
            },
            Op::Storage { dtype, .. } => Some((dtype, id)),
            Op::Load { .. } => self.tt_lower_load(id, lower),
            _ => None,
        }
    }

    /// Physical DST reuse. Every logical Register storage created by
    /// lowering is assigned a physical row by linear-scan liveness over all
    /// operands (compute, packs, unpacks). Dead inputs are released before
    /// the output is assigned, so an in-place native op can coalesce its
    /// input and output onto one row. This is the only place physical rows
    /// are chosen; lowering stays SSA.
    pub fn tt_dst_reuse(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        // Before allocating: a materialised value whose row can survive to its
        // last consumer does not need the L1 round trip at all.
        self.tt_reclaim_round_trips();
        let mut prologue = self.head;
        for _ in 0..250_000 {
            if prologue.is_null() {
                break;
            }
            match self.ops[prologue].op {
                Op::Param { .. } | Op::Storage { .. } => prologue = self.ops[prologue].next,
                _ => break,
            }
        }
        let mut ops: Vec<OpId> = Vec::new();
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() {
                break;
            }
            ops.push(id);
            id = self.next_op(id);
        }
        // Last use of every logical Register storage over all operands.
        let mut last_use: Map<OpId, usize> = Map::default();
        for (i, &op) in ops.iter().enumerate() {
            for p in self.ops[op].op.parameters() {
                if matches!(self.ops[p].op, Op::Storage { scope: MemScope::Register, .. }) {
                    last_use.insert(p, i);
                }
            }
        }
        let mut physical: Map<OpId, OpId> = Map::default();
        let mut active: Vec<(OpId, OpId, usize)> = Vec::new();
        let mut free: Vec<(DType, OpId)> = Vec::new();
        for (i, &op) in ops.iter().enumerate() {
            // Release rows whose last use was strictly before this op.
            let mut k = 0;
            while k < active.len() {
                if active[k].2 < i {
                    let (_, phys, _) = active.remove(k);
                    if let Op::Storage { dtype, .. } = self.ops[phys].op {
                        free.push((dtype, phys));
                    }
                } else {
                    k += 1;
                }
            }
            let regs: Vec<OpId> = self.ops[op]
                .op
                .parameters()
                .filter(|p| matches!(self.ops[*p].op, Op::Storage { scope: MemScope::Register, .. }))
                .collect();
            if regs.is_empty() {
                continue;
            }
            let output = self.tt_register_output(&self.ops[op].op);
            for &p in &regs {
                if Some(p) == output {
                    continue;
                }
                self.tt_ensure_phys(&mut physical, &mut active, &mut free, &last_use, p, prologue);
            }
            if let Some(out) = output {
                // Release inputs whose last use is this op before assigning
                // the output, so a dead input's row can be reused in place.
                let mut k = 0;
                while k < active.len() {
                    if active[k].2 == i && active[k].0 != out {
                        let (_, phys, _) = active.remove(k);
                        if let Op::Storage { dtype, .. } = self.ops[phys].op {
                            free.push((dtype, phys));
                        }
                    } else {
                        k += 1;
                    }
                }
                self.tt_ensure_phys(&mut physical, &mut active, &mut free, &last_use, out, prologue);
            }
        }
        // Rewrite every logical operand to its physical row.
        let mut rewrites: Vec<(OpId, usize, OpId)> = Vec::new();
        for &op in &ops {
            for (idx, p) in self.ops[op].op.parameters().enumerate() {
                if let Some(&phys) = physical.get(&p) {
                    if phys != p {
                        rewrites.push((op, idx, phys));
                    }
                }
            }
        }
        for (op, idx, phys) in rewrites {
            if let Some(slot) = self.ops[op].op.parameters_mut().nth(idx) {
                *slot = phys;
            }
        }
        // The register file is now exactly the physical storages this pass
        // created: drop the logical Register declarations they replaced, so
        // the prologue's ordered Register storages are the file codegen
        // numbers (slot i is the i-th declaration). A logical declaration
        // left behind would otherwise count against the DST budget and
        // shift every row after it.
        let live: Set<OpId> = physical.values().copied().collect();
        let mut dead: Vec<OpId> = Vec::new();
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() {
                break;
            }
            if matches!(self.ops[id].op, Op::Storage { scope: MemScope::Register, .. }) && !live.contains(&id) {
                dead.push(id);
            }
            id = self.next_op(id);
        }
        for id in dead {
            self.remove_op(id);
        }
        // The file is exactly the surviving physical storages, in prologue
        // order: stamp each with its allocation index. From here on the row is
        // the storage's own `id` (`id * 32` or `id * 64` rows), so codegen
        // reads the file instead of numbering slots by first use, and a
        // post-allocation CSE sees two registers as two storages.
        let mut next = 0u32;
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() {
                break;
            }
            if let Op::Storage { scope: MemScope::Register, id: slot, .. } = &mut self.ops[id].op {
                *slot = next;
                next += 1;
            }
            id = self.next_op(id);
        }
    }

    /// Classify a tile `Copy` by the scopes it moves between.
    fn tt_copy_kind(&self, id: OpId) -> TtCopy {
        let Op::TT(TTOp::Copy { src, dst, .. }) = self.ops[id].op else {
            panic!("tt_copy_kind: {id:?} is not a tile copy");
        };
        let scope = |o: OpId| match self.ops[o].op {
            Op::Param { .. } => 'p',
            Op::Storage { scope: MemScope::Register, .. } => 'r',
            Op::Storage { scope: MemScope::Circular, .. } => 'c',
            _ => '?',
        };
        match (scope(src), scope(dst)) {
            ('c', 'r') => TtCopy::Unpack,
            ('r', 'c') => TtCopy::Pack,
            ('p', 'c') => TtCopy::Publish,
            ('c', 'p') => TtCopy::Drain,
            _ => panic!("tt_copy_kind: copy {id:?} moves between scopes with no stream"),
        }
    }

    /// Fold a uniform constant tile back into its readers, before fusion.
    ///
    /// `tt_fill_const` materialises every scalar constant as a whole DST tile:
    /// one `SfpConst` immediate plus 32 `SfpStore`s (rows 0, 2, .. 62 of the
    /// 16-lane DST, the full 32x32 tile). Every reader then pays an `SfpLoad`
    /// and the tile takes one of the eight. On `sigmoid` that is 22 tiles and
    /// about 700 of the kernel's 734 DST stores: the constants, not the seams,
    /// are the traffic. A uniform tile's read is row-invariant, so each
    /// `SfpLoad` can become the very `SfpConst` the splat already used.
    ///
    /// Exact conditions: the slot is a `Register` storage whose only writer is
    /// one splat (`SfpConst` followed by exactly the 32 stores, same register,
    /// same format, ascending `at`), nothing else names it (no pack, no unpack,
    /// no `Sfpu` result), and every reader is an `SfpLoad` with the splat's
    /// format, so the lane word is bit-identical. The reads are rewritten in
    /// place, each carrying its word-count change to the owning `Replay`'s
    /// `len` (a `SfpConst` is two words unless its low half is zero), then the
    /// splat and the tile go. Skipped, never forced: a tile that fails any of
    /// these keeps its splat.
    fn tt_sink_const_splats(&mut self) {
        for _ in 0..250_000 {
            let mut ops: Vec<OpId> = Vec::new();
            let mut id = self.head;
            for _ in 0..250_000 {
                if id.is_null() {
                    break;
                }
                ops.push(id);
                id = self.next_op(id);
            }
            // Which `Replay` owns each body op, so a rewritten load can carry
            // its word-count change to that marker's `len`.
            let mut owner: Vec<Option<usize>> = vec![None; ops.len()];
            let mut i = 0;
            while i < ops.len() {
                if let Op::TT(TTOp::Replay { len, .. }) = self.at(ops[i]) {
                    let mut words = 0u32;
                    let mut j = i + 1;
                    while j < ops.len() && words < u32::from(*len) {
                        words += tt_row_words(self.at(ops[j]));
                        owner[j] = Some(i);
                        j += 1;
                    }
                    i = j;
                } else {
                    i += 1;
                }
            }
            // One splat: `SfpConst` + the 32 stores of a whole tile.
            let mut found: Option<(usize, OpId, u32, u8)> = None;
            for i in 0..ops.len() {
                let Op::TT(TTOp::SfpConst(c)) = self.at(ops[i]) else {
                    continue;
                };
                let (i_next, bits, reg) = (i + 1, c.bits, c.reg);
                let i_end = i_next + 32;
                if i_end > ops.len() || (i..i_end).any(|k| owner[k].is_some()) {
                    continue;
                }
                let mut pair: Option<(OpId, u8)> = None;
                let mut ok = true;
                for (k, &op) in ops[i_next..i_end].iter().enumerate() {
                    let Op::TT(TTOp::SfpStore(s)) = self.at(op) else {
                        ok = false;
                        break;
                    };
                    if s.reg != reg || s.at != (k as u16) * 2 {
                        ok = false;
                        break;
                    }
                    match pair {
                        None => pair = Some((s.slot, s.fmt)),
                        Some((slot, fmt)) => {
                            if slot != s.slot || fmt != s.fmt {
                                ok = false;
                                break;
                            }
                        }
                    }
                }
                if !ok {
                    continue;
                }
                let Some((slot, fmt)) = pair else { continue };
                // Only 32-bit formats: `FP32` and `INT32` both move the lane's
                // raw word, which is what a `Bitcast` of the constant asks for;
                // a 16-bit load would convert and is not bit-equal.
                let raw = |f: u8| f == FP32 || f == INT32;
                if !raw(fmt) {
                    continue;
                }
                if !matches!(self.at(slot), Op::Storage { scope: MemScope::Register, .. }) {
                    continue;
                }
                // Only splat stores, and loads of the same format, may name it.
                let mut loads = 0;
                let mut exclusive = true;
                for (j, &op) in ops.iter().enumerate() {
                    if (i_next..i_end).contains(&j) {
                        continue;
                    }
                    if !self.at(op).parameters().any(|p| p == slot) {
                        continue;
                    }
                    match self.at(op) {
                        Op::TT(TTOp::SfpLoad(l)) if raw(l.fmt) => loads += 1,
                        _ => {
                            exclusive = false;
                            break;
                        }
                    }
                }
                if exclusive && loads > 0 {
                    found = Some((i, slot, bits, fmt));
                    break;
                }
            }
            let Some((i, slot, bits, _)) = found else {
                return;
            };
            let run: Vec<OpId> = ops[i..i + 33].to_vec();
            let new_words = if bits & 0xffff == 0 { 1i32 } else { 2i32 };
            for (j, &op) in ops.iter().enumerate() {
                let Op::TT(TTOp::SfpLoad(l)) = self.at(op) else {
                    continue;
                };
                if l.slot != slot {
                    continue;
                }
                let reg = l.reg;
                self.ops[op].op = Op::TT(TTOp::SfpConst(Box::new(TTSfpConst { reg, bits })));
                let delta = new_words - 1;
                if delta != 0 {
                    let Some(m) = owner[j] else {
                        panic!("tt_sink_const_splats: a load outside any row loop");
                    };
                    let Op::TT(TTOp::Replay { len, .. }) = &mut self.ops[ops[m]].op else {
                        unreachable!("the owner of a body op is a `Replay`")
                    };
                    *len = (*len as i32 + delta) as u16;
                }
            }
            for op in run {
                self.remove_op(op);
            }
            self.remove_op(slot);
        }
    }

    /// Keep an intermediate in its lane register instead of writing it to its
    /// own tile and reading it back: `tt_sfpu_fusion`.
    ///
    /// Lowering emits one row loop per op, so a chain does
    ///
    /// ```text
    /// body A:  SFPLOAD r0 <- slot S  ...  SFPSTORE r2 -> slot S
    /// body B:  SFPLOAD r0 <- slot S  ...  SFPSTORE r3 -> slot T
    /// ```
    ///
    /// and `B`'s load reads back exactly what `A`'s store just wrote. This pass
    /// fuses a whole *cone* of such bodies into one and drops every store and
    /// load that only moved a value between two bodies of the cone.
    ///
    /// A value is *interior* to a cone when exactly one unit writes it and every
    /// read of it is in a later unit of the same cone; the cone is the
    /// reader-closure of its first unit, so no interior value is read outside
    /// it. Interior values then live in lane registers, and the consumer's
    /// `SFPLOAD` becomes the `SFPMOV` that hands the word over. That is what
    /// makes a composed activation (`elu = relu(x) - alpha*(1-exp(-x)).relu()`)
    /// cost the same as a hand-written one: one load of the input tile, one
    /// store of the output tile, no per-op tile and no per-op round trip.
    ///
    /// It is exact, not a heuristic, and it touches DST only -- a `Copy` in
    /// between makes the two sequences live in different roles (T2/T0), so they
    /// are not adjacent and never fuse. Two things bound a cone:
    ///
    /// - **registers**. Eight lane registers. A value that must survive into a
    ///   later unit is *live*, and a unit that writes a live register would
    ///   destroy it, so the cone stops before that unit and is retried one unit
    ///   shorter. Shuffling the value into a free register is possible in
    ///   principle and not modelled yet: a cone never guesses.
    /// - **the program slot**. A body longer than the 32-word replay buffer
    ///   becomes a block repeat and the runner's header holds at most
    ///   `loops::MAX` of them; a fusion that would need one more is refused.
    ///
    /// A gap between two bodies may hold only lane constants (row-invariant, so
    /// they run inside the merged body between its two parts) and the flags'
    /// enable `SFPENCC`. That one is *not* row state: `SFPENCC` (mod1 `2 | 8`)
    /// sets `use_flags` and resets every lane flag to true, so a body's
    /// preamble is a predicate reset, and a merged body that interleaves rows
    /// must reset again between its parts. It does: one `SFPENCC` leads the
    /// merged body and every gap's own copy stays where it was. (Hoisting the
    /// reset out of the body, or dropping it, silently produced `-inf` in one
    /// lane of `log2`.)
    pub fn tt_sfpu_fusion(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        // Reclaim first: a value whose row can survive to its last consumer
        // does not need its L1 round trip, and the pack/unpack pair is exactly
        // the `Copy` that keeps two bodies in different roles and out of one
        // cone. That is dst allocation (step 3 of the target design), which
        // precedes grouping (step 5); leaving it until `tt_dst_reuse` made every
        // materialised value a fusion barrier. Measured on the gates: sigmoid
        // 25 -> 15 loads, 10 -> 8 stores; sin_cos 59 -> 26 loads, 27 -> 18
        // stores. `tt_dst_reuse` runs it again after the fusion, where removing
        // an interior store can make a new pair reclaimable.
        self.tt_reclaim_round_trips();
        // Constant splats next: a whole-tile constant between two bodies is
        // both traffic and a barrier to fusion (the pair is no longer
        // adjacent), so sinking it into its readers is a precondition here.
        self.tt_sink_const_splats();
        for _ in 0..250_000 {
            let mut ops: Vec<OpId> = Vec::new();
            let mut id = self.head;
            for _ in 0..250_000 {
                if id.is_null() {
                    break;
                }
                ops.push(id);
                id = self.next_op(id);
            }
            // Every `Replay` marker and the body it owns, as index ranges.
            let mut units: Vec<(usize, usize)> = Vec::new();
            let mut i = 0;
            while i < ops.len() {
                let Op::TT(TTOp::Replay { len, .. }) = self.at(ops[i]) else {
                    i += 1;
                    continue;
                };
                let mut words = 0u32;
                let mut j = i + 1;
                while j < ops.len() && words < u32::from(*len) {
                    words += tt_row_words(self.at(ops[j]));
                    j += 1;
                }
                units.push((i, j));
                i = j;
            }
            if !self.tt_fuse_cone(&ops, &units) {
                return;
            }
        }
    }

    /// One cone of adjacent row-loop units, fused into one body. See
    /// [`Kernel::tt_sfpu_fusion`] for the rule; returns whether anything was
    /// spliced.
    fn tt_fuse_cone(&mut self, ops: &[OpId], units: &[(usize, usize)]) -> bool {
        use tt_isa::isa::generated::defs;
        let n = units.len();
        if n < 2 {
            return false;
        }
        let buffer = tt_isa::frontend::REPLAY_BUFFER;
        // Per unit: repeat count, the length word, the registers the body
        // writes, the body's words, and its loads/stores as `(slot, reg, op)`.
        let mut counts = vec![0u16; n];
        let mut lens = vec![0u32; n];
        let mut writes: Vec<Set<u16>> = vec![Set::default(); n];
        let mut words = vec![0u32; n];
        let mut loads: Vec<Vec<(OpId, u8, u8, OpId)>> = vec![Vec::new(); n];
        let mut stores: Vec<Vec<(OpId, u8, u8, OpId)>> = vec![Vec::new(); n];
        // Registers written by the gap *before* a unit: those consts end up
        // inside a merged body, so they clobber like the body does.
        let mut gap_writes: Vec<Set<u16>> = vec![Set::default(); n];
        for u in 0..n {
            let (m, end) = units[u];
            let Op::TT(TTOp::Replay { len, count }) = self.at(ops[m]) else {
                return false;
            };
            counts[u] = *count;
            lens[u] = u32::from(*len);
            for &op in &ops[m + 1..end] {
                words[u] += tt_row_words(self.at(op));
                writes[u].extend(tt_row_regs(self.at(op)).1);
                match self.at(op) {
                    Op::TT(TTOp::SfpLoad(l)) => loads[u].push((l.slot, l.reg, l.fmt, op)),
                    Op::TT(TTOp::SfpStore(s)) => stores[u].push((s.slot, s.reg, s.fmt, op)),
                    _ => {}
                }
            }
            if u > 0 {
                for &op in &ops[units[u - 1].1..m] {
                    if matches!(self.at(op), Op::TT(TTOp::SfpConst(_))) {
                        gap_writes[u].extend(tt_row_regs(self.at(op)).1);
                    }
                }
            }
        }
        // Who reads and who writes each slot, over the whole kernel.
        let mut readers: Map<OpId, Vec<usize>> = Map::default();
        let mut producers: Map<OpId, Vec<usize>> = Map::default();
        // Every read's format. A hand-over is `SFPMOV` between lane registers,
        // so it is bit-equal when the producer's store left the bits alone:
        // `INT32` is a raw copy (and any load format then reads the same bits),
        // while any other format is equal only to a load reading that same
        // format (`FP32` flushes denormals on the way into the tile).
        let mut load_fmts: Map<OpId, Vec<u8>> = Map::default();
        for u in 0..n {
            for (slot, _, fmt, _) in &loads[u] {
                readers.entry(*slot).or_default().push(u);
                load_fmts.entry(*slot).or_default().push(*fmt);
            }
            for (slot, _, _, _) in &stores[u] {
                producers.entry(*slot).or_default().push(u);
            }
        }
        for s in 0..n {
            // The cone closes over its readers: a value written inside it must
            // not be read outside it.
            let mut e = s;
            loop {
                let mut next = e;
                for u in s..=e {
                    for (slot, _, _, _) in &stores[u] {
                        if let Some(rs) = readers.get(slot) {
                            for &r in rs {
                                next = next.max(r);
                            }
                        }
                    }
                }
                if next == e || next >= n {
                    break;
                }
                e = next;
            }
            let mut end = e;
            while end > s {
                if counts[s..=end].iter().any(|c| *c != counts[s]) {
                    end -= 1;
                    continue;
                }
                // A gap may hold only lane constants (inside the merged body)
                // and the flags' enable (hoisted out in front of it).
                let mut gap_words = 0u32;
                // A part whose gap has no `SFPENCC` would run under the previous
                // part's lane flags, which is not what it saw as its own body:
                // interleaving rows makes every part start from a dirty mask, so
                // every part gets a reset (inside the merged body).
                let mut needs_reset: Vec<usize> = Vec::new();
                let mut ok = true;
                for u in s + 1..=end {
                    let mut reset = false;
                    for &g in &ops[units[u - 1].1..units[u].0] {
                        match self.at(g) {
                            Op::TT(TTOp::SfpConst(_)) => gap_words += tt_row_words(self.at(g)),
                            Op::TT(TTOp::Sfp(k)) if k.op == defs::SFPENCC.opcode() => {
                                gap_words += 1;
                                reset = true;
                            }
                            _ => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if !ok {
                        break;
                    }
                    if !reset {
                        needs_reset.push(u);
                        gap_words += 1;
                    }
                }
                if !ok {
                    end -= 1;
                    continue;
                }
                // Virtualize. Every register a merged unit touches becomes a
                // temporary name (`>= 16`) in a space wider than the ISA's
                // 8-bit register field, so units cannot collide no matter how
                // many values the whole cone defines; the pack below turns the
                // temporaries into emitted names by liveness. Nothing here
                // reasons about the eight-register file: `tt_lreg_alloc`
                // assigns the physical ones after this pass. An interior value
                // crosses on one `SFPMOV`, per row, between temporaries. A body
                // is straight-line, so the merged body is one block and
                // liveness there is exact.
                let mut maps: Vec<Vec<u16>> = vec![vec![u16::MAX; 256]; n];
                let mut live: Vec<(OpId, u16, usize)> = Vec::new(); // slot, name, last reader
                // Every row of the merged body starts with the flags reset: a
                // body's own preamble did exactly that once, and interleaving
                // rows leaves the previous part's lane flags behind.
                let mut total = gap_words + 1;
                let mut carry_over: Vec<(OpId, u16, u16)> = Vec::new();
                let mut hand: Vec<(OpId, u16, u16)> = Vec::new();
                let mut next = 16u16;
                let mut feasible = true;
                // The body as the allocator will see it, in temporary space: it
                // is what decides whether this cone fits the eight registers.
                let mut shadow: Vec<(Set<u16>, Set<u16>)> = Vec::new();
                for u in s..=end {
                    live.retain(|(_, _, last)| *last >= u);
                    total += words[u];
                    // Fresh names for every register this unit touches,
                    // including the gap's constants, which are now part of it.
                    let mut used = writes[u].clone();
                    if u > s {
                        used.extend(gap_writes[u].iter().copied());
                    }
                    for (_, reg, _, _) in &loads[u] {
                        used.insert(u16::from(*reg));
                    }
                    for &op in &ops[units[u].0 + 1..units[u].1] {
                        used.extend(tt_row_regs(self.at(op)).0);
                    }
                    for r in used {
                        // `8..15` are the hardware's read-only constants
                        // (`LREG_ZERO` = 9, `LREG_ONE` = 10, ...): they appear in
                        // a body's read set and must never be renamed, which is
                        // why they are not part of the allocatable file. A
                        // body an earlier merge already virtualized is entitled
                        // to a fresh name too: its old one is unique, but the
                        // pack gives both bodies new, disjoint names anyway.
                        if (8..16).contains(&r) {
                            continue;
                        }
                        maps[u][usize::from(r)] = next;
                        next += 1;
                    }
                    // Hand each interior value to this unit, in a register.
                    for (slot, reg, _, op) in &loads[u] {
                        let Some(&(_, src, _)) = live.iter().find(|(sl, _, _)| sl == slot) else {
                            continue;
                        };
                        hand.push((*op, src, maps[u][usize::from(*reg)]));
                    }
                    // Retire stores whose value never leaves the cone.
                    for (slot, reg, fmt, op) in &stores[u] {
                        let fmt_ok = *fmt == INT32 || load_fmts.get(slot).is_some_and(|fs| fs.iter().all(|f| f == fmt));
                        let interior = producers.get(slot).is_some_and(|ps| ps.len() == 1 && ps[0] == u)
                            && readers.get(slot).is_some_and(|rs| rs.iter().all(|&r| r > u && r <= end))
                            && fmt_ok;
                        if !interior {
                            continue;
                        }
                        let Some(last) = readers.get(slot).and_then(|rs| rs.iter().max()) else {
                            continue;
                        };
                        // The store becomes the copy that hands the value on, at
                        // the store's own position -- not "whatever is in the
                        // register at the end of the body", which a later write
                        // in the same unit would break.
                        let carry = next;
                        next += 1;
                        carry_over.push((*op, maps[u][usize::from(*reg)], carry));
                        live.push((*slot, carry, *last));
                    }
                }
                // The body as the allocator will see it, in temporary space: it
                // is what decides whether this cone fits the eight registers.
                if feasible {
                    let ren = |set: &Set<u16>, map: &[u16]| -> Set<u16> {
                        set.iter()
                            .map(|&r| {
                                let to = map[usize::from(r)];
                                if to == u16::MAX { r } else { to }
                            })
                            .collect()
                    };
                    for u in s..=end {
                        if u > s {
                            for &g in &ops[units[u - 1].1..units[u].0] {
                                if let Op::TT(TTOp::SfpConst(c)) = self.at(g) {
                                    let to = maps[u][usize::from(c.reg)];
                                    let to = if to == u16::MAX { u16::from(c.reg) } else { to };
                                    shadow.push((Set::default(), Set::from_iter([to])));
                                }
                            }
                        }
                        for &op in &ops[units[u].0 + 1..units[u].1] {
                            if let Some(&(_, src, dst)) = hand.iter().find(|(o, _, _)| *o == op) {
                                shadow.push((Set::from_iter([src]), Set::from_iter([dst])));
                                continue;
                            }
                            if let Some(&(_, src, dst)) = carry_over.iter().find(|(o, _, _)| *o == op) {
                                shadow.push((Set::from_iter([src]), Set::from_iter([dst])));
                                continue;
                            }
                            let (reads, writes) = tt_row_regs(self.at(op));
                            shadow.push((ren(&reads, &maps[u]), ren(&writes, &maps[u])));
                        }
                    }
                }
                if feasible && tt_alloc_regs(&shadow).is_none() {
                    feasible = false;
                }
                if !feasible || total == 0 || total >= 1 << tt_isa::mailbox::loops::LEN_BITS {
                    end -= 1;
                    continue;
                }
                // One more block repeat than the runner's header holds is a
                // program it cannot run: shrink instead.
                let mut projected = lens.iter().filter(|l| **l > buffer).count();
                for u in s..=end {
                    if lens[u] > buffer {
                        projected -= 1;
                    }
                }
                projected += usize::from(total > buffer);
                if projected > tt_isa::mailbox::loops::MAX {
                    end -= 1;
                    continue;
                }
                // The emitted body's names. The temporary space above is as
                // wide as the cone needs; the ISA's register field is not, so
                // pack by liveness: a name is reused once its interval ends,
                // and the body needs as many names as it has values live at
                // once, not as many as its units define in total. This is what
                // stops a long cone from being cut by name count.
                let (pack, packed) = tt_pack_reg_names(&shadow, next);
                if packed > 256 {
                    end -= 1;
                    continue;
                }
                // Apply.
                let mov = defs::SFPMOV.opcode();
                let marker = ops[units[s].0];
                self.insert_before(
                    ops[units[s].0 + 1],
                    Op::TT(TTOp::Sfp(TTSfpInsn { op: defs::SFPENCC.opcode(), args: [3, 0, 10, 0, 0] })),
                );
                for u in needs_reset {
                    // Inside the body (before its first op, not before the
                    // marker): the word is counted in `len`, and a reset parked
                    // outside would leave the body short of its own length.
                    self.insert_before(
                        ops[units[u].0 + 1],
                        Op::TT(TTOp::Sfp(TTSfpInsn { op: defs::SFPENCC.opcode(), args: [3, 0, 10, 0, 0] })),
                    );
                }
                // One composed map per unit: old register -> temporary -> final.
                // `tt_row_remap` looks each operand up once, so no rename can
                // chain into another even though both spaces are `>= 16`.
                let mut finals: Vec<Vec<u16>> = vec![vec![u16::MAX; 256]; n];
                for u in s..=end {
                    for r in 0..256usize {
                        let temp = maps[u][r];
                        if temp != u16::MAX {
                            finals[u][r] = pack[usize::from(temp)];
                        }
                    }
                    // The unit's own body plus the gap of constants that now
                    // precedes it (the marker between them is not a row op).
                    let mut range: Vec<OpId> = ops[units[u].0 + 1..units[u].1].to_vec();
                    if u > s {
                        range.extend_from_slice(&ops[units[u - 1].1..units[u].0]);
                    }
                    for &op in &range {
                        tt_row_remap(&mut self.ops[op].op, &finals[u]);
                    }
                }
                for (load, from, to) in hand {
                    let (from, to) = (pack[usize::from(from)], pack[usize::from(to)]);
                    self.ops[load].op = Op::TT(TTOp::Sfp(TTSfpInsn { op: mov, args: [from, to, 0, 0, 0] }));
                }
                for (store, from, to) in carry_over {
                    let (from, to) = (pack[usize::from(from)], pack[usize::from(to)]);
                    self.ops[store].op = Op::TT(TTOp::Sfp(TTSfpInsn { op: mov, args: [from, to, 0, 0, 0] }));
                }
                self.ops[marker].op = Op::TT(TTOp::Replay { len: total as u16, count: counts[s] });
                for u in s + 1..=end {
                    self.remove_op(ops[units[u].0]);
                }
                return true;
            }
        }
        false
    }

    /// Assign the eight physical lane registers to the virtuals
    /// `tt_sfpu_fusion` left behind: `tt_lreg_alloc`.
    ///
    /// The space is `0..7` general purpose, `8..15` the hardware's read-only
    /// constants (`LREG_ZERO` = 9, `LREG_ONE` = 10, ...; never remapped), and
    /// `>= 16` virtual. A merged row body is one straight-line block, so its
    /// liveness is exact: every virtual has one interval `def -> last use`, and
    /// the linear scan is not an approximation. A register frees when its
    /// interval ends and is reused, so a body with twenty values still fits
    /// eight if no more than eight are live at once.
    ///
    /// A `SFPMOV` whose two ends land on the same register is dropped (the
    /// coalescing case) and the `Replay` length is recomputed from what is
    /// left. A body that needs a ninth live value fails loudly: spilling to a
    /// DST tile is the next step, never a guess.
    pub fn tt_lreg_alloc(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        let mut ops: Vec<OpId> = Vec::new();
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() {
                break;
            }
            ops.push(id);
            id = self.next_op(id);
        }
        let mut i = 0;
        while i < ops.len() {
            let Op::TT(TTOp::Replay { len, count }) = self.at(ops[i]) else {
                i += 1;
                continue;
            };
            let (marker, count) = (ops[i], *count);
            let mut words = 0u32;
            let mut j = i + 1;
            while j < ops.len() && words < u32::from(*len) {
                words += tt_row_words(self.at(ops[j]));
                j += 1;
            }
            let body: Vec<OpId> = ops[i + 1..j].to_vec();
            i = j;
            let regs: Vec<(Set<u16>, Set<u16>)> = body.iter().map(|&op| tt_row_regs(self.at(op))).collect();
            // The same scan the fusion used to prove the cone fits, now on the
            // emitted body: here `None` is a bug, not a reason to shrink.
            let Some(assign) = tt_alloc_regs(&regs) else {
                todo!("tt_lreg_alloc: no free lane register in the body of {} ops; spilling is not implemented", body.len());
            };
            if assign.is_empty() {
                continue;
            }
            let mut map = vec![u16::MAX; 256];
            for (&r, &p) in &assign {
                map[usize::from(r)] = u16::from(p);
            }
            // Physical `0..8` are never sources here, and every operand is
            // looked up once, so no rename can chain.
            for &op in &body {
                tt_row_remap(&mut self.ops[op].op, &map);
            }
            let mut gone: Vec<OpId> = Vec::new();
            for &op in &body {
                if let Op::TT(TTOp::Sfp(insn)) = self.at(op) {
                    // `SFPMOV` with `Mod1` clear and the same source and
                    // destination is a no-op after coalescing. With any `Mod1`
                    // bit set it is not: `Mod1 = 1` is the *negate*, and
                    // dropping that is how `cast(i32)` returned `|e|` (log2 was
                    // off by exactly `-2e` for negative exponents).
                    if insn.op == tt_isa::isa::generated::defs::SFPMOV.opcode()
                        && insn.args[0] == insn.args[1]
                        && insn.args[2] == 0
                    {
                        gone.push(op);
                    }
                }
            }
            for &op in &gone {
                self.remove_op(op);
            }
            let mut words = 0u32;
            for &op in &body {
                if !gone.contains(&op) {
                    words += tt_row_words(self.at(op));
                }
            }
            self.ops[marker].op = Op::TT(TTOp::Replay { len: words as u16, count });
        }
    }

    /// Reclaim a materialised value's L1 round trip: `tt_dst_reuse`'s first
    /// half.
    ///
    /// Lowering gives every value with more than one use a CB (`tt_materialize`:
    /// pack the DST row, unpack it back for each consumer). That is only
    /// necessary when the producer's row is reused before the last consumer.
    /// When nothing claims the row in between, the value can stay in it and the
    /// pair disappears — and with it the pack on T2, the unpack on T0 and both
    /// seams `tt_reg_sync` would derive.
    ///
    /// Exact conditions: the CB is touched by exactly one pack and one unpack
    /// (no publish, no drain, so it is not a kernel output); the unpack's
    /// destination is written by nothing else; the producer's row is not
    /// written between the pack and the last consumer; and the rows still fit
    /// the tile budget afterwards, which is what makes this safe to do before
    /// allocation — a reclamation that would need a ninth tile is left alone
    /// rather than turned into a compile error. Repeats until nothing applies.
    fn tt_reclaim_round_trips(&mut self) {
        let mut tried: Vec<OpId> = Vec::new();
        for _ in 0..250_000 {
            let mut ops: Vec<OpId> = Vec::new();
            let mut id = self.head;
            for _ in 0..250_000 {
                if id.is_null() {
                    break;
                }
                ops.push(id);
                id = self.next_op(id);
            }
            let Some((pack, unpack, cb, x, y)) = self.tt_round_trip(&ops, &tried) else {
                return;
            };
            let Some(pack_idx) = ops.iter().position(|&o| o == pack) else {
                return;
            };
            let mut keep = true;
            // The unpack's row is written by the unpack alone: the consumers
            // are reads, so moving them onto `x` is a pure rename.
            let writers = ops.iter().filter(|&&op| self.tt_register_output(self.at(op)) == Some(y)).count();
            if writers != 1 {
                keep = false;
            }
            // Nothing but the consumers may write the producer's row in the
            // window the reclamation extends it over.
            if keep {
                for &op in &ops[pack_idx + 1..] {
                    if op == unpack || op == pack {
                        continue;
                    }
                    if self.tt_register_output(self.at(op)) == Some(x) {
                        keep = false;
                        break;
                    }
                }
            }
            // The rows must still fit afterwards; a reclamation that needs one
            // tile too many stays a round trip. The register file is uniform
            // (`tt_verify` enforces it), so one width sets the budget.
            let budget = ops
                .iter()
                .find_map(|&op| match self.at(op) {
                    Op::Storage { dtype, scope: MemScope::Register, .. } => Some(if dtype.bit_size() > 16 { 8 } else { 16 }),
                    _ => None,
                })
                .unwrap_or(8);
            if keep && self.tt_live_peak(&ops, Some((x, y))) > budget {
                keep = false;
            }
            if !keep {
                tried.push(cb);
                continue;
            }
            // Move every use of the unpacked row onto the producer's row, then
            // drop the pair (and the CB, if nothing else names it).
            let mut rewrites: Vec<(OpId, usize)> = Vec::new();
            for &op in &ops {
                if op == pack || op == unpack {
                    continue;
                }
                for (idx, p) in self.at(op).parameters().enumerate() {
                    if p == y {
                        rewrites.push((op, idx));
                    }
                }
            }
            for (op, idx) in rewrites {
                if let Some(slot) = self.ops[op].op.parameters_mut().nth(idx) {
                    *slot = x;
                }
            }
            self.remove_op(unpack);
            self.remove_op(pack);
            let still_used = {
                let mut id = self.head;
                let mut used = false;
                for _ in 0..250_000 {
                    if id.is_null() {
                        break;
                    }
                    if self.at(id).parameters().any(|p| p == cb) {
                        used = true;
                        break;
                    }
                    id = self.next_op(id);
                }
                used
            };
            if !still_used {
                self.remove_op(cb);
            }
            // The unpacked row is dead now; dropping it keeps the tile budget
            // honest before the allocator counts.
            self.remove_op(y);
        }
    }

    /// One reclaimable round trip in `ops`: `(pack, unpack, cb, producer row,
    /// unpack row)`. A CB already in `tried` is skipped.
    fn tt_round_trip(&self, ops: &[OpId], tried: &[OpId]) -> Option<(OpId, OpId, OpId, OpId, OpId)> {
        let mut packs: Map<OpId, Vec<OpId>> = Map::default();
        let mut unpacks: Map<OpId, Vec<OpId>> = Map::default();
        let mut touched: Vec<OpId> = Vec::new();
        for &op in ops {
            let Op::TT(TTOp::Copy { .. }) = self.at(op) else { continue };
            match self.tt_copy_kind(op) {
                TtCopy::Pack => {
                    let Op::TT(TTOp::Copy { dst, .. }) = self.at(op) else {
                        unreachable!()
                    };
                    packs.entry(*dst).or_default().push(op);
                }
                TtCopy::Unpack => {
                    let Op::TT(TTOp::Copy { src, .. }) = self.at(op) else {
                        unreachable!()
                    };
                    unpacks.entry(*src).or_default().push(op);
                }
                // A publish or drain makes the CB a DRAM boundary: not a
                // round trip.
                TtCopy::Publish | TtCopy::Drain => touched.push(op),
            }
        }
        let mut found: Option<(OpId, OpId, OpId, OpId, OpId)> = None;
        for (&cb, ps) in packs.iter() {
            if ps.len() != 1 || tried.contains(&cb) || touched.iter().any(|&op| self.at(op).parameters().any(|p| p == cb)) {
                continue;
            }
            let Some(us) = unpacks.get(&cb) else { continue };
            if us.len() != 1 {
                continue;
            }
            let (pack, unpack) = (ps[0], us[0]);
            let (Op::TT(TTOp::Copy { src: x, .. }), Op::TT(TTOp::Copy { dst: y, .. })) = (self.at(pack), self.at(unpack)) else {
                continue;
            };
            found = Some((pack, unpack, cb, *x, *y));
            break;
        }
        found
    }

    /// Peak number of live `Register` tiles over `ops`, with `merge`'s pair
    /// treated as one value in `x` that lives to `y`'s last use. That is what
    /// the allocator will need, since a row frees once its last use is past.
    fn tt_live_peak(&self, ops: &[OpId], merge: Option<(OpId, OpId)>) -> usize {
        let mut first: Map<OpId, usize> = Map::default();
        let mut last: Map<OpId, usize> = Map::default();
        for (i, &op) in ops.iter().enumerate() {
            for p in self.at(op).parameters() {
                if p.is_null() || !matches!(self.ops[p].op, Op::Storage { scope: MemScope::Register, .. }) {
                    continue;
                }
                first.entry(p).or_insert(i);
                last.insert(p, i);
            }
        }
        if let Some((x, y)) = merge {
            if let Some(&end) = last.get(&y) {
                last.insert(x, end);
            }
            first.remove(&y);
            last.remove(&y);
        }
        let mut delta = vec![0i32; ops.len() + 2];
        for (s, &f) in first.iter() {
            let l = last[s];
            delta[f] += 1;
            delta[l + 1] -= 1;
        }
        let mut run = 0i32;
        let mut peak = 0i32;
        for d in delta {
            run += d;
            peak = peak.max(run);
        }
        peak as usize
    }

    /// Role hand-off over the DST tiles: `tt_reg_sync`. This is the only pass
    /// that emits `Semset`/`Wait`/`Post` for tile registers.
    ///
    /// The hardware has no DST lock: the Sync Unit holds eight flat 4-bit
    /// counters, and the only ordering between the unpacker (T0), the math
    /// thread (T1: Matrix Unit + SFPU) and the packer (T2) is one
    /// `SEMPOST`/`SEMWAIT` per direction. So the pass derives, per physical
    /// tile, who writes it and who reads it, in program order, and emits the
    /// seams that are actually needed:
    ///
    /// - a *data* seam when a reader in another role follows a writer: the
    ///   writer posts after its last write, the reader takes before its first
    ///   read (`UNPACKED` for T0's writes, `COMPUTED` for T1's);
    /// - a *reuse* seam when a later epoch overwrites a tile some other role
    ///   read: the reader posts after its last read, the next writer takes
    ///   before its first write (`FREE`).
    ///
    /// Both are counting hand-offs, so one semaphore serves one ordered
    /// `(producer, consumer)` pair — every tile and epoch on that edge shares
    /// it. Each role's stream is in order, so posts and takes pair up FIFO.
    /// It cannot be one semaphore per *direction*: `SEMWAIT` is a level test,
    /// so two consumers on one counter both clear on a single post and the
    /// second `SEMGET` finds it empty. Reads by the role that wrote the tile
    /// need no seam — they are the same stream — and a tile written once
    /// (every single-tile kernel) gets no reuse seam at all. Initial values
    /// are zero: a producer always posts before any take of its edge.
    ///
    /// A missing seam is a real race only because a kernel with any `Semset`
    /// runs the three roles concurrently (`Schedule::Concurrent`); without
    /// semaphores the runtime runs T0, then T1, then T2.
    pub fn tt_reg_sync(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        let mut ops: Vec<OpId> = Vec::new();
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() {
                break;
            }
            ops.push(id);
            id = self.next_op(id);
        }
        // Per tile, in program order: `(op index, role, is write)`. The
        // intra-tile address is not part of an edge: a hand-off is per tile,
        // and the consumer's first row read follows the producer's last write
        // anyway.
        let mut touches: Map<OpId, Vec<(usize, u8, bool)>> = Map::default();
        let mut acc: Option<OpId> = None;
        for (i, &op) in ops.iter().enumerate() {
            let mut touched: Vec<(OpId, u8, bool)> = Vec::new();
            match self.ops[op].op {
                Op::TT(TTOp::Setrwc { dst }) => {
                    acc = Some(dst);
                    touched.push((dst, 1, true));
                }
                // The Matrix Unit's result lands in the accumulator the last
                // `SETRWC` named; `MVMUL` carries a row of it, not a storage.
                Op::TT(TTOp::Mvmul { .. }) => {
                    if let Some(a) = acc {
                        touched.push((a, 1, true));
                    }
                }
                // The SFPU is T1 too: its stores and loads join the same
                // stream as the Matrix Unit, so a tile it writes and reads
                // needs no seam.
                Op::TT(TTOp::SfpStore(ref store)) => touched.push((store.slot, 1, true)),
                Op::TT(TTOp::SfpLoad(ref load)) => touched.push((load.slot, 1, false)),
                Op::TT(TTOp::Sfpu { a, b, slot, .. }) => {
                    touched.push((a, 1, false));
                    if !b.is_null() {
                        touched.push((b, 1, false));
                    }
                    touched.push((slot, 1, true));
                }
                Op::TT(TTOp::Copy { src, dst, .. }) => match self.tt_copy_kind(op) {
                    // A `Dst` tile leaving for L1 and coming back is a
                    // hand-off between the packer and the unpacker, and it is
                    // *not* covered by the tile's own edge: the pack reads
                    // one storage and the unpack writes another, with the CB
                    // in between. Model the CB side too, or the unpack races
                    // the pack and reads a stale tile (this was the
                    // `sigmoid`/`exp2` constant-output bug).
                    TtCopy::Unpack => {
                        touched.push((dst, 0, true));
                        touched.push((src, 0, false));
                    }
                    TtCopy::Pack => {
                        touched.push((src, 2, false));
                        touched.push((dst, 2, true));
                    }
                    TtCopy::Publish | TtCopy::Drain => {}
                },
                _ => {}
            }
            for (tile, role, write) in touched {
                if matches!(self.ops[tile].op, Op::Storage { scope: MemScope::Register | MemScope::Circular, .. }) {
                    touches.entry(tile).or_default().push((i, role, write));
                }
            }
        }
        // A `Replay` body runs as one unit: a hand-off landing inside it would
        // be recorded and replayed with the body, so `Wait`s go before the
        // marker and `Post`s after the body's last op. Map a member op index
        // to the body's bounds.
        let mut replay_of: Map<usize, (usize, usize)> = Map::default();
        {
            let mut i = 0;
            while i < ops.len() {
                if let Op::TT(TTOp::Replay { len, .. }) = self.at(ops[i]) {
                    let mut words = 0u32;
                    let mut j = i + 1;
                    while j < ops.len() && words < u32::from(*len) {
                        words += tt_row_words(&self.at(ops[j]));
                        j += 1;
                    }
                    for k in i + 1..j {
                        replay_of.insert(k, (i, j));
                    }
                    i = j;
                } else {
                    i += 1;
                }
            }
        }
        let before_replay = |i: usize| replay_of.get(&i).map(|(start, _)| *start).unwrap_or(i);
        let after_replay = |i: usize| replay_of.get(&i).map(|(_, end)| end - 1).unwrap_or(i);
        let before = |role: u8| match role {
            0 => TT_BEFORE_UNPACKER,
            2 => TT_BEFORE_PACKER,
            _ => TT_BEFORE_ALL,
        };
        // A seam names the ordered `(producer, consumer)` edge it carries; the
        // semaphore index is assigned once all edges are known.
        let mut posts: Vec<(usize, (u8, u8))> = Vec::new();
        let mut takes: Vec<(usize, (u8, u8))> = Vec::new();
        for events in touches.values() {
            // One epoch per write: the data an epoch publishes is what the
            // reads that follow it (before the next write) consume.
            let mut epochs: Vec<((u8, usize), Vec<(u8, usize, usize)>)> = Vec::new();
            for &(idx, role, write) in events.iter() {
                if write {
                    epochs.push(((role, idx), Vec::new()));
                } else if let Some(((writer, _), readers)) = epochs.last_mut() {
                    if *writer != role {
                        match readers.iter_mut().find(|r| r.0 == role) {
                            Some(r) => r.2 = idx,
                            None => readers.push((role, idx, idx)),
                        }
                    }
                }
            }
            for (n, ((writer, write_idx), readers)) in epochs.iter().enumerate() {
                for &(reader, first, _) in readers {
                    posts.push((after_replay(*write_idx), (*writer, reader)));
                    takes.push((before_replay(first), (*writer, reader)));
                }
                // The reuse edge: a tile written again in the next epoch must
                // not be overwritten while a reader of this epoch still needs
                // it. Every reader that is not the next writer posts after its
                // last access, and the next writer takes on that reader's edge
                // before its first write. A writer that was also this epoch's
                // reader needs no edge: one thread's program order already
                // orders it.
                if let Some(((next_writer, next_write), _)) = epochs.get(n + 1) {
                    for &(reader, _, last) in readers {
                        if reader != *next_writer {
                            posts.push((after_replay(last), (reader, *next_writer)));
                            takes.push((before_replay(*next_write), (reader, *next_writer)));
                        }
                    }
                }
            }
        }
        if posts.is_empty() && takes.is_empty() {
            return;
        }
        let mut edges: Vec<(u8, u8)> = Vec::new();
        for &(_, edge) in posts.iter().chain(takes.iter()) {
            if !edges.contains(&edge) {
                edges.push(edge);
            }
        }
        edges.sort_unstable();
        assert!(
            edges.len() <= tt_isa::sync::Semaphore::COUNT as usize,
            "tt_reg_sync: {} role hand-off edges need more than the {} semaphores of the Sync Unit",
            edges.len(),
            tt_isa::sync::Semaphore::COUNT
        );
        let sem =
            |edge: (u8, u8)| -> u8 { edges.iter().position(|e| *e == edge).expect("every seam's edge is in `edges`") as u8 };
        // `Semset` is not an instruction: codegen folds it into the run's
        // semaphore table, and `tt_verify` requires it in the compute section.
        let used: Vec<u8> = (0..edges.len() as u8).collect();
        let mut scan = self.head;
        let mut end_reader = None;
        for _ in 0..250_000 {
            if scan.is_null() {
                break;
            }
            if matches!(self.ops[scan].op, Op::TT(TTOp::EndReader)) {
                end_reader = Some(scan);
                break;
            }
            scan = self.next_op(scan);
        }
        let end_reader = end_reader.expect("tt_reg_sync: no EndReader in a lowered TT kernel");
        for &sem in &used {
            self.insert_after(end_reader, Op::TT(TTOp::Semset { sem, value: 0 }));
        }
        for (idx, edge) in takes {
            let (_, consumer) = edge;
            self.insert_before(ops[idx], Op::TT(TTOp::Wait { sem: sem(edge), before: before(consumer), role: consumer }));
        }
        for (idx, edge) in posts {
            let (producer, _) = edge;
            // Back-pressure: the counter is four bits, so the producer must
            // not get more than fifteen ahead of the consumer.
            let post = self.insert_after(ops[idx], Op::TT(TTOp::Post { sem: sem(edge), role: producer }));
            self.insert_before(post, Op::TT(TTOp::WaitBelowMax { sem: sem(edge), before: before(producer), role: producer }));
        }
    }

    /// The Register operand an op writes, if any. Inputs precede the
    /// output in `parameters`, so the reuse pass can release dead inputs
    /// before allocating this row.
    fn tt_register_output(&self, op: &Op) -> Option<OpId> {
        match op {
            Op::TT(TTOp::Sfpu { slot, .. }) => Some(*slot),
            // A row-level store is where a ported SFPU sequence writes its
            // result tile, so it is that tile's output as far as the
            // allocator is concerned.
            Op::TT(TTOp::SfpStore(store)) => Some(store.slot),
            Op::TT(TTOp::Copy { dst, .. }) => match self.ops[*dst].op {
                Op::Storage { scope: MemScope::Register, .. } => Some(*dst),
                _ => None,
            },
            _ => None,
        }
    }

    /// Ensure `logical` has a physical row, reusing a free row of the same
    /// dtype when possible.
    fn tt_ensure_phys(
        &mut self,
        physical: &mut Map<OpId, OpId>,
        active: &mut Vec<(OpId, OpId, usize)>,
        free: &mut Vec<(DType, OpId)>,
        last_use: &Map<OpId, usize>,
        logical: OpId,
        prologue: OpId,
    ) {
        if physical.contains_key(&logical) {
            return;
        }
        let Op::Storage { dtype, .. } = self.ops[logical].op else {
            return;
        };
        let phys = if let Some(i) = free.iter().position(|(d, _)| *d == dtype) {
            free.remove(i).1
        } else {
            self.tt_new_dst(dtype, prologue)
        };
        physical.insert(logical, phys);
        active.push((logical, phys, last_use.get(&logical).copied().unwrap_or(0)));
    }

    /// Base index with lanes stripped: clone the index expression
    /// with every `Unroll` replaced by 0. No affinity assumption —
    /// substitution works on any expression shape; uncloneable leaves
    /// (traffic, control) bail the caller. Shared leaves are reused
    /// (they dominate `before` exactly like the cloned index math
    /// does); the rest is rebuilt before `before`.
    ///
    /// Group ids rebase to physical tiles: DRAM traffic addresses
    /// tile-layout buffers, so a const-scaled group id takes the
    /// physical tile stride (one full [32, 32] tile per step, times
    /// inner-group extents) and a bare group id inside Add/Sub wraps
    /// to id * stride. Symbolic extents bail the caller.
    fn strip_unrolls(&mut self, idx: OpId, before: OpId) -> Option<OpId> {
        // Clone memo per call (index math is small).
        let mut memo: Map<OpId, OpId> = Map::default();
        // Group axis -> physical tile stride (element units), built
        // lazily on first group use inside the walk below.
        let mut strides: Option<Map<u32, Dim>> = None;
        let zero = self.insert_before(before, Op::Const(Constant::idx(0i64)));
        memo.insert(zero, zero);
        let mut stack = vec![(idx, false)];
        for _ in 0..250_000 {
            let Some((n, expanded)) = stack.pop() else { break };
            if memo.contains_key(&n) {
                continue;
            }
            match self.ops[n].op {
                Op::Const(_) | Op::Range { .. } | Op::Param { .. } => {
                    memo.insert(n, n);
                }
                Op::Unroll { .. } => {
                    memo.insert(n, zero);
                }
                Op::Binary { x, y, bop } => {
                    if !expanded {
                        stack.push((n, true));
                        stack.push((x, false));
                        stack.push((y, false));
                    } else {
                        let nx = memo[&x];
                        let ny = memo[&y];
                        // Group-leaf axes of the memo images (owned
                        // copies; every borrow ends before inserts).
                        let gx: Option<u32> = match &self.ops[nx].op {
                            Op::Range { axis, kind: RangeKind::Group(_) } => Some(*axis),
                            _ => None,
                        };
                        let gy: Option<u32> = match &self.ops[ny].op {
                            Op::Range { axis, kind: RangeKind::Group(_) } => Some(*axis),
                            _ => None,
                        };
                        // Axis -> physical tile stride, built on first
                        // group use. One group step is one full
                        // [32, 32] tile (1024 elems, the unit phase 1
                        // sizes CBs, slots and byte counts in) times
                        // the extents of all inner (higher-axis) groups;
                        // split keeps the outer on the original axis.
                        // Unresolvable (symbolic) extents bail the pass:
                        // never silent wrong code.
                        if (gx.is_some() || gy.is_some()) && strides.is_none() {
                            let mut axes: Vec<(u32, Dim)> = Vec::new();
                            for (_, op) in self.iter_unordered() {
                                if let Op::Range { axis, kind: RangeKind::Group(len) } = op {
                                    let Some(e) = self.resolve_const(*len).and_then(Constant::as_dim) else {
                                        return None;
                                    };
                                    axes.push((*axis, e));
                                }
                            }
                            axes.sort();
                            let mut map: Map<u32, Dim> = Map::default();
                            let mut suffix: Dim = 1024;
                            for (a, e) in axes.iter().rev() {
                                map.insert(*a, suffix);
                                suffix = match suffix.checked_mul(*e) {
                                    Some(s) => s,
                                    None => return None,
                                };
                            }
                            strides = Some(map);
                        }
                        // Physical group bases: a const-scaled group id
                        // takes the tile stride; a bare group id inside
                        // Add/Sub wraps to id * stride. These are the
                        // forms split/pad composition emits; anything
                        // else keeps the faithful clone and trips the
                        // downstream checks loudly.
                        let mut ax = nx;
                        let mut ay = ny;
                        match bop {
                            BOp::Mul => {
                                if let Some(a) = gx {
                                    if gy.is_none() && matches!(self.ops[ny].op, Op::Const(_)) {
                                        let s = strides.as_ref().expect("strip_unrolls: stride map missing")[&a];
                                        let sc = self.insert_before(before, Op::Const(Constant::idx(s)));
                                        ax = self.insert_before(before, Op::Binary { x: nx, y: sc, bop });
                                    }
                                }
                                if let Some(a) = gy {
                                    if gx.is_none() && matches!(self.ops[nx].op, Op::Const(_)) {
                                        let s = strides.as_ref().expect("strip_unrolls: stride map missing")[&a];
                                        let sc = self.insert_before(before, Op::Const(Constant::idx(s)));
                                        ay = self.insert_before(before, Op::Binary { x: sc, y: ny, bop });
                                    }
                                }
                            }
                            BOp::Add | BOp::Sub => {
                                if let Some(a) = gx {
                                    let s = strides.as_ref().expect("strip_unrolls: stride map missing")[&a];
                                    let sc = self.insert_before(before, Op::Const(Constant::idx(s)));
                                    ax = self.insert_before(before, Op::Binary { x: nx, y: sc, bop: BOp::Mul });
                                }
                                if let Some(a) = gy {
                                    let s = strides.as_ref().expect("strip_unrolls: stride map missing")[&a];
                                    let sc = self.insert_before(before, Op::Const(Constant::idx(s)));
                                    ay = self.insert_before(before, Op::Binary { x: ny, y: sc, bop: BOp::Mul });
                                }
                            }
                            _ => {}
                        }
                        // Fold constant lanes away: the base is fully
                        // static exactly when the index is.
                        let id = match (&self.ops[ax].op, &self.ops[ay].op) {
                            (Op::Const(cx), Op::Const(cy)) => {
                                self.insert_before(before, Op::Const(Constant::binary(*cx, *cy, bop)))
                            }
                            _ => self.insert_before(before, Op::Binary { x: ax, y: ay, bop }),
                        };
                        memo.insert(n, id);
                    }
                }
                Op::Unary { x, uop } => {
                    if !expanded {
                        stack.push((n, true));
                        stack.push((x, false));
                    } else {
                        let nx = memo[&x];
                        let id = match &self.ops[nx].op {
                            Op::Const(cx) => self.insert_before(before, Op::Const(cx.unary(uop))),
                            _ => self.insert_before(before, Op::Unary { x: nx, uop }),
                        };
                        memo.insert(n, id);
                    }
                }
                Op::Cast { x, dtype } => {
                    if !expanded {
                        stack.push((n, true));
                        stack.push((x, false));
                    } else {
                        let nx = memo[&x];
                        let id = self.insert_before(before, Op::Cast { x: nx, dtype });
                        memo.insert(n, id);
                    }
                }
                _ => return None,
            }
        }
        memo.get(&idx).copied()
    }

    /// TT stream verifier: the kernel is one reader section, one compute
    /// section, and one writer section, in that order, each closed by its
    /// marker. Compute programs live only in the compute section; DRAM
    /// copies never do. `Op::Barrier` is a host barrier, never a TT
    /// instruction.
    pub fn tt_verify(&self) {
        let mut seen_reader = false;
        let mut seen_compute = false;
        let mut section = 0u8;
        let mut reader_entries = 0u32;
        let mut writer_entries = 0u32;
        let mut written: Set<OpId> = Set::default();
        let mut sems_init: Set<u8> = Set::default();
        let mut sems_used: Vec<(u8, OpId)> = Vec::new();
        // Words still expected in the current `Replay` body.
        let mut replay_left: Option<u32> = None;
        let is_dram = |op: &Op| matches!(op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. });
        let is_written = |op: &Op| matches!(op, Op::Param { kind: ParamKind::GlobalMut, .. });
        let mut id = self.head;
        while !id.is_null() {
            match self.at(id) {
                Op::TT(TTOp::EndReader) => {
                    assert!(section == 0, "tt_verify: duplicate EndReader at {id:?}");
                    assert!(!seen_reader, "tt_verify: duplicate EndReader at {id:?}");
                    seen_reader = true;
                    section = 1;
                }
                Op::TT(TTOp::EndCompute) => {
                    assert!(seen_reader, "tt_verify: EndCompute without EndReader at {id:?}");
                    assert!(!seen_compute, "tt_verify: duplicate EndCompute at {id:?}");
                    assert!(section == 1, "tt_verify: EndCompute outside the compute section at {id:?}");
                    seen_compute = true;
                    section = 2;
                }
                Op::TT(TTOp::Sfpu { .. })
                | Op::TT(TTOp::Unpacr { .. })
                | Op::TT(TTOp::Setrwc { .. })
                | Op::TT(TTOp::Setadczw { .. })
                | Op::TT(TTOp::Setadcx { .. })
                | Op::TT(TTOp::Zeroacc { .. })
                | Op::TT(TTOp::Mvmul { .. })
                | Op::TT(TTOp::Setc16 { .. })
                | Op::TT(TTOp::Setcfg { .. }) => {
                    assert!(section == 1, "tt_verify: compute op outside the compute section at {id:?}");
                }
                Op::TT(TTOp::Post { sem, .. }) | Op::TT(TTOp::Wait { sem, .. }) | Op::TT(TTOp::WaitBelowMax { sem, .. }) => {
                    assert!(section == 1, "tt_verify: semaphore op outside the compute section at {id:?}");
                    assert!(
                        *sem < tt_isa::sync::Semaphore::COUNT,
                        "tt_verify: semaphore op names {sem}, but only {} exist",
                        tt_isa::sync::Semaphore::COUNT
                    );
                    sems_used.push((*sem, id));
                }
                Op::TT(TTOp::Semset { sem, .. }) => {
                    assert!(section == 1, "tt_verify: Semset outside the compute section at {id:?}");
                    assert!(
                        *sem < tt_isa::sync::Semaphore::COUNT,
                        "tt_verify: Semset names semaphore {sem}, but only {} exist",
                        tt_isa::sync::Semaphore::COUNT
                    );
                    assert!(sems_init.insert(*sem), "tt_verify: semaphore {sem} initialised twice");
                }
                Op::TT(TTOp::DmRead(_))
                | Op::TT(TTOp::DmWrite(_))
                | Op::TT(TTOp::DmReadTransposed(_))
                | Op::TT(TTOp::DmReadBroadcastCol(_))
                | Op::TT(TTOp::DmFill(_))
                | Op::TT(TTOp::DmCopyWords(_))
                | Op::TT(TTOp::DmIndexPick(_))
                | Op::TT(TTOp::DmKernel(_))
                | Op::TT(TTOp::DmWait) => {
                    // Every `Dm` op is one entry of its section's mover list:
                    // B runs the reader's, NC the writer's. A list is issued
                    // whole, so the entry count is a compile-time property.
                    match section {
                        0 => reader_entries += 1,
                        2 => writer_entries += 1,
                        _ => panic!("tt_verify: mover op in the compute section at {id:?}"),
                    }
                }
                Op::TT(TTOp::Copy { src, dst, .. }) => {
                    let s = self.tt_storage_of(*src);
                    let d = self.tt_storage_of(*dst);
                    if section == 1 {
                        assert!(!is_dram(self.at(s)), "tt_verify: DRAM read in compute at {id:?}");
                        assert!(!is_dram(self.at(d)), "tt_verify: DRAM write in compute at {id:?}");
                    }
                    if is_written(self.at(d)) {
                        written.insert(d);
                    }
                    assert!(
                        !(is_written(self.at(s)) && written.contains(&s)),
                        "tt_verify: store->load of the same DRAM param at {id:?}"
                    );
                }
                Op::Store { dst, .. } => {
                    let d = self.tt_storage_of(*dst);
                    if section == 1 {
                        assert!(!is_dram(self.at(d)), "tt_verify: DRAM write in compute at {id:?}");
                    }
                    if is_written(self.at(d)) {
                        written.insert(d);
                    }
                }
                Op::Load { src, .. } => {
                    let s = self.tt_storage_of(*src);
                    if section == 1 {
                        assert!(!is_dram(self.at(s)), "tt_verify: DRAM read in compute at {id:?}");
                    }
                    assert!(
                        !(is_written(self.at(s)) && written.contains(&s)),
                        "tt_verify: store->load of the same DRAM param at {id:?}"
                    );
                }
                Op::TT(TTOp::Replay { len, count }) => {
                    assert!(section == 1, "tt_verify: replay outside the compute section at {id:?}");
                    assert!(replay_left.is_none(), "tt_verify: nested replay at {id:?}");
                    // A body that fits the replay buffer is a hardware `REPLAY`;
                    // a longer one is a block repeat the runner pushes
                    // (`mailbox::loops`), so the buffer is not its limit.
                    assert!(
                        *len > 0 && u32::from(*len) < 1 << tt_isa::mailbox::loops::LEN_BITS,
                        "tt_verify: replay body of {len} words at {id:?} exceeds the block repeat's length field"
                    );
                    assert!(*count >= 2, "tt_verify: replay count {count} at {id:?}");
                    replay_left = Some(u32::from(*len));
                }
                Op::Barrier => panic!("tt_verify: Op::Barrier in a TT kernel at {id:?}"),
                _ => {}
            }
            // A replay body is made of row ops only, and codegen counts words
            // exactly as `SfpConst`'s two-immediate form implies.
            if let Some(left) = replay_left {
                if !matches!(self.at(id), Op::TT(TTOp::Replay { .. })) {
                    let words = match self.at(id) {
                        Op::TT(TTOp::SfpConst(c)) => {
                            if c.bits & 0xffff == 0 {
                                1
                            } else {
                                2
                            }
                        }
                        Op::TT(TTOp::SfpLoad(_) | TTOp::SfpStore(_) | TTOp::Sfp(_)) => 1,
                        other => panic!("tt_verify: {other:?} at {id:?} is not a row op in a replay body"),
                    };
                    assert!(words <= left, "tt_verify: replay body overruns at {id:?}");
                    replay_left = (left > words).then_some(left - words);
                }
            }
            id = self.next_op(id);
        }
        assert!(replay_left.is_none(), "tt_verify: a replay body is not complete at the kernel end");
        assert!(seen_reader, "tt_verify: missing EndReader");
        assert!(seen_compute, "tt_verify: missing EndCompute");
        // A wait on an uninitialised semaphore reads whatever the previous
        // kernel left in the Sync Unit: every seam must be in the run's init
        // list.
        for &(sem, id) in &sems_used {
            assert!(sems_init.contains(&sem), "tt_verify: semaphore {sem} used at {id:?} has no Semset");
        }
        // One list per mover per launch, and the list ring holds
        // `dm::LIST_MAX` entries: a kernel that would need a second list is a
        // compile error here, never a silent split at launch.
        assert!(
            reader_entries <= tt_isa::dm::LIST_MAX,
            "tt_verify: {reader_entries} reader entries exceed LIST_MAX {}",
            tt_isa::dm::LIST_MAX
        );
        assert!(
            writer_entries <= tt_isa::dm::LIST_MAX,
            "tt_verify: {writer_entries} writer entries exceed LIST_MAX {}",
            tt_isa::dm::LIST_MAX
        );
        // The DST register file is one mode wide: 32-bit tiles (F32/I32/U32)
        // or 16-bit tiles, never both (WormholeB0/TensixTile/TensixCoprocessor/
        // Dst.md: mixing `Dst16b` and `Dst32b` is possible but is expected to
        // be exclusive). Capacity is 8 tiles of 32-bit or 16 of 16-bit.
        let mut dst_count = 0u32;
        let mut dst_slots: Set<u32> = Set::default();
        let mut dst_wide: Option<bool> = None;
        for (id, op) in self.iter_unordered() {
            let Op::Storage { dtype, scope: MemScope::Register, id: slot, .. } = op else {
                continue;
            };
            dst_count += 1;
            // The allocator stamped one index per physical tile; codegen reads
            // it as the slot, so two tiles sharing an index (or an index past
            // the file) would alias rows silently.
            assert!(dst_slots.insert(*slot), "tt_verify: DST storages share allocation index {slot} at {id:?}");
            assert!(
                dtype.bit_size() == 16 || dtype.bit_size() == 32,
                "tt_verify: DST register {id:?} is {dtype:?}; Dst holds 16-bit or 32-bit tiles only"
            );
            let wide = dtype.bit_size() > 16;
            match dst_wide {
                None => dst_wide = Some(wide),
                Some(w) => assert!(
                    w == wide,
                    "tt_verify: DST register {id:?} is {dtype:?}; 16-bit and 32-bit tiles cannot share the register file"
                ),
            }
        }
        let budget = if dst_wide == Some(true) { 8 } else { 16 };
        assert!(dst_count <= budget, "tt_verify: {dst_count} DST register tiles exceed the {budget}-tile budget");
        for slot in dst_slots.iter() {
            assert!(*slot < budget, "tt_verify: DST allocation index {slot} is past the {budget}-tile file");
        }
        // The allocated form has no SSA left: no generic memory op (a load
        // turns an effect into a value, a store turns a value into an effect)
        // and no high-level compute may survive `tt_dataflow_select`. Anything that
        // does would be silently dropped by the native emitter otherwise.
        for (id, op) in self.iter_unordered() {
            match op {
                Op::Load { .. } | Op::Store { .. } | Op::Copy { .. } => {
                    panic!("tt_verify: SSA memory op {id:?} survived register allocation")
                }
                Op::Asm { .. } => panic!("tt_verify: Op::Asm in a TT kernel at {id:?}"),
                Op::Reduce { .. } => panic!("tt_verify: reduce {id:?} is not lowered"),
                Op::Unary { .. }
                | Op::Binary { .. }
                | Op::Cast { .. }
                | Op::Bitcast { .. }
                | Op::Select { .. }
                | Op::Stack { .. } => {
                    // Index arithmetic is integer-typed and dead but harmless;
                    // any float value op means the lowering missed it.
                    if self.dtype(id).is_float() {
                        panic!("tt_verify: {id:?} is not fully lowered (SSA compute remains)");
                    }
                }
                _ => {}
            }
        }
    }

    /// Chase a `Load`/`Store`/`Copy` operand through its index expression
    /// to the underlying `Storage`/`Param`. Tile-compute results name DST
    /// slots, not buffers. The index is not needed (render uses slot 0).
    pub(crate) fn tt_storage_of(&self, op_id: OpId) -> OpId {
        match self.at(op_id) {
            Op::Load { src, .. } | Op::Copy { src, .. } => self.tt_storage_of(*src),
            Op::Store { dst, .. } => self.tt_storage_of(*dst),
            Op::Storage { .. } | Op::Param { .. } => op_id,
            Op::TT(TTOp::Sfpu { slot, .. }) => *slot,
            other => panic!("tt_storage_of: {op_id:?} is not a Load/Store/Copy, got {other:?}"),
        }
    }
}

/// The row-op arms against the metalium programs they transcribe, through
/// `kern/sfpu/interp.rs`: same tile in, bit-identical tile out. This is the
/// only coverage for the kinds no ttsim gate can reach (a `Bool` result has no
/// tile format, an `I32` input cannot unpack), and it pins every arm against a
/// refactor. Registers and `Dst` are the interpreter's; both sides run the same
/// lane data, so any divergence is the arm's.
#[cfg(test)]
mod sfp_row_ops {
    use super::*;
    use crate::backend::tenstorrent::kern::sfpu::interp::Vector;
    use crate::backend::tenstorrent::kern::sfpu::kernel::{A_ROW, B_ROW, OUT_ROW};
    use crate::backend::tenstorrent::kern::sfpu::ops::{self, kind_sfpu as sf};
    use crate::kernel::Dev;
    use std::collections::HashMap;
    use tt_isa::isa::Instruction;
    use tt_isa::sfpu as enc;

    /// The arm's ops and the interpreter row of every storage they touch.
    fn arm_ops(kind: u32) -> (Vec<Op>, HashMap<u32, u32>) {
        let mut k = Kernel::new(Dev::Auto);
        let storage =
            |k: &mut Kernel| k.push_back(Op::Storage { dtype: DType::F32, scope: MemScope::Register, len: 1024, id: 0 });
        let (a, b, slot) = (storage(&mut k), storage(&mut k), storage(&mut k));
        let b = if unary(kind) { OpId::NULL } else { b };
        let id = k.push_back(Op::Param { dtype: DType::F32, kind: ParamKind::Global, shape: OpId::NULL, id: 0 });
        assert!(k.tt_sfp_sequence(id, kind, a, b, slot), "kind {kind:#x} is not ported");
        let mut rows: HashMap<u32, u32> = HashMap::new();
        let mut next = 512;
        let mut list: Vec<Op> = Vec::new();
        let mut op = k.head;
        while !op.is_null() {
            if matches!(k.ops[op].op, Op::Storage { .. }) {
                let row = if op == a {
                    A_ROW
                } else if op == b {
                    B_ROW
                } else if op == slot {
                    OUT_ROW
                } else {
                    let r = next;
                    next += 64;
                    r
                };
                rows.insert(op.0, row);
            }
            if !matches!(k.ops[op].op, Op::Storage { .. }) {
                list.push(k.ops[op].op.clone());
            }
            op = k.ops[op].next;
        }
        (list, rows)
    }

    /// The arm's sequence as ISA instructions: `a`, `b` and `slot` at the
    /// interpreter's tile rows, any scratch tile the arm allocates after them.
    fn arm(kind: u32) -> Vec<Instruction> {
        let (list, rows) = arm_ops(kind);
        super::encode_row_ops(&list, &rows)
    }

    /// Every ported kind.
    fn kinds() -> Vec<u32> {
        let mut all = vec![
            sf::BOOL_NOT,
            sf::NEG,
            sf::ABS,
            sf::INT_NOT,
            sf::SQRT,
            sf::RSQRT,
            sf::RECIP,
            sf::FLOOR,
            sf::TRUNC,
        ];
        all.extend([
            crate::backend::tenstorrent::kern::kind::ADD,
            crate::backend::tenstorrent::kern::kind::SUB,
            crate::backend::tenstorrent::kern::kind::MUL,
            sf::DIV,
            sf::EQ,
            sf::NE,
            sf::GT,
            sf::GE,
            sf::LT,
            sf::LE,
            sf::INT_AND,
            sf::INT_OR,
            sf::INT_XOR,
            sf::INT_SHL,
            sf::INT_SHR,
        ]);
        all
    }

    /// The register model names every lane register the arms touch, and only
    /// the writable ones on the write side. A missing opcode panics inside
    /// `tt_sfp_regs`; a false write would corrupt a merged body.
    #[test]
    fn the_register_model_covers_every_ported_kind() {
        let bits = |set: &Set<u16>| set.iter().fold(0u16, |m, r| m | (1u16 << r));
        for kind in kinds() {
            let (list, _) = arm_ops(kind);
            for op in &list {
                if let Op::TT(TTOp::Sfp(i)) = op {
                    let (_, writes) = super::tt_sfp_regs(i.op, &i.args);
                    let writes = bits(&writes);
                    assert_eq!(writes & !0x00ff, 0, "kind {kind:#x} writes a read-only lane register");
                }
            }
        }
        // The in-place and predicate forms are the ones a table can get wrong.
        let (iadd, _) = (tt_isa::isa::generated::defs::SFPIADD.opcode(), 0);
        let (r, w) = super::tt_sfp_regs(iadd, &[0, 1, 2, 5, 0]);
        assert_eq!((bits(&r), bits(&w)), (1 << 1 | 1 << 2, 1 << 2), "SFPIADD reads VC and VD, writes VD");
        let (r, w) = super::tt_sfp_regs(tt_isa::isa::generated::defs::SFPSETCC.opcode(), &[0, 3, 0, 0, 0]);
        assert_eq!((bits(&r), bits(&w)), (1 << 3, 0), "SFPSETCC reads VC only and writes flags");
        let (r, w) = super::tt_sfp_regs(tt_isa::isa::generated::defs::SFPGT.opcode(), &[4, 6, 1, 0, 0]);
        assert_eq!((bits(&r), bits(&w)), (1 << 4 | 1 << 6, 0), "SFPGT reads VC and VD, writes flags");
    }

    /// Every ported kind's arity, so the arm sees the null second operand a
    /// unary emit carries.
    fn unary(kind: u32) -> bool {
        matches!(kind, sf::BOOL_NOT | sf::NEG | sf::ABS | sf::INT_NOT | sf::SQRT | sf::RSQRT | sf::RECIP | sf::FLOOR | sf::TRUNC)
    }

    fn run(program: &[Instruction], a: &[u32], b: &[u32]) -> Vec<u32> {
        let mut v = Vector::new();
        v.put_tile(A_ROW as usize, a);
        v.put_tile(B_ROW as usize, b);
        v.run(program).unwrap();
        v.tile(OUT_ROW as usize)
    }

    /// Specials (`±0`, `±inf`, `±NaN`, denormals, `±MAX`, a few finite
    /// values) mixed with a deterministic word stream.
    fn tile(seed: u32) -> Vec<u32> {
        let specials = [
            0x0000_0000,
            0x8000_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0xffc0_0000,
            1,
            0x8000_0001,
            0x0080_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x3f80_0000,
            0xbf80_0000,
            0x3f00_0000,
            0xbf00_0000,
            0x4b00_0000,
            0xcb00_0000,
            0x4b80_0000,
            0x4048_f5c3,
            0xc049_0fdb,
            0x7f00_0000,
            0xff00_0000,
            0x0000_0001,
        ];
        (0..1024)
            .map(|i| {
                if i % 3 == 0 {
                    specials[(i as u32 + seed) as usize % specials.len()]
                } else {
                    (i as u32).wrapping_mul(0x9e37_79b9).wrapping_add(seed)
                }
            })
            .collect()
    }

    fn compare(kind: u32) {
        let (a, b) = (tile(7), tile(11));
        let (_, code) = ops::code2(kind, [0.0; 2]).unwrap();
        let want = run(&code.expand(), &a, &b);
        let got = run(&arm(kind), &a, &b);
        assert_eq!(got, want, "kind {kind:#x}");
    }

    #[test]
    fn unary_arms_match_the_builder() {
        for kind in [
            sf::BOOL_NOT,
            sf::NEG,
            sf::ABS,
            sf::INT_NOT,
            sf::SQRT,
            sf::RSQRT,
            sf::RECIP,
            sf::FLOOR,
            sf::TRUNC,
        ] {
            compare(kind);
        }
    }

    #[test]
    fn binary_arms_match_the_builder() {
        for kind in [
            crate::backend::tenstorrent::kern::kind::ADD,
            crate::backend::tenstorrent::kern::kind::SUB,
            crate::backend::tenstorrent::kern::kind::MUL,
            sf::DIV,
            sf::EQ,
            sf::NE,
            sf::GT,
            sf::GE,
            sf::LT,
            sf::LE,
            sf::INT_AND,
            sf::INT_OR,
            sf::INT_XOR,
            sf::INT_SHL,
            sf::INT_SHR,
        ] {
            compare(kind);
        }
    }
}

/// The row ops as ISA instructions for the interpreter, the row map naming
/// each storage's tile. A `Replay` is expanded into its iterations with the
/// row address modifier's step (two rows per group), which is what the
/// hardware's recorded body does.
#[cfg(test)]
fn encode_row_ops(ops: &[Op], rows: &std::collections::HashMap<u32, u32>) -> Vec<tt_isa::isa::Instruction> {
    use tt_isa::isa::Instruction;
    use tt_isa::sfpu as enc;
    fn one(op: &Op, rows: &std::collections::HashMap<u32, u32>, off: u32) -> Vec<Instruction> {
        match op {
            Op::TT(TTOp::SfpLoad(s)) => {
                vec![enc::load(u32::from(s.reg), u32::from(s.fmt), 0, rows[&s.slot.0] + u32::from(s.at) + off).unwrap()]
            }
            Op::TT(TTOp::SfpStore(s)) => {
                vec![enc::store(u32::from(s.reg), u32::from(s.fmt), 0, rows[&s.slot.0] + u32::from(s.at) + off).unwrap()]
            }
            Op::TT(TTOp::SfpConst(c)) => {
                if c.bits & 0xffff == 0 {
                    vec![enc::loadi(u32::from(c.reg), enc::loadi_mode::FLOATB, c.bits >> 16).unwrap()]
                } else {
                    enc::load_f32(u32::from(c.reg), c.bits).unwrap().to_vec()
                }
            }
            Op::TT(TTOp::Sfp(i)) => vec![crate::codegen::tenstorrent::tt_sfp_encode(i).unwrap()],
            other => panic!("unexpected op {other:?}"),
        }
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i < ops.len() {
        if let Op::TT(TTOp::Replay { len, count }) = ops[i] {
            let mut j = 0;
            let mut words = 0u32;
            while words < u32::from(len) {
                words += tt_row_words(&ops[i + 1 + j]);
                j += 1;
            }
            for k in 0..u32::from(count) {
                for op in &ops[i + 1..i + 1 + j] {
                    out.extend(one(op, rows, 2 * k));
                }
            }
            i += 1 + j;
        } else {
            out.extend(one(&ops[i], rows, 0));
            i += 1;
        }
    }
    out
}

/// Coverage for the three non-kind lowering arms (`Bitcast`, `Cast`,
/// `Select`): each arm's instruction stream through `interp`, against the
/// host's own semantics. They have no metalium builder program, so there is
/// no builder to A/B; and a `Bool` condition cannot be packed for ttsim, so
/// these are the only coverage a `Select` can have.
#[cfg(test)]
mod tt_lower_ops {
    use super::*;
    use crate::backend::tenstorrent::kern::sfpu::interp::Vector;
    use crate::backend::tenstorrent::kern::sfpu::kernel::{A_ROW, B_ROW, OUT_ROW};
    use crate::kernel::Dev;
    use std::collections::HashMap;
    use tt_isa::isa::Instruction;
    use tt_isa::sfpu as enc;

    /// The scratch row the fourth tile takes (`A_ROW`/`B_ROW` are 0 and 64).
    const C_ROW: u32 = 192;

    /// Four `F32` `Register` tiles and their interpreter rows.
    fn tiles() -> (Kernel, [OpId; 4], HashMap<u32, u32>) {
        let mut k = Kernel::new(Dev::Auto);
        let mut ids = [OpId::NULL; 4];
        let mut rows = HashMap::new();
        for (id, row) in ids.iter_mut().zip([A_ROW, B_ROW, C_ROW, OUT_ROW]) {
            *id = k.push_back(Op::Storage { dtype: DType::F32, scope: MemScope::Register, len: 1024, id: 0 });
            rows.insert(id.0, row);
        }
        (k, ids, rows)
    }

    /// The arm's ops as ISA instructions, rows from `rows`.
    fn encode(ops: Vec<Op>, rows: &HashMap<u32, u32>) -> Vec<Instruction> {
        super::encode_row_ops(&ops, rows)
    }

    fn run(program: &[Instruction], a: &[u32], b: &[u32], c: &[u32]) -> Vec<u32> {
        let mut v = Vector::new();
        for (row, data) in [(A_ROW, a), (B_ROW, b), (C_ROW, c)] {
            if !data.is_empty() {
                v.put_tile(row as usize, data);
            }
        }
        v.run(program).unwrap();
        v.tile(OUT_ROW as usize)
    }

    /// Specials mixed with a deterministic word stream.
    fn tile(seed: u32) -> Vec<u32> {
        let specials = [
            0x0000_0000,
            0x8000_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0xffc0_0000,
            0x4f00_0000,
            0xcf00_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x0080_0000,
            0x0000_0001,
            0x4b00_0000,
            0xcb00_0000,
        ];
        (0..1024)
            .map(|i| {
                if i % 3 == 0 {
                    specials[(i as u32 + seed) as usize % specials.len()]
                } else {
                    (i as u32).wrapping_mul(0x9e37_79b9).wrapping_add(seed)
                }
            })
            .collect()
    }

    #[test]
    fn bitcast_copies_the_raw_bits() {
        let (mut k, ids, rows) = tiles();
        let ops = k.tt_sfp_bitcast(ids[0], ids[3]);
        let a = tile(5);
        let got = run(&encode(ops, &rows), &a, &[], &[]);
        assert_eq!(got, a);
    }

    #[test]
    fn cast_i32_to_f32_matches_the_host() {
        let (mut k, ids, rows) = tiles();
        let ops = k.tt_sfp_cast_i32_f32(ids[0], ids[3]);
        let a = tile(7);
        let got = run(&encode(ops, &rows), &a, &[], &[]);
        let want: Vec<u32> = a.iter().map(|w| (*w as i32 as f32).to_bits()).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn cast_f32_to_i32_matches_the_host() {
        let (mut k, ids, rows) = tiles();
        let ops = k.tt_sfp_cast_f32_i32(ids[0], ids[3]);
        let a = tile(11);
        let got = run(&encode(ops, &rows), &a, &[], &[]);
        let want: Vec<u32> = a.iter().map(|w| (f32::from_bits(*w) as i32) as u32).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn int_add_wraps_like_the_host() {
        // `SFPIADD`, not the float arm: the sum is the two words' two's
        // complement integer sum, wrap included.
        let (mut k, ids, rows) = tiles();
        let ops = k.tt_sfp_int_add(ids[0], ids[1], ids[3]);
        let (a, b) = (tile(23), tile(29));
        let got = run(&encode(ops, &rows), &a, &b, &[]);
        let want: Vec<u32> = a.iter().zip(b.iter()).map(|(x, y)| (*x as i32).wrapping_add(*y as i32) as u32).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn select_picks_on_a_nonzero_condition() {
        let (mut k, ids, rows) = tiles();
        let ops = k.tt_sfp_select(ids[0], ids[1], ids[2], ids[3]);
        let cond: Vec<u32> = (0..1024)
            .map(|i| match i % 5 {
                0 => 0,
                1 => 1,
                _ => (i as u32).wrapping_mul(0x9e37_79b9),
            })
            .collect();
        let (t, f) = (tile(13), tile(17));
        let got = run(&encode(ops, &rows), &cond, &t, &f);
        let want: Vec<u32> = (0..1024).map(|i| if cond[i] != 0 { t[i] } else { f[i] }).collect();
        assert_eq!(got, want);
    }
}
