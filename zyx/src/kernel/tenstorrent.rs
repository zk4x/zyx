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

/// Spills the fusion accepts in one cone: each costs one store and one load per
/// reader, so a cone that needs more is not worth keeping whole.
const FUSE_SPILL_LIMIT: usize = 4;
/// Spills the allocator may materialize in one body. Past this the body is
/// mis-shaped and must fail loudly rather than spill forever.
const ASSIGN_SPILL_LIMIT: usize = 8;

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

/// What an instruction does to the lane flags (`use_flags` + `lane_flags`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum TtFlags {
    Keep,
    /// `SFPENCC`: every lane is enabled again.
    Reset,
    /// `SFPSETCC` / `SFPCOMPC`: a subset of the lanes is enabled.
    Mask,
    Push,
    Pop,
}

fn tt_flag_effect(op: &Op) -> TtFlags {
    use tt_isa::isa::generated::defs;
    let Op::TT(TTOp::Sfp(insn)) = op else {
        return TtFlags::Keep;
    };
    if insn.op == defs::SFPENCC.opcode() {
        TtFlags::Reset
    } else if insn.op == defs::SFPSETCC.opcode() || insn.op == defs::SFPCOMPC.opcode() {
        TtFlags::Mask
    } else if insn.op == defs::SFPPUSHC.opcode() {
        TtFlags::Push
    } else if insn.op == defs::SFPPOPC.opcode() {
        TtFlags::Pop
    } else {
        TtFlags::Keep
    }
}

/// The mask state before and after each instruction: `true` means every lane is
/// enabled. A split may only store/reload at such a point, because a lane-masked
/// store copies only the enabled lanes and a lane-masked load writes only them.
fn tt_mask_states(effects: &[TtFlags]) -> (Vec<bool>, Vec<bool>) {
    let mut full = false;
    let mut stack: Vec<bool> = Vec::new();
    let mut before = Vec::with_capacity(effects.len());
    let mut after = Vec::with_capacity(effects.len());
    for &e in effects {
        before.push(full);
        match e {
            TtFlags::Reset => full = true,
            TtFlags::Mask => full = false,
            TtFlags::Push => stack.push(full),
            TtFlags::Pop => full = stack.pop().unwrap_or(false),
            TtFlags::Keep => {}
        }
        after.push(full);
    }
    (before, after)
}

/// Assign the lane registers to a straight-line body: intervals in, registers
/// out, no policy. Liveness is exact (one block), so the scan is optimal and
/// `None` means the body needs more registers than exist -- which the caller
/// removes by splitting, not by guessing here.
fn tt_assign_regs(body: &[(Set<u16>, Set<u16>)]) -> Option<Map<u16, u8>> {
    let mut def: Map<u16, i64> = Map::default();
    let mut last: Map<u16, i64> = Map::default();
    let mut first_use: Map<u16, i64> = Map::default();
    for (k, (reads, writes)) in body.iter().enumerate() {
        for &r in writes {
            def.entry(r).or_insert(k as i64);
            last.insert(r, k as i64);
        }
        for &r in reads {
            first_use.entry(r).or_insert(k as i64);
            last.insert(r, k as i64);
        }
    }
    for (&r, &use_at) in &first_use {
        if def.get(&r).is_none_or(|&d| use_at < d) {
            def.insert(r, -1);
        }
    }
    let mut order: Vec<u16> = last.keys().copied().filter(|r| *r >= 16).collect();
    order.sort_by_key(|r| (def.get(r).copied().unwrap_or(-1), *r));
    let mut assign: Map<u16, u8> = Map::default();
    let mut active: Vec<(u16, u8, i64)> = Vec::new();
    for r in order {
        let start = def.get(&r).copied().unwrap_or(-1);
        let end = last.get(&r).copied().unwrap_or(start);
        active.retain(|&(_, _, l)| l >= start);
        let used: u16 = active.iter().fold(0, |m, &(_, p, _)| m | 1u16 << p);
        let p = (0..8u8).find(|p| used & (1u16 << p) == 0)?;
        assign.insert(r, p);
        active.push((r, p, end));
    }
    Some(assign)
}

/// The value to split and the window to split it over, as `(name, a, b)`:
/// store it before `a`, reload it after `b`. One store and one reload take it
/// out of the register file across the body's over-budget run, instead of one
/// reload per use.
///
/// Constraints, each of which cost a bug to find:
/// - the value must be **live at the peak** (a dead value cannot lower it);
/// - it must not be read or written anywhere in the window;
/// - it must be **defined under a full mask**, so every lane is valid;
/// - the store must be **strictly before the peak** (a store at the peak keeps
///   the value live there and the fixpoint would loop);
/// - the window ends must be **full-mask** points (a lane-masked store/load is
///   not a bit-exact move).
fn tt_split_at_pressure(
    body: &[(Set<u16>, Set<u16>)],
    before: &[bool],
    after: &[bool],
    budget: usize,
) -> Option<(u16, usize, usize)> {
    let n = body.len();
    let mut def: Map<u16, i64> = Map::default();
    let mut last: Map<u16, i64> = Map::default();
    let mut first_use: Map<u16, i64> = Map::default();
    for (k, (reads, writes)) in body.iter().enumerate() {
        for &r in writes {
            def.entry(r).or_insert(k as i64);
            last.insert(r, k as i64);
        }
        for &r in reads {
            first_use.entry(r).or_insert(k as i64);
            last.insert(r, k as i64);
        }
    }
    for (&r, &use_at) in &first_use {
        if def.get(&r).is_none_or(|&d| use_at < d) {
            def.insert(r, -1);
        }
    }
    let names: Vec<u16> = last.keys().copied().filter(|r| *r >= 16).collect();
    let counts: Vec<usize> = (0..n as i64)
        .map(|k| {
            names.iter().filter(|r| def.get(r).copied().unwrap_or(-1) <= k && last.get(r).copied().unwrap_or(-1) >= k).count()
        })
        .collect();
    let (peak, pressure) = counts.iter().enumerate().max_by_key(|(_, c)| **c)?;
    if *pressure <= budget {
        return None;
    }
    let (mut ra, mut rb) = (peak, peak);
    while ra > 0 && counts[ra - 1] > budget {
        ra -= 1;
    }
    while rb + 1 < n && counts[rb + 1] > budget {
        rb += 1;
    }
    let mut best: Option<(usize, u16, usize, usize)> = None; // (coverage, name, a, b)
    for &r in &names {
        let d = def.get(&r).copied().unwrap_or(-1);
        let l = last.get(&r).copied().unwrap_or(-1);
        if d < 0 || !before.get(d as usize).copied().unwrap_or(false) {
            continue;
        }
        if d > peak as i64 || l < peak as i64 {
            continue;
        }
        let uses: Vec<usize> = (0..n).filter(|&k| body[k].0.contains(&r) || body[k].1.contains(&r)).collect();
        if uses.iter().any(|&k| k >= ra && k <= rb) {
            continue;
        }
        let prev = uses.iter().copied().filter(|&k| k < ra).max().unwrap_or(d as usize);
        let next = uses.iter().copied().filter(|&k| k > rb).min().unwrap_or(n);
        let Some(a) = (prev + 1..=ra).find(|&k| before.get(k).copied().unwrap_or(false)) else {
            continue;
        };
        let Some(b) = (rb..next).rev().find(|&k| after.get(k).copied().unwrap_or(false)) else {
            continue;
        };
        if a >= peak || b < peak {
            continue;
        }
        let cover = b - a + 1;
        if best.is_none_or(|(c, _, _, _)| cover > c) {
            best = Some((cover, r, a, b));
        }
    }
    best.map(|(_, name, a, b)| (name, a, b))
}

/// Simulate the split loop on an abstract body: find the over-budget run, split
/// one value around it, repeat until the body fits. Returns the words the
/// splits add, or `None` when more than `max_splits` would be needed (or no
/// split can lower the peak). The fusion uses this to decide whether a cone is
/// worth keeping before any op is touched.
fn tt_plan_splits(body: &[(Set<u16>, Set<u16>)], effects: &[TtFlags], budget: usize, max_splits: usize) -> Option<usize> {
    let mut body: Vec<(Set<u16>, Set<u16>)> = body.to_vec();
    let mut effects: Vec<TtFlags> = effects.to_vec();
    let mut next = body.iter().flat_map(|(r, w)| r.iter().chain(w)).copied().filter(|r| *r >= 16).max().map_or(16, |m| m + 1);
    let mut added = 0usize;
    for splits in 0..=max_splits {
        let (before, after) = tt_mask_states(&effects);
        let Some((name, a, b)) = tt_split_at_pressure(&body, &before, &after, budget) else {
            // No split left: feasible only if the body already fits.
            return tt_assign_regs(&body).is_some().then_some(added);
        };
        if splits == max_splits {
            return None;
        }
        // Simulate: store before `a`, reload after `b`, later uses renamed.
        let fresh = next;
        next += 1;
        let mut grown: Vec<(Set<u16>, Set<u16>)> = Vec::with_capacity(body.len() + 2);
        let mut grown_effects: Vec<TtFlags> = Vec::with_capacity(effects.len() + 2);
        for (i, (reads, writes)) in body.iter().enumerate() {
            if i == a {
                grown.push((Set::from_iter([name]), Set::default()));
                grown_effects.push(TtFlags::Keep);
            }
            let (mut r, mut w) = (reads.clone(), writes.clone());
            if i > b {
                if r.remove(&name) {
                    r.insert(fresh);
                }
                if w.remove(&name) {
                    w.insert(fresh);
                }
            }
            grown.push((r, w));
            grown_effects.push(effects[i]);
            if i == b {
                grown.push((Set::default(), Set::from_iter([fresh])));
                grown_effects.push(TtFlags::Keep);
            }
        }
        body = grown;
        effects = grown_effects;
        added += 2;
    }
    None
}

/// Linear-scan allocation with spilling: the classic block-local algorithm
/// (Poletto/Sarkar, LLVM's `RegAllocLinearScan`). Intervals are exact because a
/// row body is one straight-line block; at each start the active list expires
/// what has ended, and when no register is free the interval that *loses* is
/// the one whose last use is farthest away. A spilled name keeps no register:
/// the caller stores it at each definition and reloads it before each use.
/// Returns the assignment and the names to spill, or `None` when more than
/// `max_spills` values would have to be spilled.
#[allow(dead_code)] // exercised by the register-allocator unit tests; the fusion calls `tt_plan_spills`
fn tt_alloc_spill(body: &[(Set<u16>, Set<u16>)], max_spills: usize) -> Option<(Map<u16, u8>, Vec<u16>)> {
    // Which values go to memory: the ones that are live at the body's peak but
    // not used there. This is the peak, not the interval start -- a straight
    // line body can have nine values alive at one instruction while every
    // interval start looks fine, and evicting by interval start then cascades
    // (the freed register is needed again before the next use).
    let spills = tt_peak_spills(body);
    if spills.len() > max_spills {
        return None;
    }
    let spilled: Set<u16> = spills.iter().copied().collect();
    let mut def: Map<u16, i64> = Map::default();
    let mut last: Map<u16, i64> = Map::default();
    let mut first_use: Map<u16, i64> = Map::default();
    for (k, (reads, writes)) in body.iter().enumerate() {
        for &r in writes {
            def.entry(r).or_insert(k as i64);
            last.insert(r, k as i64);
        }
        for &r in reads {
            first_use.entry(r).or_insert(k as i64);
            last.insert(r, k as i64);
        }
    }
    // Read before any write in the body: live from the start.
    for (&r, &use_at) in &first_use {
        if def.get(&r).is_none_or(|&d| use_at < d) {
            def.insert(r, -1);
        }
    }
    let mut order: Vec<u16> = last.keys().copied().filter(|r| *r >= 16 && !spilled.contains(r)).collect();
    order.sort_by_key(|r| (def.get(r).copied().unwrap_or(-1), *r));
    let mut assign: Map<u16, u8> = Map::default();
    let mut active: Vec<(u16, u8, i64)> = Vec::new(); // (name, phys, last use)
    for r in order {
        let start = def.get(&r).copied().unwrap_or(-1);
        let end = last.get(&r).copied().unwrap_or(start);
        active.retain(|&(_, _, l)| l >= start);
        let used: u16 = active.iter().fold(0, |m, &(_, p, _)| m | 1u16 << p);
        if let Some(p) = (0..8u8).find(|p| used & (1u16 << p) == 0) {
            assign.insert(r, p);
            active.push((r, p, end));
        } else {
            // The peak set was supposed to make this unreachable; if it is
            // reached the body needs more than eight *simultaneous* reads and
            // the caller must refuse it rather than guess.
            return None;
        }
    }
    Some((assign, spills))
}

/// The values to keep in memory so the body fits eight registers, chosen at the
/// body's pressure peak. A spilled value is live only at its defining
/// instruction and at each read (that is what materialization produces: one
/// store at the definition, one reload per read), so spilling removes it from
/// every other point.
#[allow(dead_code)] // called by `tt_alloc_spill`, itself test-only for now
fn tt_peak_spills(body: &[(Set<u16>, Set<u16>)]) -> Vec<u16> {
    let mut def: Map<u16, i64> = Map::default();
    let mut last: Map<u16, i64> = Map::default();
    let mut first_use: Map<u16, i64> = Map::default();
    for (k, (reads, writes)) in body.iter().enumerate() {
        for &r in writes {
            def.entry(r).or_insert(k as i64);
            last.insert(r, k as i64);
        }
        for &r in reads {
            first_use.entry(r).or_insert(k as i64);
            last.insert(r, k as i64);
        }
    }
    for (&r, &use_at) in &first_use {
        if def.get(&r).is_none_or(|&d| use_at < d) {
            def.insert(r, -1);
        }
    }
    let mut names: Vec<u16> = last.keys().copied().filter(|r| *r >= 16).collect();
    names.sort_unstable();
    let mut reads_count: Map<u16, usize> = Map::default();
    for (reads, _) in body {
        for r in reads {
            *reads_count.entry(*r).or_default() += 1;
        }
    }
    let mut spilled: Set<u16> = Set::default();
    // Pressure at each instruction under a spill set: values used there (a
    // spilled value at its definition still needs a register for the store)
    // plus the ones merely live through it.
    let pressure = |spilled: &Set<u16>| -> (usize, usize, Vec<u16>) {
        let mut peak: (usize, usize, Vec<u16>) = (0, 0, Vec::new());
        for (k, (reads, _)) in body.iter().enumerate() {
            let mut here = 0usize;
            let mut live: Vec<u16> = Vec::new();
            for &n in &names {
                let used = reads.contains(&n) || (spilled.contains(&n) && def.get(&n).copied().unwrap_or(-1) == k as i64);
                if used {
                    here += 1;
                    continue;
                }
                let alive = !spilled.contains(&n)
                    && def.get(&n).copied().unwrap_or(-1) <= k as i64
                    && last.get(&n).copied().unwrap_or(-1) >= k as i64;
                if alive {
                    live.push(n);
                }
            }
            let total = here + live.len();
            if total > peak.0 {
                peak = (total, k, live);
            }
        }
        peak
    };
    loop {
        let peak = pressure(&spilled);
        if peak.0 <= 8 {
            return spilled.into_iter().collect();
        }
        // Choose at the peak, with lookahead: the cheapest candidate (fewest
        // reads; a spill is one store plus one reload per read) whose eviction
        // keeps *every* instruction at eight or fewer. Without the lookahead a
        // cheap eviction just moves the peak to the evicted value's next read.
        let mut best: Option<(usize, std::cmp::Reverse<i64>, u16)> = None;
        for &c in &peak.2 {
            let mut trial = spilled.clone();
            trial.insert(c);
            if pressure(&trial).0 > 8 {
                continue;
            }
            let key = (reads_count.get(&c).copied().unwrap_or(0), std::cmp::Reverse(last.get(&c).copied().unwrap_or(-1)), c);
            if best.is_none_or(|b| key < b) {
                best = Some(key);
            }
        }
        // Nothing is feasible on its own: take the cheapest candidate anyway and
        // let the next iteration fix the peak the eviction creates (the next
        // round also has the lookahead, so this does not cascade blindly).
        let victim = best.map(|(_, _, c)| c).or_else(|| {
            peak.2
                .iter()
                .copied()
                .min_by_key(|n| (reads_count.get(n).copied().unwrap_or(0), std::cmp::Reverse(last.get(n).copied().unwrap_or(-1))))
        });
        let Some(victim) = victim else {
            // Every value at the peak is read there: more than eight reads in
            // one instruction, which no spill can fix.
            return spilled.into_iter().collect();
        };
        spilled.insert(victim);
    }
}

/// Simulate [`tt_alloc_spill`] plus its materialization on an abstract body:
/// spill, splice a reload before each read and a store after each write, and
/// plan again -- exactly what `tt_lreg_alloc` does to the real ops. Returns the
/// number of words the spills will add, or `None` when more than `max_spills`
/// values would have to be spilled. The fusion uses this to decide whether a
/// cone is worth keeping before any op is touched.
#[allow(dead_code)] // exercised by the register-allocator unit tests until the fusion calls it
fn tt_plan_spills(body: &[(Set<u16>, Set<u16>)], max_spills: usize) -> Option<(usize, Vec<u16>)> {
    let mut body: Vec<(Set<u16>, Set<u16>)> = body.to_vec();
    let mut next = body.iter().flat_map(|(r, w)| r.iter().chain(w)).copied().filter(|r| *r >= 16).max().map_or(16, |m| m + 1);
    let mut added = 0usize;
    let mut spilled = 0usize;
    let mut names: Vec<u16> = Vec::new();
    loop {
        let spills = tt_alloc_spill(&body, usize::MAX).expect("no limit").1;
        if spills.is_empty() {
            return Some((added, names));
        }
        names.extend(spills.iter().copied());
        spilled += spills.len();
        if spilled > max_spills {
            return None;
        }
        let mut grown: Vec<(Set<u16>, Set<u16>)> = Vec::with_capacity(body.len() * 2);
        for (reads, writes) in &body {
            let read = spills.iter().find(|s| reads.contains(s)).copied();
            let write = spills.iter().find(|s| writes.contains(s)).copied();
            if read.is_none() && write.is_none() {
                grown.push((reads.clone(), writes.clone()));
                continue;
            }
            let mut reg = None;
            let mut r = reads.clone();
            if let Some(name) = read {
                let fresh = next;
                next += 1;
                grown.push((Set::default(), Set::from_iter([fresh])));
                added += 1;
                r.remove(&name);
                r.insert(fresh);
                reg = Some(fresh);
            }
            let mut w = writes.clone();
            if let Some(name) = write {
                // A read-modify-write is renamed whole, as the allocator does.
                w.remove(&name);
                if let Some(fresh) = reg {
                    w.insert(fresh);
                }
            }
            grown.push((r, w));
            if let Some(name) = write {
                grown.push((Set::from_iter([reg.unwrap_or(name)]), Set::default()));
                added += 1;
            }
        }
        body = grown;
    }
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
    use crate::codegen::tenstorrent::sfpu::ops::kind_sfpu as sf;
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
    use crate::codegen::tenstorrent::kind;
    use crate::codegen::tenstorrent::sfpu::ops::kind_sfpu as sf;
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
    /// A loop-carried `Register` accumulator's DST slot. The storage is a
    /// logical name; the value lives in the slot its producer writes, and the
    /// loop's next iteration reads that slot in place.
    reg_slot: Map<OpId, OpId>,
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

    /// CB sync constructor: `cb.reserve_back(n)` — open `n` back slots
    /// of circular buffer `cb` for writing.
    pub fn tt_reserve_back(&mut self, cb: OpId, n: u8) -> OpId {
        debug_assert!(
            matches!(self.at(cb), Op::Storage { scope: MemScope::Circular, .. }),
            "tt_reserve_back: cb {cb} is not a Circular storage"
        );
        debug_assert!(n != 0, "tt_reserve_back: n must be positive");
        self.push_back(Op::TT(TTOp::ReserveBack { cb, n }))
    }

    /// CB sync constructor: `cb.push_back(n)` — publish `n` written slots.
    pub fn tt_push_back(&mut self, cb: OpId, n: u8) -> OpId {
        debug_assert!(
            matches!(self.at(cb), Op::Storage { scope: MemScope::Circular, .. }),
            "tt_push_back: cb {cb} is not a Circular storage"
        );
        debug_assert!(n != 0, "tt_push_back: n must be positive");
        self.push_back(Op::TT(TTOp::PushBack { cb, n }))
    }

    /// CB sync constructor: `cb.wait_front(n)` — block until `n` front
    /// slots hold data.
    pub fn tt_wait_front(&mut self, cb: OpId, n: u8) -> OpId {
        debug_assert!(
            matches!(self.at(cb), Op::Storage { scope: MemScope::Circular, .. }),
            "tt_wait_front: cb {cb} is not a Circular storage"
        );
        debug_assert!(n != 0, "tt_wait_front: n must be positive");
        self.push_back(Op::TT(TTOp::WaitFront { cb, n }))
    }

    /// CB sync constructor: `cb.pop_front(n)` — release `n` consumed
    /// slots.
    pub fn tt_pop_front(&mut self, cb: OpId, n: u8) -> OpId {
        debug_assert!(
            matches!(self.at(cb), Op::Storage { scope: MemScope::Circular, .. }),
            "tt_pop_front: cb {cb} is not a Circular storage"
        );
        self.push_back(Op::TT(TTOp::PopFront { cb, n }))
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
        // Normalize the loop tiling the search left to the one shape the
        // tile IR has: `Loop(k_tiles) + Unroll(32)`. Three input shapes
        // reach here.
        //
        // - A loop in **element** units (the unsplit reduce): pad to 32 and
        //   split. That is where the lane axis and the `Reduce` come from.
        // - A loop the search **strip-mined** (`SplitMode::Loop`:
        //   `factor*outer + inner`): fold the pair back into one loop of
        //   `outer*inner` trips, then pad + split. Padding the outer alone
        //   counts the inner trips again: softmax's 8-tile reduce became
        //   32 iterations and a 4 MB reader block.
        // - A loop the search already **split into lanes**
        //   (`SplitMode::Unroll`): its trip count is tiles, not elements,
        //   so it is left alone (same over-read if padded).
        //
        // Folding validates the whole pair before mutating: the inner var's
        // only use is the composition split leaves right after the inner
        // header, the outer var is used by nothing inside the inner body, and
        // no side effect sits in the outer body between the two loops (it
        // would run once per flat iteration instead of once per outer one).
        for _ in 0..64 {
            let mut inner_ops: Vec<OpId> = Vec::new();
            for (op, x) in self.iter_unordered() {
                if matches!(x, Op::Loop { .. }) {
                    inner_ops.push(op);
                }
            }
            let mut folded = false;
            for inner in inner_ops {
                let Some(outer) = self.tt_enclosing_loop(inner) else {
                    continue;
                };
                let composed = self.next_op(inner);
                if composed.is_null() {
                    continue;
                }
                let Op::Binary { x, y, bop: BOp::Add } = self.ops[composed].op else {
                    continue;
                };
                let acc = if y == inner {
                    x
                } else if x == inner {
                    y
                } else {
                    continue;
                };
                if !self.depends_on(acc, outer, &mut Set::default()) {
                    continue;
                }
                let (Op::Loop { len: outer_len }, Op::Loop { len: inner_len }) =
                    (self.ops[outer].op.clone(), self.ops[inner].op.clone())
                else {
                    continue;
                };
                let (Some(ol), Some(il)) = (
                    self.resolve_const(outer_len).and_then(Constant::as_dim),
                    self.resolve_const(inner_len).and_then(Constant::as_dim),
                ) else {
                    continue;
                };
                // The fold rewrites the composition's value to the flat loop
                // variable, which enumerates `outer * il + inner`. The
                // substitution is exact only when the accumulator side
                // advances by exactly `il` per outer trip: in
                // `acc + inner`, `acc` must have outer coefficient `il`.
                // `depends_on` alone is not enough — a strip-mined index that
                // also carries a base (`row * cols + outer * 32`, what a
                // fused reduce over `exp(x - max)` produces) would lose that
                // base. The invariant part of `acc` is kept below.
                let Some(strides) = self.get_strides(acc) else {
                    continue;
                };
                let Some(&(_, scale)) = strides.get(&outer) else {
                    continue;
                };
                if scale != il {
                    continue;
                }
                let inner_end = self.get_last_dim_op(inner);
                let outer_end = self.get_last_dim_op(outer);
                // The inner var belongs to the composition alone.
                let mut ok = true;
                for (op, x) in self.iter_unordered() {
                    if op != composed && x.parameters().any(|p| p == inner) {
                        ok = false;
                        break;
                    }
                }
                // A value is *tainted* when it reaches the outer var without
                // going through the composition. After folding the outer var
                // enumerates the flat trips, so a tainted value read inside
                // the inner body (or after it) would see the wrong index. The
                // composition itself bridges the two loop names: its consumers
                // are downstream of the flat index and stay correct.
                let mut tainted: Set<OpId> = Set::default();
                let mut walk = self.head;
                while !walk.is_null() {
                    if walk != composed
                        && self.ops[walk].op.parameters().any(|p| p == outer || (p != composed && tainted.contains(&p)))
                    {
                        tainted.insert(walk);
                    }
                    walk = self.next_op(walk);
                }
                // Between the two loops: the composition's own math, pure.
                let mut scan = self.next_op(outer);
                while ok && !scan.is_null() && scan != inner {
                    if matches!(self.ops[scan].op, Op::Store { .. } | Op::Copy { .. }) {
                        ok = false;
                    }
                    scan = self.next_op(scan);
                }
                // The inner body may write its accumulator, but no tainted
                // value may reach it.
                let mut scan = self.next_op(inner);
                while ok && !scan.is_null() && scan != inner_end {
                    if scan != composed && tainted.contains(&scan) {
                        ok = false;
                    }
                    scan = self.next_op(scan);
                }
                // After the inner loop, still in the outer body: no side
                // effects (they would run per flat trip), no tainted value.
                let mut scan = self.next_op(inner_end);
                while ok && !scan.is_null() && scan != outer_end {
                    if matches!(self.ops[scan].op, Op::Store { .. } | Op::Copy { .. }) || tainted.contains(&scan) {
                        ok = false;
                    }
                    scan = self.next_op(scan);
                }
                if !ok {
                    continue;
                }
                let flat = self.insert_const_idx_before(outer, ol * il);
                self.ops[outer].op = Op::Loop { len: flat };
                // `composed` is `acc + inner`. With the outer var now
                // enumerating `flat = outer*il + inner`, the same value is
                // `(acc - flat*il) + flat`: `acc - outer*il` is the part of
                // the index that does not advance with the outer trip (a row
                // base, a zero), and `flat` replaces the strip-mined pair.
                let il_c = self.insert_const_idx_before(composed, il);
                let scaled = self.insert_before(composed, Op::Binary { x: outer, y: il_c, bop: BOp::Mul });
                let base = self.insert_before(composed, Op::Binary { x: acc, y: scaled, bop: BOp::Sub });
                let folded_idx = self.insert_before(composed, Op::Binary { x: base, y: outer, bop: BOp::Add });
                self.remap(composed, folded_idx);
                self.remove_op(composed);
                self.remove_op(inner_end);
                self.remove_op(inner);
                folded = true;
                break;
            }
            if !folded {
                break;
            }
        }
        // Gather loops, pad to 32, split unroll-32 FIRST: split's
        // loop arm allows exactly one acc store per body, so it must
        // run before coarsening replicates the body. pad_to pads
        // lengths only on TT (no guards); already-multiples are
        // untouched. Non-reduce loops are a no-op inside (no acc
        // found); anything else keeps its shape for later cuts.
        let mut pad_loops: Vec<(OpId, bool)> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if !matches!(x, Op::Loop { .. }) {
                continue;
            }
            // `split(Loop, Unroll, 32)` leaves `loop*32 + lane`: the loop
            // advances one 32-lane chunk (one tile) per trip, so its length
            // is tiles already.
            let mut tiled = false;
            let end = self.get_last_dim_op(op);
            let mut scan = self.next_op(op);
            while !scan.is_null() && scan != end && !tiled {
                let idx = match self.ops[scan].op {
                    Op::Load { src_idx, .. } => src_idx,
                    Op::Store { dst_idx, .. } => dst_idx,
                    Op::Copy { src_idx, .. } => src_idx,
                    _ => OpId::NULL,
                };
                if !idx.is_null()
                    && let Some(strides) = self.get_strides(idx)
                    && strides.get(&op).is_some_and(|&(_, s)| s == 32)
                    && strides
                        .iter()
                        .any(|(v, &(_, s))| *v != op && s == 1 && matches!(self.ops[*v].op, Op::Unroll { len: 32, .. }))
                {
                    tiled = true;
                }
                scan = self.next_op(scan);
            }
            pad_loops.push((op, tiled));
        }
        for (l, tiled) in pad_loops {
            if tiled {
                continue;
            }
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
        // `tt_reindex` rewrites it for the movers — but only when the reduce
        // loop is gone. With a reduce loop still present the template
        // downstream is `tt_reindex`'s `row2`: one tile per iteration, i.e. the
        // reduce lane plus one row axis. Splitting a group on top of that
        // makes a rank-3 access whose split group is a *grid* axis padded into
        // lanes; `drop3` refuses grids, and the padded lanes read past the
        // tensor (measured on softmax's `max`: `r27` up to 261888 for a 65536
        // element input, and the load bails in both `tt_reindex` and
        // `tt_add_movers`). So the group split gets the same rank budget as the
        // elementwise case: `axes` lanes while a loop remains, `axes + 1` when
        // the loop was fully split into lanes.
        let rank = self.iter_unordered().fold(0, |rank, (_, op)| {
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
        });
        let mut pad_groups: Vec<OpId> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if matches!(x, Op::Range { kind: RangeKind::Group(_), .. }) {
                pad_groups.push(op);
            }
        }
        if pad_groups.len() > 2 {
            todo!("tt_tile: more than two group ranges");
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
        } else if self.iter_unordered().any(|(_, op)| matches!(op, Op::Loop { .. })) {
            axes.saturating_sub(rank)
        } else {
            (axes + 1).saturating_sub(rank)
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

        // The launch grid is the device's Tensix grid: a group axis longer than
        // `max_global_work_dims[axis]` cannot be dispatched, so it is split
        // into an outer group the device can hold and a loop over the rest
        // (`split`'s `outer * factor + inner`, `factor` the smallest divisor
        // of the length that leaves the outer part within the limit). The
        // loop wraps the whole kernel body, so every iteration is one of the
        // group's points.
        let max_grid = self.device_info().max_global_work_dims.clone();
        let mut oversized: Vec<(OpId, i64)> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if let Op::Range { axis, kind: RangeKind::Group(len) } = *x {
                let Some(l) = self.resolve_const(len).and_then(Constant::as_dim) else {
                    continue;
                };
                let Some(&max) = max_grid.get(axis as usize) else {
                    continue;
                };
                if l > max {
                    oversized.push((op, max));
                }
            }
        }
        for (group, max) in oversized {
            let Op::Range { kind: RangeKind::Group(len), .. } = self.ops[group].op else {
                unreachable!("tt_tile: collected op is a group range");
            };
            let Some(l) = self.resolve_const(len).and_then(Constant::as_dim) else {
                continue;
            };
            let min_factor = (l + max - 1) / max;
            let Some(factor) = (min_factor..=l).find(|d| l % d == 0) else {
                todo!("tt_tile: group range {l} has no divisor leaving ≤ {max}")
            };
            let Ok(factor) = u16::try_from(factor) else {
                todo!("tt_tile: group split factor does not fit u16")
            };
            self.split(group, SplitMode::Loop, factor);
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
    /// DRAM store — and rewrites both indices to the physical **tile** pitch.
    /// The rewrite is traffic-only: it does not care whether the fold is a sum
    /// or a maximum, and `tt_dataflow_select`'s pattern table has both, so
    /// gating on `Add` here would leave every other reduce at the logical
    /// pitch and unsuited to the movers. Anything else is left untouched
    /// (multi-reduce, matmul cubes).
    ///
    /// The rewritten index is the truth: `tt_slice` runs later, in a fixed
    /// order, and reads the base and the grid strides from these indices
    /// instead of reconstructing them. A tile is 1024 elements; the reduce's
    /// tile loop advances one tile per iteration, and a grid axis advances by
    /// whatever the input's stride says (whole tiles).
    pub fn tt_reindex(&mut self) {
        if !self.device_info().tenstorrent {
            return;
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
        // The store drains the reduce directly, or an accumulator tile that
        // the loop updates (a reduce over more than one tile: the loop carries
        // the partial result in a `Register` storage).
        let drains_acc = matches!(self.ops[ssrc].op, Op::Load { src, .. }
            if matches!(self.ops[src].op, Op::Storage { scope: MemScope::Register, .. }));
        if ssrc != root && !drains_acc {
            return;
        }
        if !matches!(self.ops[k].op, Op::Unroll { len: 32, .. }) {
            return;
        }
        // The tiled buffer's geometry is a function of the value's shape: the
        // last two dims are the plane the upload tilize laid out (rank-1 is a
        // single column, `tilize_transfer`'s one rule), the rest is the outer
        // batch. That shape is the datum `linearize` keeps for TT: the access
        // alone cannot tell a `[1, N]` row from an `[N]` column, because both
        // are the same flat index.
        let geom = |shape: &[Dim]| -> Option<(Dim, Dim)> {
            let (rows, cols) = match shape {
                [.., r, c] => (*r, *c),
                [c] => (*c, 1),
                [] => (1, 1),
            };
            if rows <= 0 || cols <= 0 {
                return None;
            }
            Some((rows, cols))
        };
        let mul = |k: &mut Kernel, at: OpId, x: OpId, c: Dim| -> OpId {
            let cc = k.insert_const_idx_before(at, c);
            k.insert_before(at, Op::Binary { x, y: cc, bop: BOp::Mul })
        };
        let add =
            |k: &mut Kernel, at: OpId, x: OpId, y: OpId| -> OpId { k.insert_before(at, Op::Binary { x, y, bop: BOp::Add }) };

        // Rewrite each side's flat logical index into the tiled layout:
        //
        //   physical = tile * 1024 + row_in_tile * 32 + col_in_tile
        //   tile     = outer * tiles_per_outer + row_hi * tiles_across + col_hi
        //
        // Every term is classified by its stride against the plane, never by
        // a shape verdict: a multiple of the plane size is an outer slice, a
        // multiple of 32 rows is a tile row, a multiple of the row width is a
        // row, a multiple of 32 columns is a tile column, and the rest is a
        // lane. Whatever does not divide cleanly bails (loudly, in `tt_slice`
        // if the boundary then cannot be placed) rather than guessing.
        for side in [load, store] {
            let (base, idx) = match self.ops[side].op {
                Op::Load { src, src_idx } => (src, src_idx),
                Op::Store { dst, dst_idx, .. } => (dst, dst_idx),
                _ => return,
            };
            if idx.is_null() {
                return;
            }
            let Some((rows, cols)) = geom(&self.shape(base)) else {
                return;
            };
            let (ntr, ntc) = ((rows + 31) / 32, (cols + 31) / 32);
            let plane = rows * cols;
            let Some(strides) = self.get_strides(idx) else {
                return;
            };
            let mut outer_t: Vec<(OpId, Dim)> = Vec::new();
            let mut row_hi_t: Vec<(OpId, Dim)> = Vec::new();
            let mut row_lo_t: Vec<(OpId, Dim)> = Vec::new();
            let mut col_hi_t: Vec<(OpId, Dim)> = Vec::new();
            let mut col_lo_t: Vec<(OpId, Dim)> = Vec::new();
            for (var, (len, stride)) in strides.iter() {
                let (var, s) = (*var, *stride);
                if var == OpId::NULL || s == 0 {
                    continue;
                }
                if s % plane == 0 && s >= plane {
                    outer_t.push((var, s / plane));
                } else if s % (32 * cols) == 0 && s >= 32 * cols {
                    row_hi_t.push((var, s / (32 * cols)));
                } else if s % cols == 0 {
                    // A row: it must stay inside one tile row.
                    let coeff = s / cols;
                    if (*len - 1) * coeff >= 32 {
                        return;
                    }
                    row_lo_t.push((var, coeff));
                } else if s < 32 {
                    if (*len - 1) * s >= 32 {
                        return;
                    }
                    col_lo_t.push((var, s));
                } else if s % 32 == 0 {
                    col_hi_t.push((var, s / 32));
                } else {
                    return;
                }
            }
            // A concrete offset splits the same way; nothing is assumed zero.
            let cst = strides.get(&OpId::NULL).map_or(0, |(_, s)| *s);
            let row_c = (cst / cols) % rows;
            let col_c = cst % cols;
            let tile_c = (cst / plane) * ntr * ntc + (row_c / 32) * ntc + col_c / 32;
            let pos_c = (row_c % 32) * 32 + col_c % 32;

            let mut tile = self.insert_const_idx_before(side, 0i64);
            let mut pos = self.insert_const_idx_before(side, 0i64);
            for &(v, c) in outer_t.iter() {
                let t = mul(self, side, v, c * ntr * ntc);
                tile = add(self, side, tile, t);
            }
            for &(v, c) in row_hi_t.iter() {
                let t = mul(self, side, v, c * ntc);
                tile = add(self, side, tile, t);
            }
            for &(v, c) in col_hi_t.iter() {
                let t = mul(self, side, v, c);
                tile = add(self, side, tile, t);
            }
            if tile_c != 0 {
                let cc = self.insert_const_idx_before(side, tile_c);
                tile = add(self, side, tile, cc);
            }
            for &(v, c) in row_lo_t.iter() {
                let t = mul(self, side, v, c * 32);
                pos = add(self, side, pos, t);
            }
            for &(v, c) in col_lo_t.iter() {
                let t = mul(self, side, v, c);
                pos = add(self, side, pos, t);
            }
            if pos_c != 0 {
                let cc = self.insert_const_idx_before(side, pos_c);
                pos = add(self, side, pos, cc);
            }
            let base_t = mul(self, side, tile, 1024);
            let new_idx = add(self, side, base_t, pos);
            match &mut self.ops[side].op {
                Op::Load { src_idx, .. } => *src_idx = new_idx,
                Op::Store { dst_idx, .. } => *dst_idx = new_idx,
                _ => unreachable!("tt_reindex: side lost its index"),
            }
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
        // No Tensix SFPU has a tensor max: spell `max(a, b)` as
        // `a > b ? a : b`. This runs *after* `tt_tile`, so the reduce's loop
        // and its lane fold are already shaped — `split` needs the `Binary
        // Max` accumulator form to build the fold, and `Reduce { rop: Max }`
        // is a different op and stays untouched.
        let mut maxes: Vec<OpId> = Vec::new();
        for (id, x) in self.iter_unordered() {
            if matches!(x, Op::Binary { bop: BOp::Max, .. }) {
                maxes.push(id);
            }
        }
        for id in maxes {
            let Op::Binary { x, y, .. } = self.ops[id].op else {
                unreachable!("tt_unfold collected a non-Binary op");
            };
            let cond = self.insert_before(id, Op::Binary { x, y, bop: BOp::Cmpgt });
            self.ops[id].op = Op::Select { cond, if_true: x, if_false: y };
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
    /// exponent assembled into the result's bit pattern.
    ///
    /// No `Select` anywhere: the clamps and the three specials of the exact
    /// version (denormal -> +0, overflow -> +inf, NaN) are dropped as an ML
    /// approximation (tt-metal's exp2 clamps with `vec_min_max`; the SFPU has
    /// no `Min`, and a conditional clamp is exactly the wide form being
    /// removed). Removing five selects removes the SSA values that kept the
    /// fused body wide.
    fn tt_exp2(&mut self, at: OpId, x: OpId) -> OpId {
        let konst = |k: &mut Kernel, v: f32| k.insert_before(at, Op::Const(Constant::new(v)));
        let bin = |k: &mut Kernel, x: OpId, y: OpId, bop: BOp| k.insert_before(at, Op::Binary { x, y, bop });
        let magic = konst(self, 12_582_912.0); // 1.5 * 2^23
        // `n = round(x)` by the magic constant, `r = x - n` exactly. No clamp:
        // the magic rounding is exact for |x| < 2^22, and the ML range is far
        // inside it. Out-of-range inputs saturate through the exponent add
        // (wrapping past `f32::MAX` is the one accepted approximation).
        let t = bin(self, x, magic, BOp::Add);
        let nf = bin(self, t, magic, BOp::Sub);
        let r = bin(self, x, nf, BOp::Sub);
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
        self.insert_before(at, Op::Bitcast { x: scale, dtype: DType::F32 })
    }

    /// `log2 x` by tt-metal's `calculate_log_body` (Norbert Juffa's `logf`).
    ///
    /// The reduction is Juffa's: `bits(x) - bits(0.75)` with the mantissa
    /// cleared is the exponent as `k << 23`, and subtracting it from `bits(x)`
    /// leaves the mantissa with the exponent of 1.0, so `x = m 2^k` with `m` in
    /// `[0.75, 1.5)`. `log1p(m - 1)` is one Horner chain over `[-0.25, 0.5)` --
    /// the LLK's shared degree-9 polynomial, with its fp32 `vConstFloatPrgm1/2`
    /// constants -- and `log2 x = k + log2(m)` folds the `ln 2`/`log2 e` round
    /// trip into a single multiply by `log2 e`.
    ///
    /// Worst case on the gate's metric (`|err| / max(1, |log2 x|)`) over
    /// `x in [0.01, 100]` is **1.4e-7**, the accuracy the LLK's fp32 path has.
    /// This replaces the atanh series, which needed a `sqrt 2` normalization
    /// (two `Select`s and an extra live `e`) to reach the same place. The
    /// specials (`±0`/denormal -> `-inf`, negative -> `NaN`, `+inf` -> `+inf`,
    /// NaN stays one) are dropped, the same ML approximation `tt_exp2` takes: a
    /// non-positive or NaN input lands on the polynomial's value.
    fn tt_log2(&mut self, at: OpId, x: OpId) -> OpId {
        let konst = |k: &mut Kernel, v: f32| k.insert_before(at, Op::Const(Constant::new(v)));
        let icons = |k: &mut Kernel, v: i32| k.insert_before(at, Op::Const(Constant::new(v)));
        let bin = |k: &mut Kernel, x: OpId, y: OpId, bop: BOp| k.insert_before(at, Op::Binary { x, y, bop });
        let bits = self.insert_before(at, Op::Bitcast { x, dtype: DType::I32 });
        // `e = setman(bits - bits(0.75), 0)`: the exponent as `k << 23`.
        let c75 = icons(self, 0x3f40_0000);
        let e = bin(self, bits, c75, BOp::Sub);
        let emask = icons(self, 0xff80_0000u32 as i32);
        let e = bin(self, e, emask, BOp::BitAnd);
        // `m = bits - e` is the mantissa with the exponent of 1.0.
        let m = bin(self, bits, e, BOp::Sub);
        let m = self.insert_before(at, Op::Bitcast { x: m, dtype: DType::F32 });
        let one = konst(self, 1.0);
        let m = bin(self, m, one, BOp::Sub);
        let s = bin(self, m, m, BOp::Mul);
        // Juffa's `log1p` chain, with the LLK's fp32 constant set.
        let mut r = konst(self, f32::from_bits(0xbd49_6000));
        for c in [
            0x3ddc_2000,
            0xbe06_2000,
            0x3e13_a000,
            0xbe2a_8000,
            0x3e4c_c000,
            0xbe80_000d,
            0x3eaa_aab9,
            0xbf00_0000,
        ] {
            let cc = konst(self, f32::from_bits(c));
            let p = bin(self, r, m, BOp::Mul);
            r = bin(self, p, cc, BOp::Add);
        }
        let r = bin(self, r, s, BOp::Mul);
        let r = bin(self, r, m, BOp::Add);
        // `e` is `k << 23` as an integer, so `e * 2^-23` is exactly `k`.
        let ef = self.insert_before(at, Op::Cast { x: e, dtype: DType::F32 });
        let two_m23 = konst(self, f32::from_bits(0x3400_0000));
        let kf = bin(self, ef, two_m23, BOp::Mul);
        let log2e = konst(self, std::f32::consts::LOG2_E);
        let rl = bin(self, r, log2e, BOp::Mul);
        bin(self, kf, rl, BOp::Add)
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

    /// Upper bound of an affine index (the constant term plus every
    /// variable's `(extent - 1) * stride`), or `None` when the index is not
    /// affine. Used only to tell a logical index from the tile-datum form
    /// `tt_reindex` writes: a logical index is always `< outer * rows * cols`,
    /// because that is the buffer it addresses.
    fn tt_index_max(&self, idx: OpId) -> Option<i64> {
        let strides = self.get_strides(idx)?;
        let mut max = strides.get(&OpId::NULL).map_or(0, |(_, s)| *s);
        for (var, (len, stride)) in strides.iter() {
            if *var == OpId::NULL || *stride <= 0 {
                continue;
            }
            let span = len.checked_sub(1)?.checked_mul(*stride)?;
            max = max.checked_add(span)?;
        }
        Some(max)
    }

    /// Clone an index cone with every loop and lane variable bound to zero.
    ///
    /// The block form of a boundary copy publishes the whole block with one
    /// entry, so the per-iteration term is not in scope at its anchor; the
    /// lane axes are the access's within-tile offsets and the copy is
    /// tile-granular. A `Local` range is a lane by another name. Group ids are
    /// left alone: they are the grid, and the canonical transform reads them
    /// as logical coordinates.
    fn tt_zero_iteration_vars(
        &mut self,
        idx: OpId,
        before: OpId,
        keep: OpId,
        loops: bool,
        groups: bool,
        lanes: bool,
        memo: &mut Map<OpId, OpId>,
    ) -> Option<OpId> {
        if let Some(&done) = memo.get(&idx) {
            return Some(done);
        }
        let op = self.ops[idx].op.clone();
        let zero = match &op {
            _ if idx == keep && !keep.is_null() => false,
            Op::Loop { .. } => loops,
            Op::Range { kind: RangeKind::Local(_), .. } => true,
            Op::Range { kind: RangeKind::Group(_), .. } => groups,
            Op::Unroll { .. } => lanes,
            _ => false,
        };
        if zero {
            let zero = self.insert_const_idx_before(before, 0i64);
            memo.insert(idx, zero);
            return Some(zero);
        }
        let out = match op {
            Op::Const(_) | Op::Range { .. } | Op::Unroll { .. } | Op::Loop { .. } | Op::Param { .. } => idx,
            Op::Binary { x, y, bop } => {
                let x = self.tt_zero_iteration_vars(x, before, keep, loops, groups, lanes, memo)?;
                let y = self.tt_zero_iteration_vars(y, before, keep, loops, groups, lanes, memo)?;
                self.insert_before(before, Op::Binary { x, y, bop })
            }
            Op::Unary { x, uop } => {
                let x = self.tt_zero_iteration_vars(x, before, keep, loops, groups, lanes, memo)?;
                self.insert_before(before, Op::Unary { x, uop })
            }
            Op::Cast { x, dtype } => {
                let x = self.tt_zero_iteration_vars(x, before, keep, loops, groups, lanes, memo)?;
                self.insert_before(before, Op::Cast { x, dtype })
            }
            _ => return None,
        };
        memo.insert(idx, out);
        Some(out)
    }

    /// Clone an index cone with the loop variable `from` replaced by `to`
    /// (another loop's induction variable). Every other leaf is shared: it
    /// dominates `before` exactly like the original cone does.
    fn tt_replace_loop(&mut self, idx: OpId, from: OpId, to: OpId, before: OpId, memo: &mut Map<OpId, OpId>) -> Option<OpId> {
        if let Some(&done) = memo.get(&idx) {
            return Some(done);
        }
        let op = self.ops[idx].op.clone();
        let out = match op {
            Op::Loop { .. } if idx == from => to,
            Op::Loop { .. } => return None,
            Op::Const(_) | Op::Range { .. } | Op::Param { .. } | Op::Unroll { .. } => idx,
            Op::Binary { x, y, bop } => {
                let x = self.tt_replace_loop(x, from, to, before, memo)?;
                let y = self.tt_replace_loop(y, from, to, before, memo)?;
                self.insert_before(before, Op::Binary { x, y, bop })
            }
            Op::Unary { x, uop } => {
                let x = self.tt_replace_loop(x, from, to, before, memo)?;
                self.insert_before(before, Op::Unary { x, uop })
            }
            Op::Cast { x, dtype } => {
                let x = self.tt_replace_loop(x, from, to, before, memo)?;
                self.insert_before(before, Op::Cast { x, dtype })
            }
            _ => return None,
        };
        memo.insert(idx, out);
        Some(out)
    }

    /// Canonicalize one boundary access into the two forms a mover copy needs.
    ///
    /// `dram` is the parameter's **physical address, in bytes**: the access's
    /// own index with the tilize inverse *appended* — outer slice, tile row,
    /// tile column, then face and position inside the face — plus the tile
    /// image's pitch and 16-byte header. The index is never rewritten into a
    /// tile ordinal and never divided by an access footprint; a `/` or `%`
    /// the access already carries stays in the expression.
    ///
    /// `l1` is the tile's slot in the compute-side CB,
    /// `tile * 1024 + row_in_tile * 32 + col_in_tile` — the coordinates the
    /// unpack/pack selectors and the SFPU already use, so
    /// [`Kernel::tt_dataflow_select`] and compute need no layout knowledge.
    ///
    /// Geometry comes from the parameter's shape (`Op::Param.shape`, kept for
    /// TT by `linearize`): the last two dims are the plane the upload tilize
    /// laid out, the rest is the outer batch; a `[N]` parameter is `N x 1`.
    /// Two index conventions are accepted: the buffer's flat logical index,
    /// and the tile-datum form `tt_reindex` writes (`tile * 1024 + pos`).
    fn tt_canonical_tile(&mut self, base: OpId, idx: OpId, at: OpId) -> Option<(OpId, OpId)> {
        fn cst(k: &mut Kernel, at: OpId, v: i64) -> OpId {
            k.insert_const_idx_before(at, v)
        }
        fn bin(k: &mut Kernel, at: OpId, x: OpId, y: OpId, op: BOp) -> OpId {
            k.insert_before(at, Op::Binary { x, y, bop: op })
        }
        fn mul(k: &mut Kernel, at: OpId, x: OpId, c: i64) -> OpId {
            let cc = cst(k, at, c);
            bin(k, at, x, cc, BOp::Mul)
        }
        fn div(k: &mut Kernel, at: OpId, x: OpId, c: i64) -> OpId {
            let cc = cst(k, at, c);
            bin(k, at, x, cc, BOp::Div)
        }
        fn rem(k: &mut Kernel, at: OpId, x: OpId, c: i64) -> OpId {
            let cc = cst(k, at, c);
            bin(k, at, x, cc, BOp::Mod)
        }

        let shape = self.shape(base);
        let rank = shape.len();
        let (rows, cols) = match shape.as_slice() {
            [.., r, c] => (*r, *c),
            [c] => (*c, 1),
            [] => (1, 1),
        };
        if rows <= 0 || cols <= 0 {
            return None;
        }
        let outer: i64 = if rank >= 2 { shape[..rank - 2].iter().product() } else { 1 };
        if outer <= 0 {
            return None;
        }
        let el = i64::from(self.dtype(base).bit_size()) / 8;
        if el <= 0 {
            return None;
        }
        let (ntr, ntc) = ((rows + 31) / 32, (cols + 31) / 32);
        // `tt_reindex` rewrote this index into tile-datum units. Everything
        // else is the buffer's flat logical index. Three tells, any of which
        // settles it: the rewritten shape (`tile*1024+pos` over a grid *and* a
        // loop), an index that runs past the buffer (padding), or a loop that
        // steps exactly one tile image — `tt_reindex`'s own template, which a
        // logical index only reproduces by accident.
        let tile_units = self.tt_index_is_tile_units(idx)
            || self.tt_index_max(idx).is_some_and(|m| m >= outer * rows * cols)
            || self.get_strides(idx).is_some_and(|s| {
                s.iter().any(|(var, &(_, stride))| {
                    *var != OpId::NULL && stride != 0 && stride % 1024 == 0 && matches!(self.ops[*var].op, Op::Loop { .. })
                })
            });

        let (tile, row_lo, col_lo) = if tile_units {
            let tile = div(self, at, idx, 1024);
            let pos = rem(self, at, idx, 1024);
            let row = div(self, at, pos, 32);
            let col = rem(self, at, pos, 32);
            (tile, row, col)
        } else {
            let col = rem(self, at, idx, cols);
            let flat = div(self, at, idx, cols);
            let row = rem(self, at, flat, rows);
            let outer_slice = div(self, at, flat, rows);
            let tile_row = div(self, at, row, 32);
            let tile_col = div(self, at, col, 32);
            let tile = mul(self, at, outer_slice, ntr * ntc);
            let tr_tile = mul(self, at, tile_row, ntc);
            let tile = bin(self, at, tile, tr_tile, BOp::Add);
            let tile = bin(self, at, tile, tile_col, BOp::Add);
            let row_lo = rem(self, at, row, 32);
            let col_lo = rem(self, at, col, 32);
            (tile, row_lo, col_lo)
        };
        // Canonical tile slot in the CB: row-major inside the tile, which is
        // the coordinate the unpack/pack selectors name.
        let l1_tile = mul(self, at, tile, 1024);
        let l1_row = mul(self, at, row_lo, 32);
        let l1 = bin(self, at, l1_tile, l1_row, BOp::Add);
        let l1 = bin(self, at, l1, col_lo, BOp::Add);
        // The tile image: four 16x16 faces in face order, then the datum.
        let face_row = div(self, at, row_lo, 16);
        let face_col = div(self, at, col_lo, 16);
        let face_row2 = mul(self, at, face_row, 2);
        let face = bin(self, at, face_row2, face_col, BOp::Add);
        let within_row = rem(self, at, row_lo, 16);
        let within_col = rem(self, at, col_lo, 16);
        let within_row16 = mul(self, at, within_row, 16);
        let within = bin(self, at, within_row16, within_col, BOp::Add);
        let face256 = mul(self, at, face, 256);
        let datum = bin(self, at, face256, within, BOp::Add);
        let slot = mul(self, at, tile, tt_isa::dm::TILE_SLOT as i64);
        let hdr = cst(self, at, tt_isa::dm::TILE_DATA as i64);
        let slot = bin(self, at, slot, hdr, BOp::Add);
        let datum_el = mul(self, at, datum, el);
        let dram = bin(self, at, slot, datum_el, BOp::Add);
        Some((dram, l1))
    }

    /// The reader/writer traffic pass. `debug()` at entry and exit; the
    /// debugging gate now sits at the start of `tt_dataflow_select`, where the
    /// CB syncs must be visible in the IR.
    pub fn tt_add_movers(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        self.debug();
        self.tt_add_movers_body();
        self.debug();
    }

    /// Materialise the circular-buffer plan as ordinary SSA before the phase
    /// boundary.
    ///
    /// One `Op::Copy` per DRAM parameter publishes its block into a
    /// `MemScope::Circular` storage (slot 0, the whole `Π trips × tile`
    /// range), every `Op::Load` names the block slot it consumes, every output
    /// store targets a writer CB drained by a `Copy` at the tail, and the
    /// `EndReader`/`EndCompute` markers are placed from that traffic. After
    /// this pass the plan is ordinary IR: generic passes and `cost` see it, and
    /// [`Kernel::tt_dataflow_select`] only encodes it.
    ///
    /// The DRAM side of every copy is [`Kernel::tt_canonical_tile`]'s
    /// `dram` — the access's own index with the tilize inverse appended — so
    /// it may be any expression the access already carries, `/` and `%`
    /// included. The L1 side is the tile's slot in the CB. Everything after
    /// this pass sees canonical tiles only.
    pub fn tt_add_movers_body(&mut self) {
        // CB Storage definitions live in the prologue (past leading
        // params/storages, before everything else).
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
        // Traffic that already exists as SSA: its boundary index is normalised
        // below, but nothing else about it is decided here.
        let mut existing: Vec<(OpId, bool)> = Vec::new();
        for (id, op) in self.iter_unordered() {
            let Op::Copy { src, dst, .. } = *op else { continue };
            if !matches!(self.ops[src].op, Op::Param { .. } | Op::Storage { .. })
                || !matches!(self.ops[dst].op, Op::Param { .. } | Op::Storage { .. })
            {
                continue;
            }
            let src_dram = matches!(self.ops[src].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. });
            let dst_dram = matches!(self.ops[dst].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. });
            if src_dram && self.is_circular(dst) {
                existing.push((id, true));
            } else if self.is_circular(src) && dst_dram {
                existing.push((id, false));
            }
        }
        // Read-only scan: nothing is mutated until every load and store is
        // known to be expressible.
        // (load, DRAM param, index, tiles in the block, innermost enclosing loop)
        let mut loads: Vec<(OpId, OpId, OpId, i64, OpId)> = Vec::new();
        // (store, DRAM param, index, value)
        let mut stores: Vec<(OpId, OpId, OpId, OpId)> = Vec::new();
        let mut loop_trips: Vec<i64> = Vec::new();
        let mut loop_ops: Vec<OpId> = Vec::new();
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() {
                break;
            }
            match self.ops[id].op {
                Op::Loop { len } => {
                    let Some(trip) = self.resolve_const(len).and_then(Constant::as_dim) else {
                        return;
                    };
                    if trip <= 0 {
                        return;
                    }
                    loop_trips.push(trip);
                    loop_ops.push(id);
                }
                Op::EndLoop => {
                    loop_trips.pop();
                    loop_ops.pop();
                }
                Op::Load { src: base, src_idx: off }
                    if matches!(self.ops[base].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. }) =>
                {
                    // A `[32, 32]` tile, a `[32]` vector — one tile row — or
                    // a scalar (one lane; a pure copy kernel has nothing
                    // else). The element-to-tile unit below carries the
                    // difference.
                    let shape = self.shape(id);
                    if shape != vec![32, 32] && shape != vec![32] && !shape.is_empty() {
                        return;
                    }
                    // An affine index cone only; a loop variable is a scalar
                    // leaf that becomes the block slot.
                    let mut stack = vec![off];
                    for _ in 0..250_000 {
                        let Some(n) = stack.pop() else { break };
                        match self.ops[n].op {
                            Op::Const(..) | Op::Range { .. } | Op::Param { .. } | Op::Unroll { .. } | Op::Loop { .. } => {}
                            Op::Binary { x, y, .. } => {
                                stack.push(x);
                                stack.push(y);
                            }
                            Op::Unary { x, .. } => stack.push(x),
                            Op::Cast { x, .. } => stack.push(x),
                            _ => return,
                        }
                    }
                    let mut tiles = 1i64;
                    for trip in loop_trips.iter() {
                        let Some(t) = tiles.checked_mul(*trip) else { return };
                        tiles = t;
                    }
                    let hoist = loop_ops.last().copied().unwrap_or(OpId::NULL);
                    loads.push((id, base, off, tiles, hoist));
                }
                Op::Store { dst: base, dst_idx: off, src }
                    if matches!(self.ops[base].op, Op::Param { kind: ParamKind::GlobalMut, .. }) =>
                {
                    // Tiled stores, row stores (a reduce row is 32 elements)
                    // and scalar stores (a whole-tile reduce's first lane).
                    match self.shape(id).as_slice() {
                        [32, 32] | [32] | [1024] | [] => {}
                        _ => return,
                    }
                    stores.push((id, base, off, src));
                }
                _ => {}
            }
            id = self.ops[id].next;
        }
        // Reader: publish each block — or, for a load inside a loop, one tile
        // per iteration from its own reader loop — then close the section
        // before the first compute op. A reader section cannot share a loop
        // with compute, so streaming needs the separate loop; when the loads
        // do not all sit in the same single loop, the block is hoisted as
        // before.
        if !loads.is_empty() {
            let anchor = match loads[0] {
                (_, _, _, _, hoist) if !hoist.is_null() => hoist,
                (load, ..) => load,
            };
            let streaming =
                matches!(self.ops[anchor].op, Op::Loop { .. }) && loads.iter().all(|(_, _, _, _, hoist)| *hoist == anchor);
            let mut cb_of: Map<OpId, OpId> = Map::default();
            if streaming {
                let Op::Loop { len } = self.ops[anchor].op else {
                    unreachable!("tt_add_movers: the streaming anchor is a loop")
                };
                // The reader's own induction variable: the compute loop keeps
                // its slot arithmetic, and each entry advances one tile.
                let reader_loop = self.insert_before(anchor, Op::Loop { len });
                for (load, base, off, _tiles, _) in loads.iter().copied() {
                    let Op::Param { dtype, .. } = self.ops[base].op else { return };
                    // The ring is one tile deep: one page per iteration, like
                    // `cb_reserve_back(cb, 1)` in the tt-metal example.
                    let len = 1024;
                    let storage = Op::Storage { dtype, scope: MemScope::Circular, len, id: 0 };
                    let cb = if prologue.is_null() {
                        self.push_back(storage)
                    } else {
                        self.insert_before(prologue, storage)
                    };
                    // The access's own index with every other loop and lane
                    // bound to zero and the compute loop replaced by this
                    // reader's counter: the parameter's physical address for
                    // one tile of *this* iteration. Nothing is added per
                    // iteration by hand — the transform reads the counter, so
                    // a permuted index enumerates what it actually names.
                    let mut memo: Map<OpId, OpId> = Map::default();
                    let Some(sub) = self.tt_replace_loop(off, anchor, reader_loop, anchor, &mut memo) else {
                        todo!("tt_add_movers: reader index {off} is not loop arithmetic")
                    };
                    let mut zero: Map<OpId, OpId> = Map::default();
                    let Some(flat) = self.tt_zero_iteration_vars(sub, anchor, reader_loop, true, false, true, &mut zero) else {
                        todo!("tt_add_movers: reader index {off} is not iteration arithmetic")
                    };
                    let Some((dram, _l1)) = self.tt_canonical_tile(base, flat, anchor) else {
                        todo!("tt_add_movers: reader index {off} does not canonicalize")
                    };
                    // One page per iteration: the ring slot is 0.
                    let slot = self.insert_const_idx_before(anchor, 0i64);
                    self.insert_before(anchor, Op::Copy { src: base, src_idx: dram, dst: cb, dst_idx: slot });
                    cb_of.insert(load, cb);
                }
                self.insert_before(anchor, Op::EndLoop);
            } else {
                let slot_zero = self.insert_const_idx_before(anchor, 0i64);
                let mut zero: Map<OpId, OpId> = Map::default();
                for (load, base, off, tiles, _) in loads.iter().copied() {
                    let Op::Param { dtype, .. } = self.ops[base].op else { return };
                    let Some(len) = tiles.checked_mul(1024) else { return };
                    let storage = Op::Storage { dtype, scope: MemScope::Circular, len, id: 0 };
                    let cb = if prologue.is_null() {
                        self.push_back(storage)
                    } else {
                        self.insert_before(prologue, storage)
                    };
                    let Some(flat) = self.tt_zero_iteration_vars(off, anchor, OpId::NULL, true, false, true, &mut zero) else {
                        todo!("tt_add_movers: block reader index {off} is not iteration arithmetic")
                    };
                    let Some((dram, _l1)) = self.tt_canonical_tile(base, flat, anchor) else {
                        todo!("tt_add_movers: block reader index {off} does not canonicalize")
                    };
                    self.insert_before(anchor, Op::Copy { src: base, src_idx: dram, dst: cb, dst_idx: slot_zero });
                    cb_of.insert(load, cb);
                }
            }
            self.insert_before(anchor, Op::TT(TTOp::EndReader));
            // The value reads from its block now, at the **canonical tile
            // index**: the within-tile row/column the unpack selects, with the
            // grid bound to zero (a core's block starts at tile 0 of its CB)
            // and the loop kept (each iteration has its own slot). The matmul
            // roles read this index to tell A from B, and their tell is the
            // lane stride, which the transform preserves.
            for (load, base, off, _, _) in loads.iter().copied() {
                let cb = cb_of[&load];
                let mut zero: Map<OpId, OpId> = Map::default();
                let Some(flat) = self.tt_zero_iteration_vars(off, load, OpId::NULL, true, true, false, &mut zero) else {
                    todo!("tt_add_movers: compute load index {off} is not tile arithmetic")
                };
                let Some((_dram, l1)) = self.tt_canonical_tile(base, flat, load) else {
                    todo!("tt_add_movers: compute load index {off} does not canonicalize")
                };
                self.ops[load].op = Op::Load { src: cb, src_idx: l1 };
            }
        }
        // Writer: the store targets a writer CB and the drain to DRAM is the
        // section's traffic. A store-only kernel gets an empty reader section.
        if !stores.is_empty() {
            if loads.is_empty() {
                let first = stores[0].0;
                self.insert_before(first, Op::TT(TTOp::EndReader));
            }
            self.push_back(Op::TT(TTOp::EndCompute));
            // Stores that share one loop are a block: each iteration packs into
            // its own tile slot of the writer CB, and the drain is that same
            // loop at the tail. (Anything else keeps the single-slot form.)
            let store_loop = self.tt_enclosing_loop(stores[0].0).unwrap_or(OpId::NULL);
            let looped =
                !store_loop.is_null() && stores.iter().all(|(store, ..)| self.tt_enclosing_loop(*store) == Some(store_loop));
            for (store, base, off, value) in stores.iter().copied() {
                // One page per iteration, as the reader's ring.
                let len = 1024;
                let storage = Op::Storage { dtype: DType::F32, scope: MemScope::Circular, len, id: 0 };
                let cb = if prologue.is_null() {
                    self.push_back(storage)
                } else {
                    self.insert_before(prologue, storage)
                };
                if looped {
                    // The compute side packs into the iteration's own slot; the
                    // pack's destination offset is set where the pack is made
                    // (`tt_dataflow_select`). The store's access index is the
                    // canonical tile index within the block: the grid bound to
                    // zero, the loop kept.
                    let mut zero: Map<OpId, OpId> = Map::default();
                    let Some(flat) = self.tt_zero_iteration_vars(off, store, OpId::NULL, true, true, false, &mut zero) else {
                        todo!("tt_add_movers: looped store index {off} is not tile arithmetic")
                    };
                    let Some((_dram, l1)) = self.tt_canonical_tile(base, flat, store) else {
                        todo!("tt_add_movers: looped store index {off} does not canonicalize")
                    };
                    self.ops[store].op = Op::Store { dst: cb, dst_idx: l1, src: value };
                    // ...and the drain repeats the loop, at the tail. Its body
                    // is built before a temporary anchor so every emitted op
                    // lands after the loop header.
                    let Op::Loop { len: loop_len } = self.ops[store_loop].op else {
                        return;
                    };
                    let drain = self.push_back(Op::Loop { len: loop_len });
                    let anchor = self.push_back(Op::Const(Constant::idx(0i64)));
                    let mut memo: Map<OpId, OpId> = Map::default();
                    let Some(sub) = self.tt_replace_loop(off, store_loop, drain, anchor, &mut memo) else {
                        todo!("tt_add_movers: drain index {off} is not loop arithmetic")
                    };
                    let mut zero: Map<OpId, OpId> = Map::default();
                    let Some(flat) = self.tt_zero_iteration_vars(sub, anchor, drain, true, false, true, &mut zero) else {
                        todo!("tt_add_movers: drain index {off} is not iteration arithmetic")
                    };
                    let Some((dram, _l1)) = self.tt_canonical_tile(base, flat, anchor) else {
                        todo!("tt_add_movers: drain index {off} does not canonicalize")
                    };
                    // One page per iteration: the ring slot is 0.
                    let slot = self.insert_const_idx_before(anchor, 0i64);
                    self.push_back(Op::Copy { src: cb, src_idx: slot, dst: base, dst_idx: dram });
                    self.remove_op(anchor);
                    self.push_back(Op::EndLoop);
                } else {
                    let mut zero: Map<OpId, OpId> = Map::default();
                    let Some(flat) = self.tt_zero_iteration_vars(off, self.tail, OpId::NULL, true, false, true, &mut zero) else {
                        todo!("tt_add_movers: drain index {off} is not iteration arithmetic")
                    };
                    let Some((dram, _l1)) = self.tt_canonical_tile(base, flat, self.tail) else {
                        todo!("tt_add_movers: drain index {off} does not canonicalize")
                    };
                    // The compute index is the canonical tile index within the
                    // block, with the grid bound to zero.
                    let mut czero: Map<OpId, OpId> = Map::default();
                    let Some(cflat) = self.tt_zero_iteration_vars(off, store, OpId::NULL, false, true, false, &mut czero) else {
                        todo!("tt_add_movers: store index {off} is not tile arithmetic")
                    };
                    let Some((_d, l1)) = self.tt_canonical_tile(base, cflat, store) else {
                        todo!("tt_add_movers: store index {off} does not canonicalize")
                    };
                    self.ops[store].op = Op::Store { dst: cb, dst_idx: l1, src: value };
                    let slot_zero = self.push_back(Op::Const(Constant::idx(0i64)));
                    self.push_back(Op::Copy { src: cb, src_idx: slot_zero, dst: base, dst_idx: dram });
                }
            }
        }
        // Normalise the boundary indices of traffic that was already SSA: the
        // encoder slices a lane-free base. A loop variable is kept (`false`), so
        // a per-iteration copy reaches `tt_slice` and fails loudly instead of
        // being silently dropped.
        for (copy, reader) in existing {
            let idx = match self.ops[copy].op {
                Op::Copy { src_idx, dst_idx, .. } => {
                    if reader {
                        src_idx
                    } else {
                        dst_idx
                    }
                }
                _ => unreachable!("tt_add_movers: a collected op is not a copy"),
            };
            let Some(stripped) = self.strip_unrolls(idx, copy, false) else {
                todo!("tt_add_movers: existing copy {copy:?} has a non-affine boundary index {idx}")
            };
            match &mut self.ops[copy].op {
                Op::Copy { src_idx, dst_idx, .. } => {
                    if reader {
                        *src_idx = stripped;
                    } else {
                        *dst_idx = stripped;
                    }
                }
                _ => unreachable!("tt_add_movers: a collected op is not a copy"),
            }
        }
        // The transform leaves constant subexpressions behind (a zeroed loop
        // makes most of a tile's arithmetic constant) and the copies share
        // cones. The generic pipeline is still legal here — the phase boundary
        // has not been crossed — so clean up before the IR is handed on.
        self.default_epilogue();
    }

    /// Place the four CB syncs on the reader/writer traffic.
    ///
    /// A publisher is a `Copy` into a `Circular` storage (the reader's
    /// DRAM→CB move) or a `Store` into one (the pack). Both fill whole pages,
    /// and a run of publishes that together fills whole pages of one CB is
    /// bracketed by one `ReserveBack` before the run and one `PushBack` after
    /// it, `n` = the pages the run fills. The run is scoped to one loop body:
    /// at depth 1 the sync must execute once per iteration (tt-metal's
    /// `for i { reserve; read; push }`), never be hoisted across the loop.
    ///
    /// A consumer is a `Load` from a CB (the unpack) or a `Copy` out of one
    /// (the writer's drain); each gets a `WaitFront` immediately before it.
    /// `n` is in pages throughout, as in `cb_reserve_back`/`cb_push_back`.
    pub fn tt_sync_cbs(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Frame {
            Trip(u64),
            Symbolic,
            Conditional,
        }
        // Datums moved by a publish op. A `Copy` into a CB moves whole pages:
        // one per iteration when it sits in a loop, the CB's whole page count
        // when it is straight-line (`tt_dataflow_select` computes the same
        // transfer: `1024 * el` in a loop, `len * el` otherwise). The index
        // shape is the address expression, not the transfer.
        let publish_datums = |k: &Kernel, id: OpId, in_loop: bool| -> u64 {
            match k.at(id) {
                Op::Store { src, dst, .. } if k.is_circular(*dst) => {
                    let datums: u64 = k.shape(*src).iter().map(|d| *d as u64).product();
                    datums.max(1)
                }
                Op::Copy { src, dst, .. } if k.is_circular(*dst) && !k.is_circular(*src) => {
                    if in_loop {
                        1024
                    } else {
                        match k.at(*dst) {
                            Op::Storage { len, .. } => u64::try_from(*len)
                                .unwrap_or_else(|_| panic!("tt_sync_cbs: CB {dst:?} has negative length {len}")),
                            other => panic!("tt_sync_cbs: copy {id:?} targets {other:?}, not a storage"),
                        }
                    }
                }
                _ => 0,
            }
        };
        // One run per (section, innermost loop, CB).
        let mut open: Map<OpId, (OpId, OpId, u64)> = Map::default();
        let mut tallies: Vec<(OpId, OpId, OpId, u8)> = Vec::new();
        let mut stack: Vec<(OpId, Frame)> = Vec::new();
        let mut section = 0u8;
        let mut scope = (0u8, OpId::NULL);
        // Close every open run of the previous scope.
        let flush = |open: &mut Map<OpId, (OpId, OpId, u64)>, tallies: &mut Vec<(OpId, OpId, OpId, u8)>| {
            for (cb, (first, last, datums)) in open.drain() {
                if datums == 0 || datums % 1024 != 0 {
                    panic!("tt_sync_cbs: CB{cb:?} publish run fills {datums} datums, not whole pages");
                }
                let pages = datums / 1024;
                let Ok(n) = u8::try_from(pages) else {
                    panic!("tt_sync_cbs: CB{cb:?} publish run fills {pages} pages, more than u8")
                };
                tallies.push((first, last, cb, n));
            }
        };
        let mut scan = self.head;
        for _ in 0..250_000 {
            if scan.is_null() {
                break;
            }
            match self.ops[scan].op {
                Op::TT(TTOp::EndReader) => {
                    flush(&mut open, &mut tallies);
                    section = 1;
                    scope = (section, stack.last().map_or(OpId::NULL, |(op, _)| *op));
                }
                Op::TT(TTOp::EndCompute) => {
                    flush(&mut open, &mut tallies);
                    section = 2;
                    scope = (section, stack.last().map_or(OpId::NULL, |(op, _)| *op));
                }
                Op::Loop { len } => {
                    flush(&mut open, &mut tallies);
                    let frame = if let Op::Const(c) = self.at(len) {
                        match c.as_dim() {
                            Some(t) if t > 0 => Frame::Trip(t as u64),
                            Some(t) => panic!("tt_sync_cbs: loop {scan:?} has non-positive trip count {t}"),
                            None => Frame::Symbolic,
                        }
                    } else if self.dtype(len) == DType::Bool {
                        Frame::Conditional
                    } else {
                        Frame::Symbolic
                    };
                    stack.push((scan, frame));
                    scope = (section, scan);
                }
                Op::EndLoop => {
                    flush(&mut open, &mut tallies);
                    stack.pop().unwrap_or_else(|| panic!("tt_sync_cbs: EndLoop without Loop at {scan:?}"));
                    scope = (section, stack.last().map_or(OpId::NULL, |(op, _)| *op));
                }
                _ => {
                    let in_loop = !stack.is_empty();
                    let datums = publish_datums(self, scan, in_loop);
                    if datums == 0 {
                        scan = self.next_op(scan);
                        continue;
                    }
                    // The run's per-body count multiplies the trips of the
                    // loops *outside* the innermost one; the innermost loop's
                    // own trips are the body executions the run lives in.
                    let mut mult = 1u64;
                    for (_, frame) in stack.iter().take(stack.len().saturating_sub(1)) {
                        match frame {
                            Frame::Trip(t) => {
                                mult = mult.checked_mul(*t).unwrap_or_else(|| panic!("tt_sync_cbs: trip product overflows"))
                            }
                            Frame::Symbolic => {
                                panic!("tt_sync_cbs: CB publish {scan:?} under a symbolic enclosing loop")
                            }
                            Frame::Conditional => {}
                        }
                    }
                    let datums = mult.checked_mul(datums).unwrap_or_else(|| panic!("tt_sync_cbs: page count overflows"));
                    let cb = match self.ops[scan].op {
                        Op::Store { dst, .. } | Op::Copy { dst, .. } => dst,
                        _ => unreachable!("tt_sync_cbs: a publish is a Store or Copy"),
                    };
                    if scope != (section, stack.last().map_or(OpId::NULL, |(op, _)| *op)) {
                        flush(&mut open, &mut tallies);
                        scope = (section, stack.last().map_or(OpId::NULL, |(op, _)| *op));
                    }
                    let entry = open.entry(cb).or_insert((scan, scan, 0));
                    entry.1 = scan;
                    entry.2 =
                        entry.2.checked_add(datums).unwrap_or_else(|| panic!("tt_sync_cbs: CB{cb:?} publish total overflows"));
                }
            }
            scan = self.next_op(scan);
        }
        flush(&mut open, &mut tallies);
        for (first, last, cb, n) in tallies {
            self.insert_before(first, Op::TT(TTOp::ReserveBack { cb, n }));
            self.insert_after(last, Op::TT(TTOp::PushBack { cb, n }));
        }
        // Consumer waits.
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match self.ops[op_id].op {
                Op::Copy { src, dst, .. } => {
                    let src_cb = self.is_circular(src);
                    let dst_cb = self.is_circular(dst);
                    if src_cb && dst_cb {
                        panic!("tt_sync_cbs: CB-to-CB copy {op_id:?} has no page plan");
                    }
                    if src_cb {
                        self.insert_before(op_id, Op::TT(TTOp::WaitFront { cb: src, n: 1 }));
                    }
                    debug_assert!(src_cb || dst_cb, "tt_sync_cbs: copy {op_id:?} touches no CB");
                }
                Op::TT(TTOp::Copy { src, dst, .. }) => {
                    let src_cb = self.is_circular(src);
                    let dst_cb = self.is_circular(dst);
                    if src_cb && dst_cb {
                        panic!("tt_sync_cbs: CB-to-CB copy {op_id:?} has no page plan");
                    }
                    if src_cb {
                        self.insert_before(op_id, Op::TT(TTOp::WaitFront { cb: src, n: 1 }));
                    }
                }
                Op::Load { src, .. } if self.is_circular(src) => {
                    self.insert_before(op_id, Op::TT(TTOp::WaitFront { cb: src, n: 1 }));
                }
                _ => {}
            }
            op_id = next;
        }
        self.verify();
    }

    /// Merge adjacent `PushBack`s on one CB: two consecutive publishes of the
    /// same buffer are one publish of their sum.
    pub fn tt_dedup_pushes(&mut self) {
        let mut scan = self.head;
        while !scan.is_null() {
            let next = self.next_op(scan);
            if let Op::TT(TTOp::PushBack { cb, n }) = self.ops[scan].op {
                let prev = self.prev_op(scan);
                if !prev.is_null()
                    && let Op::TT(TTOp::PushBack { cb: pcb, n: pn }) = self.ops[prev].op
                    && pcb == cb
                {
                    let Some(sum) = pn.checked_add(n) else {
                        panic!("tt_dedup_pushes: CB{cb:?} push count overflows u8")
                    };
                    self.ops[prev].op = Op::TT(TTOp::PushBack { cb, n: sum });
                    self.remove_op(scan);
                }
            }
            scan = next;
        }
    }

    /// Place `PopFront` after the last read of each consumed CB page: after
    /// the last same-section use of a `Load` from a CB, after the `Load`
    /// itself when it has no user, and after a `Copy` whose source is a CB
    /// (the writer's drain, one page per copy).
    ///
    /// The CB address a kernel computes is relative to the CB's current read
    /// pointer (`get_read_ptr`, tt-metal `writer.cpp`), so a pop moves the
    /// window and no index is rebased.
    pub fn tt_place_pops(&mut self) {
        let mut users: Map<OpId, Vec<OpId>> = Map::default();
        let mut section: Map<OpId, u8> = Map::default();
        let mut cur = 0u8;
        let mut pos: Map<OpId, usize> = Map::default();
        let mut order = 0usize;
        let mut scan = self.head;
        while !scan.is_null() {
            if matches!(self.ops[scan].op, Op::TT(TTOp::EndReader)) {
                cur = 1;
            } else if matches!(self.ops[scan].op, Op::TT(TTOp::EndCompute)) {
                cur = 2;
            }
            section.insert(scan, cur);
            pos.insert(scan, order);
            order += 1;
            for p in self.at(scan).parameters() {
                if !p.is_null() {
                    users.entry(p).or_default().push(scan);
                }
            }
            scan = self.next_op(scan);
        }
        // Last same-section user per CB load.
        let mut last_use: Map<OpId, OpId> = Map::default();
        for (load, us) in users.iter() {
            if !matches!(self.ops[*load].op, Op::Load { src, .. } if self.is_circular(src)) {
                continue;
            }
            let lsec = section[load];
            let mut best: Option<OpId> = None;
            for u in us {
                if section[u] != lsec {
                    continue;
                }
                if best.is_none_or(|b| pos[u] > pos[&b]) {
                    best = Some(*u);
                }
            }
            if let Some(b) = best {
                last_use.insert(*load, b);
            }
        }
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            let mut pops: Vec<OpId> = Vec::new();
            match self.ops[op_id].op {
                Op::Copy { src, .. } | Op::TT(TTOp::Copy { src, .. }) if self.is_circular(src) => {
                    pops.push(src);
                }
                Op::Load { src, .. } if self.is_circular(src) => {
                    if !users.contains_key(&op_id) {
                        pops.push(src);
                    }
                }
                _ => {}
            }
            for (load, user) in last_use.iter() {
                if *user == op_id {
                    let Op::Load { src, .. } = self.ops[*load].op else {
                        unreachable!("tt_place_pops: last use of a non-load")
                    };
                    pops.push(src);
                }
            }
            for cb in pops {
                self.insert_after(op_id, Op::TT(TTOp::PopFront { cb, n: 1 }));
            }
            op_id = next;
        }
    }

    /// 64-byte congruence for one DRAM-touching copy: the board crashes on
    /// incongruent NOC addresses. A read needs the DRAM and L1 addresses
    /// congruent mod 64, a write mod 16. A DRAM index is a byte address; a CB
    /// index is in datums, and a CB is placed at `≡16 mod 64` (where a tile
    /// image's datum block lives). A non-affine index is unprovable and is a
    /// loud rejection, never silently skipped.
    fn fifo_check_copy(&self, op_id: OpId, src: OpId, src_idx: OpId, dst: OpId, dst_idx: OpId) {
        let is_dram = |base: OpId| matches!(self.at(base), Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. });
        let src_dram = is_dram(src);
        let dst_dram = is_dram(dst);
        if !(src_dram || dst_dram) {
            return;
        }
        let align: i64 = if src_dram { 64 } else { 16 };
        let elem = |base: OpId| match self.at(base) {
            Op::Param { dtype, .. } | Op::Storage { dtype, .. } => {
                let bits = dtype.bit_size();
                if bits % 8 != 0 {
                    panic!("tt_fifo_check: copy {op_id:?} side {base:?} has sub-byte dtype {dtype:?}");
                }
                (bits / 8) as i64
            }
            other => panic!("tt_fifo_check: copy {op_id:?} side {base:?} is not a buffer: {other:?}"),
        };
        // Byte offset of one index on `base`: a DRAM index is already bytes, a
        // CB index is datums, and the CB's datum block starts at 16.
        let decompose = |idx: OpId, base: OpId| -> (i64, Map<OpId, i64>) {
            let base_off = if is_dram(base) { 0 } else { tt_isa::dm::TILE_DATA as i64 };
            let scale = if is_dram(base) { 1 } else { elem(base) };
            if idx.is_null() {
                return (base_off, Map::default());
            }
            let strides =
                self.get_strides(idx).unwrap_or_else(|| panic!("tt_fifo_check: copy {op_id:?} has unprovable index {idx:?}"));
            let mut vars: Map<OpId, i64> = Map::default();
            let mut base_bytes = base_off;
            for (var, (_, stride)) in strides {
                if var.is_null() {
                    base_bytes += stride * scale;
                } else {
                    vars.insert(var, stride * scale);
                }
            }
            (base_bytes, vars)
        };
        let (bs, sm) = decompose(src_idx, src);
        let (bd, dm) = decompose(dst_idx, dst);
        let check = |what: &str, a: i64, b: i64| {
            if (a - b).rem_euclid(align) != 0 {
                panic!("tt_fifo_check: copy {op_id:?} is {align}-byte incongruent at {what}: {a} vs {b}");
            }
        };
        for (v, ss) in sm.iter() {
            check(&std::format!("{v:?}"), *ss, dm.get(v).copied().unwrap_or(0));
        }
        for (v, ds) in dm.iter() {
            if sm.contains_key(v) {
                continue;
            }
            check(&std::format!("{v:?}"), 0, *ds);
        }
        check("base", bs, bd);
    }

    /// FIFO simulation over the placed sync ops: per-CB reserve/wait balance at
    /// section ends, pushed == popped program-wide per CB, producer-past-depth
    /// (the deadlock guard, a compile error), and the symbolic-loop traffic
    /// error. Loop trip counts multiply straight-line counts; boolean loops
    /// (conditionals) do not; a non-const non-bool loop length is symbolic and
    /// rejects sync traffic under it. Read-only; panics loudly, never launches
    /// a stalled kernel.
    pub(crate) fn tt_fifo_check(&self) {
        #[derive(Default)]
        struct Fifo {
            reserved: u64,
            avail: u64,
            waited: u64,
            pushed: u64,
            popped: u64,
        }
        let mut fifos: Map<OpId, Fifo> = Map::default();
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Frame {
            Trip(u64),
            Symbolic,
            Conditional,
        }
        let mut mult: u64 = 1;
        let mut stack: Vec<Frame> = Vec::new();
        let mut sym_loops: u32 = 0;
        let mut run_cb: Option<OpId> = None;
        let mut run_tiles: u64 = 0;
        let depth_of = |kernel: &Kernel, cb: OpId| -> u64 {
            match kernel.at(cb) {
                Op::Storage { len, .. } => (*len as u64) / 1024,
                other => panic!("tt_fifo_check: CB {cb:?} is not a storage op, got {other:?}"),
            }
        };
        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::TT(TTOp::EndReader) | Op::TT(TTOp::EndCompute) => {
                    for (cb, f) in fifos.iter() {
                        if f.reserved != 0 {
                            panic!("tt_fifo_check: CB{cb:?} reserve open at section end");
                        }
                        if f.waited != 0 {
                            panic!("tt_fifo_check: CB{cb:?} wait open at section end");
                        }
                    }
                    if let Some(prev) = run_cb {
                        if run_tiles > depth_of(self, prev) {
                            panic!(
                                "tt_fifo_check: producer pushes {run_tiles} tiles to CB{prev:?} (depth {}) before any other CB",
                                depth_of(self, prev)
                            );
                        }
                        run_cb = None;
                        run_tiles = 0;
                    }
                }
                Op::Loop { len } => {
                    if let Op::Const(c) = self.at(len) {
                        match c.as_dim() {
                            Some(t) => {
                                mult *= t as u64;
                                stack.push(Frame::Trip(t as u64));
                            }
                            None => panic!("tt_fifo_check: negative loop length under {op_id:?}"),
                        }
                    } else if self.dtype(len) == DType::Bool {
                        stack.push(Frame::Conditional);
                    } else {
                        sym_loops += 1;
                        stack.push(Frame::Symbolic);
                    }
                }
                Op::EndLoop => match stack.pop() {
                    Some(Frame::Trip(t)) => mult /= t,
                    Some(Frame::Symbolic) => sym_loops -= 1,
                    Some(Frame::Conditional) => {}
                    None => panic!("tt_fifo_check: EndLoop without Loop at {op_id:?}"),
                },
                Op::TT(TTOp::ReserveBack { cb, n }) => {
                    if sym_loops != 0 {
                        panic!("tt_fifo_check: FIFO traffic under a symbolic loop at {op_id:?}");
                    }
                    let f = fifos.entry(cb).or_default();
                    if f.reserved != 0 || f.waited != 0 {
                        panic!("tt_fifo_check: reserve CB{cb:?} with open transaction at {op_id:?}");
                    }
                    f.reserved = u64::from(n) * mult;
                }
                Op::TT(TTOp::PushBack { cb, n }) => {
                    if sym_loops != 0 {
                        panic!("tt_fifo_check: FIFO traffic under a symbolic loop at {op_id:?}");
                    }
                    let need = u64::from(n) * mult;
                    let f = fifos.entry(cb).or_default();
                    if f.reserved < need {
                        panic!("tt_fifo_check: push CB{cb:?} with {} reserved at {op_id:?}", f.reserved);
                    }
                    f.reserved -= need;
                    f.avail += need;
                    f.pushed += need;
                    if run_cb != Some(cb) {
                        if let Some(prev) = run_cb {
                            if run_tiles > depth_of(self, prev) {
                                panic!(
                                    "tt_fifo_check: producer pushes {run_tiles} tiles to CB{prev:?} (depth {}) before any other CB",
                                    depth_of(self, prev)
                                );
                            }
                        }
                        run_cb = Some(cb);
                        run_tiles = 0;
                    }
                    run_tiles += u64::from(n);
                }
                Op::TT(TTOp::WaitFront { cb, n }) => {
                    if sym_loops != 0 {
                        panic!("tt_fifo_check: FIFO traffic under a symbolic loop at {op_id:?}");
                    }
                    let need = u64::from(n) * mult;
                    let f = fifos.entry(cb).or_default();
                    if f.avail < need {
                        panic!("tt_fifo_check: wait CB{cb:?} with {} available at {op_id:?}", f.avail);
                    }
                    f.waited += need;
                    f.avail -= need;
                }
                Op::TT(TTOp::PopFront { cb, n }) => {
                    if sym_loops != 0 {
                        panic!("tt_fifo_check: FIFO traffic under a symbolic loop at {op_id:?}");
                    }
                    let need = u64::from(n) * mult;
                    let f = fifos.entry(cb).or_default();
                    if f.waited < need {
                        panic!("tt_fifo_check: pop CB{cb:?} with {} waited at {op_id:?}", f.waited);
                    }
                    f.waited -= need;
                    f.popped += need;
                }
                Op::Copy { src, src_idx, dst, dst_idx, .. } => {
                    self.fifo_check_copy(op_id, src, src_idx, dst, dst_idx);
                }
                Op::TT(TTOp::Copy { src, src_idx, dst, dst_idx, .. }) => {
                    self.fifo_check_copy(op_id, src, src_idx, dst, dst_idx);
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        if !stack.is_empty() {
            panic!("tt_fifo_check: kernel ends inside a loop");
        }
        if let Some(prev) = run_cb {
            if run_tiles > depth_of(self, prev) {
                panic!(
                    "tt_fifo_check: producer pushes {run_tiles} tiles to CB{prev:?} (depth {}) before any other CB",
                    depth_of(self, prev)
                );
            }
        }
        for (cb, f) in fifos.iter() {
            if f.pushed != f.popped {
                panic!("tt_fifo_check: CB{cb:?} pushed {} but popped {}", f.pushed, f.popped);
            }
        }
    }

    /// Lower the remaining plain ops of a TT kernel to their `TTOp` form:
    /// elementwise/select to SFPU programs, copes to L1 moves, and the
    /// phase sections around the reader and compute roles.
    // The debugging gate below makes the rest of this function unreachable on
    // purpose while a kernel is being brought up.
    #[allow(unreachable_code)]
    pub fn tt_dataflow_select(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        // Debugging gate. A kernel that does not pass stops here, so its IR can
        // be checked for manually visible reserve/wait/push/pop before this pass
        // runs. Remove this `todo!()` for that kernel only once its IR at this
        // boundary is good; then this pass and the ones after it are debugged.
        todo!("tt_dataflow_select: the IR here must show reserve/wait/push/pop; remove this todo once it is good");
        // A **broadcast operand** is a value with fewer tile axes than the op
        // it feeds (`exp(x - max)`: the `[32]` max row against a `[32,32]`
        // tile). Re-materialise the `Op::Expand` the frontend wrote and
        // `linearize` consumed, so the binary is `[32,32] op [32,32]`
        // downstream: the pattern table matches, the lowering resolves the
        // view to its reader CB, and the B mover broadcasts
        // (`READ_BROADCAST_COL`). It happens here, not at the end of
        // `tt_tile`, because the generic `verify` (which `tt_reindex` still
        // runs) rejects graph-shape ops after linearization; this pass is the
        // phase boundary. A scalar operand stays bare -- it is an `SfpuConst`.
        let mut expands: Vec<(OpId, OpId, OpId)> = Vec::new();
        for (id, x) in self.iter_unordered() {
            let (a, b) = match *x {
                Op::Binary { x, y, .. } => (x, y),
                Op::Select { if_true, if_false, .. } => (if_true, if_false),
                _ => continue,
            };
            let (ra, rb) = (self.shape(a).len(), self.shape(b).len());
            if ra == rb {
                continue;
            }
            let (narrow, wide) = if ra < rb { (a, b) } else { (b, a) };
            if self.shape(narrow).is_empty() {
                continue;
            }
            expands.push((id, narrow, wide));
        }
        for (id, narrow, wide) in expands {
            let dims = self.shape(wide);
            if dims.is_empty() || dims.iter().any(|d| *d <= 0) {
                continue;
            }
            let mut shape_ops: Vec<OpId> = Vec::with_capacity(dims.len());
            for d in dims {
                shape_ops.push(self.insert_const_idx_before(id, d));
            }
            let shape = if shape_ops.len() == 1 {
                shape_ops[0]
            } else {
                self.insert_before(id, Op::Stack { ops: shape_ops.into() })
            };
            let expand = self.insert_before(id, Op::Expand { x: narrow, shape });
            match &mut self.ops[id].op {
                Op::Binary { x, y, .. } => {
                    if *x == narrow {
                        *x = expand;
                    } else {
                        *y = expand;
                    }
                }
                Op::Select { if_true, if_false, .. } => {
                    if *if_true == narrow {
                        *if_true = expand;
                    } else {
                        *if_false = expand;
                    }
                }
                other => unreachable!("tt_dataflow_select: broadcast wrap on {other:?}"),
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
            // `[32,32] -> [32]`: `Row` or `Col` is decided by the reduce
            // axis in the emitter (the reduce variable's stride); the shape
            // alone cannot tell the two apart. See `TtEmit::Reduce` there.
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
            // A scalar value: a loop-carried accumulator over a reduce that
            // already collapsed its lanes (`sum` over one row). The SFPU is
            // lane-wise and the value lives in lane 0, so the rank-1 body
            // computes it correctly.
            let p0 = Pat::unary(Pat::bind('x').shape([] as [i64; 0]), uop).shape([] as [i64; 0]);
            pats.push((Pat::Any(Box::new([p2, p1, p0])), TtEmit::Sfpu { kind, arity: 1 }));
        }
        for &bop in BOPS {
            let Some(kind) = bop_kind(bop) else { continue };
            let p2 = Pat::binary(bop, Pat::bind('x').shape([32i64, 32]), Pat::bind('y').shape([32i64, 32])).shape([32i64, 32]);
            let p1 = Pat::binary(bop, Pat::bind('x').shape([32i64]), Pat::bind('y').shape([32i64])).shape([32i64]);
            let p0 = Pat::binary(bop, Pat::bind('x').shape([] as [i64; 0]), Pat::bind('y').shape([] as [i64; 0]))
                .shape([] as [i64; 0]);
            pats.push((Pat::Any(Box::new([p2, p1, p0])), TtEmit::Sfpu { kind, arity: 2 }));
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
        // The writer CBs and their drains: a store into one of them is only
        // the pack side, the `Op::Copy` to the DRAM param is the section's
        // traffic. Collected before the encoder turns those copies into
        // `TTOp::Copy`.
        let mut writer_cbs: Set<OpId> = Set::default();
        let mut drain_of: Map<OpId, OpId> = Map::default();
        for (id, op) in self.iter_unordered() {
            if let Op::Copy { src, dst, .. } = *op
                && self.is_circular(src)
                && matches!(self.ops[dst].op, Op::Param { kind: ParamKind::GlobalMut, .. })
            {
                writer_cbs.insert(src);
                drain_of.insert(src, id);
            }
        }
        // Phase 1 — encode the plan `tt_add_movers` wrote. The CBs, the block
        // copies and the two section markers are ordinary SSA by now; this
        // walk is mechanical: a parameter-side copy becomes the mover's
        // `TTOp::Copy` with an explicit byte count taken from the storage it
        // moves into (reader) or out of (writer).
        let reader_cb_of: Map<OpId, OpId> = Map::default();
        // A `[32]` row published for an `Op::Expand` consumer is one whole
        // tile image: the shaped read takes `TILE_SLOT`, not a datum block.
        let mut broadcast: Set<OpId> = Set::default();
        for (_, op) in self.iter_unordered() {
            if let Op::Expand { x, .. } = *op
                && let Op::Load { src, .. } = self.ops[x].op
                && self.is_circular(src)
            {
                broadcast.insert(src);
            }
        }
        // (copy, reader direction, inside a loop). A copy inside a loop moves
        // one iteration's slice, so it is sized as one tile; the loop itself is
        // unrolled into list entries at emission.
        let mut boundary: Vec<(OpId, bool, bool)> = Vec::new();
        let mut loop_depth = 0u32;
        let mut scan = self.head;
        while !scan.is_null() {
            match self.ops[scan].op {
                Op::Loop { .. } => loop_depth += 1,
                Op::EndLoop => loop_depth = loop_depth.saturating_sub(1),
                Op::Copy { src, dst, .. } => {
                    let src_dram = matches!(self.ops[src].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. });
                    let dst_dram = matches!(self.ops[dst].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. });
                    if src_dram && self.is_circular(dst) {
                        boundary.push((scan, true, loop_depth > 0));
                    } else if self.is_circular(src) && dst_dram {
                        boundary.push((scan, false, loop_depth > 0));
                    }
                }
                _ => {}
            }
            scan = self.next_op(scan);
        }
        for (id, reader, in_loop) in boundary {
            let Op::Copy { src, src_idx, dst, dst_idx } = self.ops[id].op else {
                unreachable!("tt_dataflow_select: a boundary op is not a copy")
            };
            let bytes = if reader {
                if broadcast.contains(&dst) {
                    tt_isa::dm::TILE_SLOT as u32
                } else if in_loop {
                    // One entry per iteration: one tile's datums.
                    let Some(bytes) = (self.dtype(src).bit_size() as u32 / 8).checked_mul(1024) else {
                        return;
                    };
                    bytes
                } else {
                    let Op::Storage { len, .. } = self.ops[dst].op else {
                        unreachable!("tt_dataflow_select: a reader copy targets a storage")
                    };
                    let Some(bytes) = u32::try_from(len).ok().and_then(|l| l.checked_mul(self.dtype(src).bit_size() as u32 / 8))
                    else {
                        return;
                    };
                    bytes
                }
            } else {
                // The writer CB holds one packed tile: the packet is a whole
                // `Dst` tile whatever the store's access shape.
                let Some(bytes) = (self.dtype(src).bit_size() as u32 / 8).checked_mul(1024) else {
                    return;
                };
                bytes
            };
            self.ops[id].op = Op::TT(TTOp::Copy { src, src_idx, dst, dst_idx, bytes });
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
        // Writes to a `Register` storage are not data movement: the storage is
        // a DST row, and a store/load pair through it is a hand-off of the
        // value from the producer's tile to the storage's tile. Give the
        // storage one DST slot up front, so its loads resolve there and its
        // producer is redirected to it.
        //
        // A loop-carried accumulator is the common case: it is written inside
        // a loop and read by the next iteration in place, so its pre-loop
        // initialiser fills the same slot ahead of the loop header. But the
        // same shape occurs straight-line after `tt_tile` unrolls a loop into
        // lanes (matmul's K-loop: `reduce` then `acc = reduce + zero`, with
        // the loop gone). There is no loop header then, so the write itself
        // is the anchor.
        let mut accumulators: Map<OpId, (OpId, OpId)> = Map::default(); // storage -> (slot, anchor)
        let mut loop_stack: Vec<OpId> = Vec::new();
        // In-loop writes first: they anchor on their loop, so the pre-loop
        // initialiser is filled ahead of that loop (which is already in the
        // compute section). Straight-line writes anchor on themselves.
        let mut pending_loop: Vec<(OpId, OpId)> = Vec::new();
        let mut pending_linear: Vec<(OpId, OpId)> = Vec::new();
        let mut scan = self.head;
        for _ in 0..250_000 {
            if scan.is_null() {
                break;
            }
            match self.ops[scan].op {
                Op::Loop { .. } => loop_stack.push(scan),
                Op::EndLoop => {
                    loop_stack.pop();
                }
                Op::Store { dst, .. } if matches!(self.ops[dst].op, Op::Storage { scope: MemScope::Register, .. }) => {
                    match loop_stack.last() {
                        Some(&header) => pending_loop.push((dst, header)),
                        None => pending_linear.push((dst, scan)),
                    }
                }
                _ => {}
            }
            scan = self.next_op(scan);
        }
        for (storage, header) in pending_loop.into_iter().chain(pending_linear) {
            if accumulators.contains_key(&storage) {
                continue;
            }
            let Op::Storage { dtype, .. } = self.ops[storage].op else {
                unreachable!("an accumulator is a storage")
            };
            let slot = self.tt_new_dst(dtype, prologue);
            accumulators.insert(storage, (slot, header));
        }
        let mut lower = TtLowering {
            pats: &pats,
            matched: &matched,
            prologue,
            reader_cb_of: &reader_cb_of,
            rcs: &rcs,
            materialized: Map::default(),
            unpacked: Map::default(),
            reg_slot: accumulators.iter().map(|(&s, &(slot, _))| (s, slot)).collect(),
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
                // A DRAM store drains a value, and a Register store is the
                // loop-carried accumulator: its source is a root too, or a
                // reduce that only feeds the accumulator is never lowered.
                let interesting = matches!(
                    self.ops[dst].op,
                    Op::Param { kind: ParamKind::GlobalMut, .. }
                        | Op::Storage { scope: MemScope::Register | MemScope::Circular, .. }
                );
                if interesting {
                    roots.push(src);
                }
            }
            scan = self.next_op(scan);
        }
        for root in roots {
            let _ = self.tt_lower_value(root, &mut lower);
        }
        // Accumulator hand-off: the produced value lives in a DST slot; move
        // it onto the accumulator's slot, fill the pre-loop initialiser there,
        // and drop the symbolic store. The loads already resolved through
        // `reg_slot`.
        let mut removes: Vec<OpId> = Vec::new();
        let mut scan = self.head;
        for _ in 0..250_000 {
            if scan.is_null() {
                break;
            }
            if let Op::Store { dst, src, .. } = self.ops[scan].op
                && let Some(&(slot, header)) = accumulators.get(&dst)
            {
                if matches!(self.ops[src].op, Op::Const(_)) {
                    self.tt_fill_const_into(src, slot, header);
                } else if let Some((_, produced)) = self.tt_lowered_storage(src, &lower) {
                    let produced_is_slot = matches!(self.ops[produced].op, Op::Storage { scope: MemScope::Register, .. });
                    if produced_is_slot && produced != slot {
                        self.tt_redirect_slot(produced, slot);
                        self.remove_op(produced);
                    }
                }
                removes.push(scan);
            }
            scan = self.next_op(scan);
        }
        for op in removes {
            self.remove_op(op);
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
        //
        // The writer CBs and their drains are the ones `tt_add_movers` planned:
        // a store into one of them is only the pack side.
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
                    if writer_cbs.contains(&base) {
                        stores.push((id, base, off, src));
                    } else if matches!(self.ops[base].op, Op::Storage { .. }) {
                        // Already lowered; nothing owned here.
                    } else {
                        // A DRAM store survived `tt_add_movers`: its plan bailed,
                        // and nothing here may guess at one.
                        return;
                    }
                }
                _ => {}
            }
            id = self.ops[id].next;
        }
        if !stores.is_empty() {
            // Compute-side rewrites in head order; drains are already SSA.
            for (store, base, _off, src) in stores.iter() {
                // A constant store (a `zeros`/`full` buffer — the KV cache
                // initialiser is the model's first kernel) has no DST producer
                // to pack. Fill the writer CB with the value and let the
                // planned drain move it: the CB is FP32 whatever the
                // parameter's dtype, and the host boundary narrows on the way
                // out. The store's index cone dies with it.
                if let Op::Const(c) = self.ops[*src].op {
                    if !writer_cbs.contains(base) {
                        todo!("a constant store outside a writer CB has no lowering");
                    }
                    let bits = (c.as_float().expect("a stored constant is numeric") as f32).to_bits();
                    self.insert_before(*store, Op::TT(TTOp::SfpConst(Box::new(TTSfpConst { reg: L0, bits }))));
                    for o in (0..64u16).step_by(2) {
                        self.insert_before(
                            *store,
                            Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot: *base, at: o, reg: L0, fmt: FP32 }))),
                        );
                    }
                    self.remove_op(*store);
                    continue;
                }
                let Some((_dtype, storage)) = self.tt_lowered_storage(*src, &lower) else {
                    // Unlowered value: stays for later increments.
                    continue;
                };
                if writer_cbs.contains(base) {
                    // The drain to DRAM is already SSA (`tt_add_movers`); this
                    // only fills the writer CB.
                    if self.is_circular(storage) && storage != *base {
                        // Multi-use value already in L1: drain it directly by
                        // retargeting the planned copy, and drop the writer CB
                        // it replaces. The drain may already be encoded
                        // (`tt_dataflow_select` phase 1 runs before this), so
                        // both forms are rewritten and the byte count follows
                        // the tile it now moves.
                        let drain = drain_of[base];
                        let zero = self.insert_const_idx_before(drain, 0i64);
                        let Some(bytes) = (self.dtype(storage).bit_size() as u32 / 8).checked_mul(1024) else {
                            return;
                        };
                        match &mut self.ops[drain].op {
                            Op::Copy { src, src_idx, .. } => {
                                *src = storage;
                                *src_idx = zero;
                            }
                            Op::TT(TTOp::Copy { src, src_idx, bytes: b, .. }) => {
                                *src = storage;
                                *src_idx = zero;
                                *b = bytes;
                            }
                            other => unreachable!("the planned drain is not a copy: {other:?}"),
                        }
                        self.remove_op(*base);
                    } else if !self.is_circular(storage) {
                        // Output boundary: pack the DST row into the writer CB.
                        // FP32, because the packer writes FP32 out of `Dst`;
                        // when the param is 16-bit, the host boundary narrows
                        // on the way back (`TTDmMove::narrow`).
                        let pack_bytes = match (DType::F32.bit_size() as u32 / 8).checked_mul(1024) {
                            Some(b) if b > 0 => b,
                            _ => return,
                        };
                        let zero_pack = self.insert_const_idx_before(*store, 0i64);
                        // A store inside a loop packs into that iteration's own
                        // tile slot, so the drain (the same loop at the tail)
                        // finds each iteration's value where it left it.
                        let dst_idx = match self.tt_enclosing_loop(*store) {
                            Some(lv) => {
                                let stride = self.insert_const_idx_before(*store, 1024i64);
                                self.insert_before(*store, Op::Binary { x: lv, y: stride, bop: BOp::Mul })
                            }
                            None => zero_pack,
                        };
                        // The pack is a tile move (DST -> L1), not an SSA
                        // store: after allocation there is no SSA, so it goes
                        // out as `Copy` like every other memory-level move. It
                        // keeps the storage as its source, which is what the
                        // reuse pass needs to know the row is still live.
                        self.insert_before(
                            *store,
                            Op::TT(TTOp::Copy { src: storage, src_idx: zero_pack, dst: *base, dst_idx, bytes: pack_bytes }),
                        );
                    }
                    self.remove_op(*store);
                }
            }
        }
        // A CB `Load` is a value read now, not a FIFO pop: the `Op::Copy` is the
        // traffic. DCE keeps it (a circular load used to be the pop side
        // effect), so the ones compute resolved are dropped here; the value
        // lives in its DST tile. A load a remaining op still names was *not*
        // resolved — dropping it would leave a dangling operand, so it stays
        // and the backend's verifier reports the unlowered consumer.
        let mut resolved: Vec<OpId> = Vec::new();
        for (id, op) in self.iter_unordered() {
            if let Op::Load { src, .. } = *op
                && self.is_circular(src)
            {
                resolved.push(id);
            }
        }
        for id in resolved {
            let live = self.iter_unordered().any(|(other, op)| other != id && op.parameters().any(|p| p == id));
            if !live {
                self.remove_op(id);
            }
        }
        // The drains are moves now, so the high-level value chains they
        // replaced (an accumulator load, its index math) are dead. This DCE
        // runs after phase 3, unlike the phase-2 one, which still saw the
        // store as a root.
        self.dead_code_elimination();
        // No generic `verify` here: the phase boundary has been crossed. The
        // lowered IR is not generic SSA — a value is a DST tile, a `[32]`
        // vector is only a shape carrier, and row ops own their values — so
        // `tt_verify` is the checker from here on.
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
        let (_bits, _fmt) = match c {
            Constant::F32(b) => (u32::from_le_bytes(b), FP32),
            Constant::I32(v) => (v as u32, INT32),
            other => todo!("a fill of a {other:?} constant has no native form"),
        };
        let dtype = c.dtype();
        let slot = self.tt_new_dst(dtype, prologue);
        self.tt_fill_const_into(konst, slot, consumer);
        Some((dtype, slot))
    }

    /// Fill `slot` with `konst`'s value (one lane-register immediate, stored
    /// to every row group), before `before`.
    fn tt_fill_const_into(&mut self, konst: OpId, slot: OpId, before: OpId) {
        let Op::Const(c) = self.ops[konst].op else {
            return;
        };
        let (bits, fmt) = match c {
            Constant::F32(b) => (u32::from_le_bytes(b), FP32),
            Constant::I32(v) => (v as u32, INT32),
            other => todo!("a fill of a {other:?} constant has no native form"),
        };
        self.insert_before(before, Op::TT(TTOp::SfpConst(Box::new(TTSfpConst { reg: L0, bits }))));
        for o in (0..64u16).step_by(2) {
            self.insert_before(before, Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L0, fmt }))));
        }
    }

    /// The loop-carried accumulator slot this value is the back edge of, if it
    /// is one: a `Store` inside a loop into a `Register` storage whose slot was
    /// reserved up front (`TtLowering::reg_slot`). The producer must write
    /// *that* slot, not a fresh tile: the loop's next iteration reads the
    /// storage in place, so a fresh destination would leave it untouched and
    /// every iteration would start from the pre-loop value.
    fn tt_accumulator_dst(&self, value: OpId, lower: &TtLowering) -> Option<OpId> {
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() {
                return None;
            }
            if let Op::Store { dst, src, .. } = self.ops[id].op
                && src == value
                && let Some(&slot) = lower.reg_slot.get(&dst)
            {
                return Some(slot);
            }
            id = self.next_op(id);
        }
        None
    }

    /// Rewrite every reference to DST storage `from` to `to`.
    fn tt_redirect_slot(&mut self, from: OpId, to: OpId) {
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() {
                break;
            }
            let next = self.next_op(id);
            match &mut self.ops[id].op {
                Op::TT(TTOp::SfpLoad(load)) if load.slot == from => load.slot = to,
                Op::TT(TTOp::SfpStore(store)) if store.slot == from => store.slot = to,
                Op::TT(TTOp::Sfpu { slot, .. }) if *slot == from => *slot = to,
                Op::TT(TTOp::Setrwc { dst }) if *dst == from => *dst = to,
                Op::TT(TTOp::Copy { src, dst, .. }) => {
                    if *src == from {
                        *src = to;
                    }
                    if *dst == from {
                        *dst = to;
                    }
                }
                _ => {}
            }
            id = next;
        }
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
            // A broadcast view is the value it wraps: the reader publishes the
            // `[32]` row as one tile, the mover replicates it across the
            // tile's lanes, and every consumer reads the same reader CB.
            Op::Expand { x, .. } => return self.tt_lower_value(x, lower),
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
                // The source's *IR* dtype is what the cast means, and it must
                // be read **before** `tt_resolve_dst`: that lowers `x` in
                // place, and a lowered value reports its tile's dtype (a
                // comparison's tile is labelled F32 and holds the raw word).
                //
                // A `Bool` is a raw 1/0 word there, not 1.0/0.0 -- the
                // comparison programs store with `fmt: INT32` and `L6 = 1`
                // is "a Bool's true" -- so `f32(bool)` (`Tensor::relu` is
                // `cmpgt(0) * x`) is a real int -> float conversion, even
                // though nothing about the tile says so.
                let src = self.dtype(x);
                let (from, a) = self.tt_resolve_dst(x, id, lower)?;
                if src == dtype {
                    // Identity: the result *is* the source tile. Recording it
                    // in `materialized` resolves every later use to that tile,
                    // and the now-dead `Cast` goes with the DCE below.
                    lower.materialized.insert(id, a);
                    return Some((dtype, a));
                }
                let slot = self.tt_new_dst(dtype, lower.prologue);
                let mut ops = match (src, dtype) {
                    (DType::I32 | DType::Bool, DType::F32) => self.tt_sfp_cast_i32_f32(a, slot),
                    (DType::F32, DType::I32) => self.tt_sfp_cast_f32_i32(a, slot),
                    _ => todo!("cast {src:?} -> {dtype:?} has no native row-op program (tile {from:?})"),
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
                // A single-use comparison folds into the predicate: the Bool
                // tile would exist only to become lane flags, and materialising
                // it costs the whole IEEE comparison program -- ~40
                // instructions holding `L0..L7`, plus the `NaN`/`+0`
                // canonicalisation ML does not need. The predicate form is what
                // tt-metal's own relu/leaky-relu uses (`SFPLOAD; SFPSETCC(x);
                // SFPMUL; SFPENCC; SFPSTORE`), and it is what keeps a fused
                // `relu`-composed chain (`elu`) inside the register file.
                let fold = if lower.rcs.get(&cond).copied().unwrap_or(0) <= 1 {
                    if let &Op::Binary { x: ca, y: cb, bop } = self.at(cond) {
                        matches!(bop, BOp::Cmpgt | BOp::Cmpge | BOp::Cmplt | BOp::Eq | BOp::NotEq).then_some((ca, cb, bop))
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some((ca, cb, bop)) = fold {
                    use tt_isa::isa::generated::defs;
                    let (mov, gt, ple, pushc, popc) = (
                        defs::SFPMOV.opcode(),
                        defs::SFPGT.opcode(),
                        defs::SFPLE.opcode(),
                        defs::SFPPUSHC.opcode(),
                        defs::SFPPOPC.opcode(),
                    );
                    let sfp = |ops: &mut Vec<Op>, op: u8, args: [u16; 5]| {
                        ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op, args })));
                    };
                    /// What a select operand costs to put in a lane register:
                    /// a load of its tile, or (for a scalar 0/1) a move from the
                    /// hardware's read-only constant register -- no tile, no
                    /// DST traffic.
                    #[derive(Clone, Copy, PartialEq)]
                    enum Put {
                        Load(OpId),
                        Mov(u16),
                    }
                    let plan = |k: &mut Kernel, o: OpId, at: OpId, lower: &mut TtLowering| -> Option<Put> {
                        if let Op::Const(c) = *k.at(o) {
                            let bits = match c {
                                Constant::F32(b) => u32::from_le_bytes(b),
                                Constant::I32(v) => v as u32,
                                _ => 1,
                            };
                            if bits == 0 {
                                return Some(Put::Mov(LREG_ZERO));
                            }
                            if bits == 0x3f80_0000 {
                                return Some(Put::Mov(LREG_ONE));
                            }
                        }
                        let (_, slot) = if matches!(k.ops[o].op, Op::Const(_)) {
                            k.tt_fill_const(o, at, lower.prologue)?
                        } else {
                            k.tt_resolve_dst(o, at, lower)?
                        };
                        Some(Put::Load(slot))
                    };
                    let dtype = self.dtype(if_true);
                    let (a, b) = (plan(self, ca, id, lower)?, plan(self, cb, id, lower)?);
                    let (t, f) = (plan(self, if_true, id, lower)?, plan(self, if_false, id, lower)?);
                    // `a` and `b` live in `L0`/`L1` because the predicate's
                    // encoding names them; `t`/`f` reuse a register when they
                    // are the same value, so `cmpgt(x, 0) * x` loads `x` once.
                    let rt = if t == a {
                        L0
                    } else if t == b {
                        L1
                    } else if t == f {
                        L2
                    } else {
                        L2
                    };
                    let rf = if f == a {
                        L0
                    } else if f == b {
                        L1
                    } else if f == t {
                        rt
                    } else {
                        L3
                    };
                    let slot = self.tt_new_dst(dtype, lower.prologue);
                    // `ne`'s predicate is equality, so it starts from the true
                    // word and overwrites with the false one; every other kind
                    // starts from the false word.
                    let (dflt, pick) = if bop == BOp::NotEq { (rt, rf) } else { (rf, rt) };
                    let mut ops = self.tt_row_loop(|ops, o| {
                        let put = |ops: &mut Vec<Op>, p: Put, reg: u8| match p {
                            Put::Load(slot) => {
                                ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot, at: o, reg, fmt: INT32 }))));
                            }
                            Put::Mov(c) => sfp(ops, mov, [c, u16::from(reg), 0, 0, 0]),
                        };
                        put(ops, a, L0);
                        put(ops, b, L1);
                        if rt != L0 && rt != L1 {
                            put(ops, t, rt);
                        }
                        if rf != rt && rf != L0 && rf != L1 {
                            put(ops, f, rf);
                        }
                        sfp(ops, mov, [u16::from(dflt), u16::from(L4), 0, 0, 0]);
                        // `SFPGT [p, q]` is `q > p` and `SFPLE [p, q]` is
                        // `p <= q` (the first operand is the object), so each
                        // kind is a pair pick, not an opcode per direction.
                        let (first, second) = match bop {
                            BOp::Cmpgt => (gt, [u16::from(L1), u16::from(L0)]),
                            BOp::Cmpge => (ple, [u16::from(L0), u16::from(L1)]),
                            BOp::Cmplt => (gt, [u16::from(L0), u16::from(L1)]),
                            _ => (ple, [u16::from(L0), u16::from(L1)]),
                        };
                        sfp(ops, pushc, [0; 5]);
                        sfp(ops, first, [second[0], second[1], 1, 0, 0]);
                        if matches!(bop, BOp::Eq | BOp::NotEq) {
                            // Equality is `a <= b` and `b <= a`.
                            sfp(ops, pushc, [0; 5]);
                            sfp(ops, ple, [u16::from(L1), u16::from(L0), 1, 0, 0]);
                        }
                        sfp(ops, mov, [u16::from(pick), u16::from(L4), 0, 0, 0]);
                        sfp(ops, popc, [0; 5]);
                        if matches!(bop, BOp::Eq | BOp::NotEq) {
                            sfp(ops, popc, [0; 5]);
                        }
                        ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L4, fmt: INT32 }))));
                    });
                    let last = ops.pop().expect("the sequence is never empty");
                    for op in ops {
                        self.insert_before(id, op);
                    }
                    self.ops[id].op = last;
                    return self.tt_materialize(id, dtype, slot, lower);
                }
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
            // A loop accumulator's logical storage resolves to its DST slot.
            Op::Storage { dtype, .. } => Some((dtype, lower.reg_slot.get(&base).copied().unwrap_or(base))),
            _ => None,
        }
    }

    /// The innermost open `Op::Loop` at `at`, if any. Linear over the kernel;
    /// only the multi-tile CB path asks.
    fn tt_enclosing_loop(&self, at: OpId) -> Option<OpId> {
        let mut stack: Vec<OpId> = Vec::new();
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() || id == at {
                break;
            }
            match self.ops[id].op {
                Op::Loop { .. } => stack.push(id),
                Op::EndLoop => {
                    stack.pop();
                }
                _ => {}
            }
            id = self.next_op(id);
        }
        stack.last().copied()
    }

    /// Resolve a value to a DST row, unpacking a CB if needed. This is the
    /// only CB -> DST conversion in compute; the copy is single-use and the
    /// reuse pass can coalesce its row with the producer.
    fn tt_resolve_dst(&mut self, id: OpId, consumer: OpId, lower: &mut TtLowering) -> Option<(DType, OpId)> {
        let (dtype, storage) = self.tt_lower_value(id, lower)?;
        match self.ops[storage].op {
            Op::Storage { scope: MemScope::Register, .. } => Some((dtype, storage)),
            Op::Storage { scope: MemScope::Circular, .. } => {
                // A multi-tile CB consumed inside a loop is a block: the slot
                // is the loop counter, and every iteration unpacks afresh (so
                // the one-DST-tile cache below does not apply).
                if let Some(lv) = self.tt_enclosing_loop(consumer) {
                    if matches!(self.ops[storage].op, Op::Storage { len, .. } if len > 1024) {
                        let dst = self.tt_new_dst(dtype, lower.prologue);
                        let stride = self.insert_const_idx_before(consumer, 1024i64);
                        let src_idx = self.insert_before(consumer, Op::Binary { x: lv, y: stride, bop: BOp::Mul });
                        let zero = self.insert_const_idx_before(consumer, 0i64);
                        let bytes = (dtype.bit_size() as u32 / 8).checked_mul(1024)?;
                        self.insert_before(consumer, Op::TT(TTOp::Copy { src: storage, src_idx, dst, dst_idx: zero, bytes }));
                        return Some((dtype, dst));
                    }
                }
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
        use crate::codegen::tenstorrent::kind;
        use crate::codegen::tenstorrent::sfpu::ops::kind_sfpu as sf;
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
        let body = self.tt_row_loop(|ops, o| {
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
        self.tt_row_loop(|ops, o| {
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
        self.tt_row_loop(|ops, o| {
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
        self.tt_row_loop(|ops, o| {
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
        self.tt_row_loop(|ops, o| {
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
        let body = self.tt_row_loop(|ops, o| {
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
        self.tt_row_loop(|ops, o| {
            ops.push(Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: a, at: o, reg: L0, fmt: INT32 }))));
            ops.push(Op::TT(TTOp::Sfp(TTSfpInsn { op: not, args: [u16::from(L0), u16::from(L2), 0, 0, 0] })));
            ops.push(Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot, at: o, reg: L2, fmt: INT32 }))));
        })
    }

    /// `ops::sqrt_program`, `RSQRT` when `kind` says so.
    fn tt_sfp_sqrt(&mut self, kind: u32, a: OpId, slot: OpId) -> Vec<Op> {
        use crate::codegen::tenstorrent::sfpu::ops::kind_sfpu as sf;
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
        use crate::codegen::tenstorrent::sfpu::ops::kind_sfpu as sf;
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
        self.tt_row_loop(|ops, o| {
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
        let body = self.tt_row_loop(|ops, o| {
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
        let body = self.tt_row_loop(|ops, o| {
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
        use crate::codegen::tenstorrent::sfpu::ops::kind_sfpu as sf;
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
        use crate::codegen::tenstorrent::sfpu::ops::kind_sfpu as sf;
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
                let slot = match self.tt_accumulator_dst(id, lower) {
                    Some(slot) => slot,
                    None => self.tt_new_dst(dtype, lower.prologue),
                };
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
            TtEmit::Reduce(table_dim) => {
                let Op::Reduce { rop, x: reduced, reduce_axis } = self.ops[id].op else {
                    return None;
                };
                // The fold runs along the tile axis the feeder access names.
                // After `tt_reindex` the index is in tile units, so the reduce
                // variable's stride is the tell: 1 = the lanes are tile
                // columns (`Row` folds the lane axis), 32 = they are tile rows
                // (`Col` folds the row axis). The pattern table only supplies
                // the fallback when the feeder is not a typed access.
                let dim = match self.ops[reduced].op {
                    Op::Load { src_idx, .. } => self
                        .get_strides(src_idx)
                        .and_then(|s| s.get(&reduce_axis).copied())
                        .map(|(_, stride)| if stride == 1 { TileDim::Row } else { TileDim::Col })
                        .unwrap_or(table_dim),
                    _ => table_dim,
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
                        ((*pa, *la, self.dtype(*pa)), (*pb, *lb, self.dtype(*pb)))
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
                let Some(((pa, dtype_a), (pb, _dtype_b), k)) = fab else {
                    return None;
                };
                // The reduced axis must be the shared K lane: contracting
                // any other axis is not a matmul.
                if !matches!(&self.ops[root].op, Op::Reduce { reduce_axis, .. } if *reduce_axis == k) {
                    return None;
                }
                let cb_of = |s: OpId| {
                    if self.is_circular(s) {
                        Some(s)
                    } else {
                        lower.reader_cb_of.get(&s).copied()
                    }
                };
                let (Some(cb_a), Some(cb_b)) = (cb_of(pa), cb_of(pb)) else {
                    return None;
                };
                // The native matmul is the UNPACR/MVMUL sequence, not a
                // structured op: `SrcA`/`SrcB` are two small banks the unpack
                // and math roles hand back and forth (`flip` on the unpack,
                // `flip_a`/`flip_b` on the `MVMUL` that releases one). One
                // output tile is 64 DST rows: `fi` 0 accumulates rows 0..31,
                // `MATH_AM_FI` moves the base on 32 rows for `fi` 1, so the
                // accumulator is the tile's slot directly.
                use crate::codegen::tenstorrent::matmul::{MATH_AM_FI, MATH_AM_GROUP, MATH_AM_HALF, MATH_AM_PHASE};
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
        // Last use of every logical Register storage over all operands. A
        // value defined *outside* a loop and read inside it is read again on
        // the next iteration, after the body's last write: its row must stay
        // allocated to the loop's end, or a loop temporary takes it over and
        // the second iteration reads the temporary (measured: softmax's
        // `sum(exp(x - max))`, where `max` is loaded before the loop and used
        // in every iteration -- the device sum was 15% high, data-dependent
        // because the ramps we used before had identical maxima per row).
        let mut pos: Map<OpId, usize> = Map::default();
        for (i, &op) in ops.iter().enumerate() {
            pos.insert(op, i);
        }
        let mut loops: Vec<(usize, usize)> = Vec::new();
        let mut open: Vec<usize> = Vec::new();
        for (i, &op) in ops.iter().enumerate() {
            match self.ops[op].op {
                Op::Loop { .. } => open.push(i),
                Op::EndLoop => {
                    if let Some(start) = open.pop() {
                        loops.push((start, i));
                    }
                }
                _ => {}
            }
        }
        let mut last_use: Map<OpId, usize> = Map::default();
        for (i, &op) in ops.iter().enumerate() {
            for p in self.ops[op].op.parameters() {
                if !matches!(self.ops[p].op, Op::Storage { scope: MemScope::Register, .. }) {
                    continue;
                }
                let mut last = i;
                let def = pos.get(&p).copied().unwrap_or(usize::MAX);
                for &(start, end) in &loops {
                    if def < start && start < i && i < end {
                        last = last.max(end);
                    }
                }
                let slot = last_use.entry(p).or_insert(last);
                *slot = (*slot).max(last);
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
        // A register hand-over ignores the tile formats. `SFPLOAD` copies the
        // tile's bits in either format, and the only store-side difference
        // (`FP32` flushes denormals on the way into the tile) is a flush of
        // values below 2^-126, which is below any ML tolerance -- and not doing
        // it is closer to IEEE than the tile path, not further.
        for u in 0..n {
            for (slot, _, _, _) in &loads[u] {
                readers.entry(*slot).or_default().push(u);
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
                // A whole-tile move in a gap cannot be *inlined*: the merged
                // body runs once per row, a tile copy once per tile. It can be
                // **hoisted** in front of the merged body instead, where it
                // still runs once per tile. That is sound when no part of the
                // cone writes the tile it fills or its circular buffer: nothing
                // in the body then depends on where the fill happens. This is
                // what lets a kernel whose argument is the kernel input fold its
                // whole chain (`exp2`): the input's unpack sits between the
                // bodies that read it, and hoisting it in front makes the two
                // bodies share the tile the way `sigmoid` already does.
                let mut hoist: Vec<OpId> = Vec::new();
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
                            Op::TT(TTOp::Copy { src, dst, .. })
                                // An unpack only: a pack writes a circular buffer
                                // the cone reads from, and stays where it is.
                                if self.tt_copy_kind(g) == TtCopy::Unpack
                                    && !(s..=end)
                                        .any(|p| stores[p].iter().any(|(slot, ..)| *slot == *dst || *slot == *src)) =>
                            {
                                hoist.push(g);
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
                // One tile read more than once in the cone: the first read keeps
                // its `SFPLOAD` and copies the value into a carry right there,
                // every later read is handed the register. When the cone stores
                // to the slot, a read only sees the same value as an earlier one
                // *within* a segment -- the stretch between two stores -- so the
                // reads are grouped by the store that precedes them and each
                // group carries separately. The formats need not agree: a load
                // only copies the tile's bits.
                //
                // Keyed by the load that starts a segment; the value is that
                // segment's last reading unit.
                let mut segment_start: Map<OpId, (u8, usize)> = Map::default();
                let mut segments: Map<(OpId, Option<usize>), Vec<(usize, u8, OpId)>> = Map::default();
                for u in s..=end {
                    for (slot, reg, _, op) in &loads[u] {
                        let stores: Vec<usize> =
                            producers.get(slot).into_iter().flatten().copied().filter(|&p| p >= s && p <= end).collect();
                        // A read in a unit that also stores the slot is left
                        // alone: inside one unit the op order decides which
                        // value it sees.
                        if stores.contains(&u) {
                            continue;
                        }
                        let key = stores.iter().copied().filter(|&v| v < u).max();
                        segments.entry((*slot, key)).or_default().push((u, *reg, *op));
                    }
                }
                for ((_slot, _), group) in &segments {
                    if group.len() < 2 {
                        continue;
                    }
                    let (first, reg, load_op) = group[0];
                    let last = group.iter().map(|(u, _, _)| *u).max().expect("group");
                    // The carry is a copy right after the load, and the copy is
                    // lane-masked like the load: it fills every lane only when
                    // the load runs with the flags reset. Anything that can set
                    // a lane flag before it makes the carry partial, and a
                    // consumer that enables a lane this load did not would read
                    // a stale register instead of the tile.
                    let start = units[first].0 + 1;
                    let Some(pos) = ops[start..units[first].1].iter().position(|&o| o == load_op) else {
                        continue;
                    };
                    // Only the four flag opcodes can change the lane mask:
                    // `SFPLOAD`, `SFPSTORE` and every arithmetic op leave it
                    // alone (`tt_flag_effect` says `Keep`). Asking whether the
                    // mask *could* have been narrowed is the real question; a
                    // preceding load must not veto the carry, which is exactly
                    // what `exp2`'s second `nf` read paid a reload for.
                    let flags_touched = ops[start..start + pos].iter().any(|&o| tt_flag_effect(self.at(o)) != TtFlags::Keep);
                    if flags_touched {
                        continue;
                    }
                    segment_start.insert(load_op, (reg, last));
                }
                // Build the cone. If it needs more lane registers than exist,
                // do not shrink it: demote an interior value (the one whose
                // live range is longest) back to a tile round trip and build
                // again. The store is already in the body; demoting leaves it
                // there and lets the consumer load from the tile again, which
                // costs one store/load pair -- exactly the handwritten spill --
                // while keeping the whole chain in one row body. Shrinking is
                // the last resort, because it costs a separate replay.
                let mut demoted: Set<OpId> = Set::default();
                let built = loop {
                    // Virtualize. Every register a merged unit touches becomes a
                    // temporary name (`>= 16`) in a space wider than the ISA's
                    // 8-bit register field, so units cannot collide no matter
                    // how many values the whole cone defines; the pack below
                    // turns the temporaries into emitted names by liveness.
                    // Nothing here reasons about the eight-register file:
                    // `tt_lreg_alloc` assigns the physical ones after this
                    // pass. An interior value crosses on one `SFPMOV`, per row,
                    // between temporaries. A body is straight-line, so the
                    // merged body is one block and liveness there is exact.
                    let mut maps: Vec<Vec<u16>> = vec![vec![u16::MAX; 256]; n];
                    let mut live: Vec<(OpId, u16, usize)> = Vec::new(); // slot, name, last reader
                    // Every row of the merged body starts with the flags reset:
                    // a body's own preamble did exactly that once, and
                    // interleaving rows leaves the previous part's lane flags
                    // behind.
                    let mut total = gap_words + 1;
                    let spilled_words;
                    let mut carry_over: Vec<(OpId, u16, u16)> = Vec::new();
                    let mut hand: Vec<(OpId, u16, u16)> = Vec::new();
                    let mut cse_setup: Vec<(OpId, u16, u16)> = Vec::new(); // load, source, carry
                    // The values retired as register carries, as
                    // `(slot, producer unit, last reader unit)`; a demotion
                    // candidate is picked from here by live-range length.
                    let mut retired: Vec<(OpId, usize, usize)> = Vec::new();
                    let mut next = 16u16;
                    // The body as the allocator will see it, in temporary
                    // space: it is what decides whether this cone fits the
                    // eight registers.
                    let mut shadow: Vec<(Set<u16>, Set<u16>)> = Vec::new();
                    let mut shadow_flags: Vec<TtFlags> = Vec::new();
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
                            // (`LREG_ZERO` = 9, `LREG_ONE` = 10, ...): they
                            // appear in a body's read set and must never be
                            // renamed, which is why they are not part of the
                            // allocatable file. A body an earlier merge already
                            // virtualized is entitled to a fresh name too: its
                            // old one is unique, but the pack gives both bodies
                            // new, disjoint names anyway.
                            if (8..16).contains(&r) {
                                continue;
                            }
                            maps[u][usize::from(r)] = next;
                            next += 1;
                        }
                        // Hand each interior value to this unit, in a register;
                        // the first reader of a segment keeps its load, copies
                        // it into a carry at that position, and later readers
                        // of the segment are handed the carry.
                        for (slot, reg, _, op) in &loads[u] {
                            if let Some(&(_, src, _)) = live.iter().find(|(sl, _, _)| sl == slot) {
                                hand.push((*op, src, maps[u][usize::from(*reg)]));
                                continue;
                            }
                            if let Some(&(_first_reg, last)) = segment_start.get(op) {
                                // The load's own register is a body register and
                                // may be rewritten before the unit ends, so the
                                // carry is a copy at the load's position.
                                let carry = next;
                                next += 1;
                                cse_setup.push((*op, maps[u][usize::from(*reg)], carry));
                                live.retain(|(sl, _, _)| sl != slot);
                                live.push((*slot, carry, last));
                            }
                        }
                        // Retire stores whose value never leaves the cone. The
                        // register hand-over is format-agnostic (see above).
                        for (slot, reg, _fmt, op) in &stores[u] {
                            let interior = !demoted.contains(slot)
                                && producers.get(slot).is_some_and(|ps| ps.len() == 1 && ps[0] == u)
                                && readers.get(slot).is_some_and(|rs| rs.iter().all(|&r| r > u && r <= end));
                            if !interior {
                                continue;
                            }
                            let Some(last) = readers.get(slot).and_then(|rs| rs.iter().max()) else {
                                continue;
                            };
                            // The store becomes the copy that hands the value
                            // on, at the store's own position -- not "whatever
                            // is in the register at the end of the body", which
                            // a later write in the same unit would break.
                            let carry = next;
                            next += 1;
                            carry_over.push((*op, maps[u][usize::from(*reg)], carry));
                            live.push((*slot, carry, *last));
                            retired.push((*slot, u, *last));
                        }
                    }
                    // Each tile carry is one more `SFPMOV` word in the body.
                    total += cse_setup.len() as u32;
                    let ren = |set: &Set<u16>, map: &[u16]| -> Set<u16> {
                        set.iter()
                            .map(|&r| {
                                let to = map[usize::from(r)];
                                if to == u16::MAX { r } else { to }
                            })
                            .collect()
                    };
                    for u in s..=end {
                        // Every part starts with a flags reset in the emitted
                        // body (the fusion inserts one where the gap has none).
                        shadow_flags.push(TtFlags::Reset);
                        if u > s {
                            for &g in &ops[units[u - 1].1..units[u].0] {
                                if let Op::TT(TTOp::SfpConst(c)) = self.at(g) {
                                    let to = maps[u][usize::from(c.reg)];
                                    let to = if to == u16::MAX { u16::from(c.reg) } else { to };
                                    shadow.push((Set::default(), Set::from_iter([to])));
                                    shadow_flags.push(TtFlags::Keep);
                                }
                            }
                        }
                        for &op in &ops[units[u].0 + 1..units[u].1] {
                            if let Some(&(_, src, dst)) = hand.iter().find(|(o, _, _)| *o == op) {
                                shadow.push((Set::from_iter([src]), Set::from_iter([dst])));
                                shadow_flags.push(TtFlags::Keep);
                                continue;
                            }
                            if let Some(&(_, src, dst)) = carry_over.iter().find(|(o, _, _)| *o == op) {
                                shadow.push((Set::from_iter([src]), Set::from_iter([dst])));
                                shadow_flags.push(TtFlags::Keep);
                                continue;
                            }
                            let (reads, writes) = tt_row_regs(self.at(op));
                            shadow.push((ren(&reads, &maps[u]), ren(&writes, &maps[u])));
                            shadow_flags.push(tt_flag_effect(self.at(op)));
                            // The tile carry's copy sits at the load's position.
                            if let Some(&(_, src, carry)) = cse_setup.iter().find(|(o, _, _)| *o == op) {
                                shadow.push((Set::from_iter([src]), Set::from_iter([carry])));
                                shadow_flags.push(TtFlags::Keep);
                            }
                        }
                    }
                    // The temp names are already dense from 16 (every build
                    // restarts the counter). Do not reuse a name across
                    // disjoint intervals here: `tt_lreg_alloc` recomputes
                    // liveness from the emitted names, and a reused name would
                    // look like one long value, which both overstates pressure
                    // and makes spilling it a self-feeding loop.
                    if next > 256 {
                        break None;
                    }
                    if let Some(added) = tt_plan_splits(&shadow, &shadow_flags, 8, FUSE_SPILL_LIMIT) {
                        // The allocator will insert those stores/loads; they are
                        // budgeted apart from `total`, which must stay the
                        // emitted body's exact length.
                        spilled_words = added as u32;
                        break Some((maps, hand, carry_over, total, cse_setup, spilled_words));
                    }
                    // Demote the carried value whose spill is cheapest -- one
                    // store plus one load per reader -- and try the same cone
                    // again. A cone that needs many spills is not worth keeping:
                    // past a small budget, shrink instead. No candidate left:
                    // the pressure is in values the cone cannot give up.
                    if demoted.len() >= 2 {
                        break None;
                    }
                    let victim = retired
                        .iter()
                        .filter(|(slot, _, _)| !demoted.contains(slot))
                        .min_by_key(|(slot, _, _)| readers.get(slot).map_or(0, |rs| rs.len()))
                        .map(|(slot, _, _)| *slot);
                    let Some(victim) = victim else {
                        break None;
                    };
                    demoted.insert(victim);
                };
                let Some((maps, hand, carry_over, total, cse_setup, spilled_words)) = built else {
                    end -= 1;
                    continue;
                };
                let emitted = total + spilled_words;
                if emitted == 0 || emitted >= 1 << tt_isa::mailbox::loops::LEN_BITS {
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
                projected += usize::from(emitted > buffer);
                if projected > tt_isa::mailbox::loops::MAX {
                    end -= 1;
                    continue;
                }
                // The names are already packed (the spill plan had to see the
                // same intervals the allocator will); `pack` is part of `built`.
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
                // One map per unit: old register -> emitted name. `tt_row_remap`
                // looks each operand up once, so no rename can chain.
                let mut finals: Vec<Vec<u16>> = vec![vec![u16::MAX; 256]; n];
                for u in s..=end {
                    for r in 0..256usize {
                        finals[u][r] = maps[u][r];
                    }
                    // The unit's own body plus the gap of constants that now
                    // precedes it (the marker between them is not a row op).
                    let mut range: Vec<OpId> = ops[units[u].0 + 1..units[u].1].to_vec();
                    if u > s {
                        range.extend_from_slice(&ops[units[u - 1].1..units[u].0]);
                    }
                    for &op in &range {
                        // A hoisted unpack leaves the body; it has no lane
                        // registers to rename.
                        if hoist.contains(&op) {
                            continue;
                        }
                        tt_row_remap(&mut self.ops[op].op, &finals[u]);
                    }
                }
                for (load, from, to) in hand {
                    self.ops[load].op = Op::TT(TTOp::Sfp(TTSfpInsn { op: mov, args: [from, to, 0, 0, 0] }));
                }
                for (store, from, to) in carry_over {
                    self.ops[store].op = Op::TT(TTOp::Sfp(TTSfpInsn { op: mov, args: [from, to, 0, 0, 0] }));
                }
                for (load, src, carry) in cse_setup {
                    self.insert_after(load, Op::TT(TTOp::Sfp(TTSfpInsn { op: mov, args: [src, carry, 0, 0, 0] })));
                }
                // The hoisted unpacks run once per tile, in front of the row
                // body -- which is where their readers (now inside one body)
                // need them, and the role hand-off is the same seam a kernel
                // input already gets.
                for &op in &hoist {
                    let copy = self.ops[op].op.clone();
                    self.remove_op(op);
                    self.insert_before(marker, copy);
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
        // Spill tiles go in the prologue, like every other logical Register
        // tile: `tt_dst_reuse` later assigns them a physical row.
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
        let dtype = {
            let mut id = self.head;
            let mut found = None;
            for _ in 0..250_000 {
                if id.is_null() {
                    break;
                }
                if let Op::Storage { dtype, scope: MemScope::Register, .. } = self.ops[id].op {
                    found = Some(dtype);
                    break;
                }
                id = self.next_op(id);
            }
            found.unwrap_or(DType::F32)
        };
        let mut id = self.head;
        while !id.is_null() {
            let Op::TT(TTOp::Replay { len, count }) = self.at(id) else {
                id = self.next_op(id);
                continue;
            };
            let (marker, count) = (id, *count);
            let len = u32::from(*len);
            let mut body: Vec<OpId> = Vec::new();
            let mut words = 0u32;
            let mut cur = self.next_op(marker);
            while !cur.is_null() && words < len {
                words += tt_row_words(self.at(cur));
                body.push(cur);
                cur = self.next_op(cur);
            }
            // Allocate with reuse; wherever the body still needs more than the
            // register budget, split one value around that over-budget run
            // (store before it, reload after it) and allocate again, until the
            // body fits. One pair per run, not one reload per use.
            let budget = if dtype.bit_size() > 16 { 8 } else { 16 };
            let mut next = body
                .iter()
                .flat_map(|&op| {
                    let (r, w) = tt_row_regs(self.at(op));
                    r.into_iter().chain(w)
                })
                .filter(|r| *r >= 16)
                .max()
                .map_or(16, |m| m + 1);
            let mut splits = 0usize;
            let assign = loop {
                let regs: Vec<(Set<u16>, Set<u16>)> = body.iter().map(|&op| tt_row_regs(self.at(op))).collect();
                let effects: Vec<TtFlags> = body.iter().map(|&op| tt_flag_effect(self.at(op))).collect();
                let (before, after) = tt_mask_states(&effects);
                let Some((name, a, b)) = tt_split_at_pressure(&regs, &before, &after, budget) else {
                    break tt_assign_regs(&regs).unwrap_or_else(|| {
                        todo!(
                            "tt_lreg_alloc: body of {} ops still does not fit {budget} registers after {splits} splits",
                            body.len()
                        )
                    });
                };
                splits += 1;
                if splits > ASSIGN_SPILL_LIMIT {
                    todo!("tt_lreg_alloc: split budget exhausted after {splits} splits");
                }
                let fresh = next;
                next += 1;
                if fresh > 255 {
                    todo!("tt_lreg_alloc: out of register names for reloads");
                }
                let tile = self.tt_new_dst(dtype, prologue);
                let store = self.insert_before(
                    body[a],
                    Op::TT(TTOp::SfpStore(Box::new(TTSfpStore { slot: tile, at: 0, reg: name as u8, fmt: INT32 }))),
                );
                let load = self.insert_after(
                    body[b],
                    Op::TT(TTOp::SfpLoad(Box::new(TTSfpLoad { slot: tile, at: 0, reg: fresh as u8, fmt: INT32 }))),
                );
                for &op in &body[b + 1..] {
                    tt_row_rename(&mut self.ops[op].op, name as u8, fresh as u8);
                }
                let mut grown: Vec<OpId> = Vec::with_capacity(body.len() + 2);
                grown.extend_from_slice(&body[..a]);
                grown.push(store);
                grown.extend_from_slice(&body[a..=b]);
                grown.push(load);
                grown.extend_from_slice(&body[b + 1..]);
                body = grown;
            };
            if assign.is_empty() {
                id = cur;
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
            id = cur;
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
        let mut loop_of: Map<usize, (usize, usize)> = Map::default();
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
        // An `Op::Loop` runs its body `trip` times, so an access inside it is
        // one hand-off per iteration, and a consumer *after* the loop must
        // wait for the last iteration, not the first: map a member op to the
        // loop's bounds exactly as a `Replay` body is mapped. Without this the
        // consumer's `Wait` came before the loop and the reader raced it.
        {
            let mut open: Vec<(usize, usize)> = Vec::new(); // (loop index, body start)
            for i in 0..ops.len() {
                match self.at(ops[i]) {
                    Op::Loop { .. } => open.push((i, i + 1)),
                    Op::EndLoop => {
                        if let Some((start, body)) = open.pop() {
                            for k in body..i {
                                loop_of.insert(k, (start, i));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        let before_replay = |i: usize| replay_of.get(&i).map(|(start, _)| *start).unwrap_or(i);
        let after_replay = |i: usize| replay_of.get(&i).map(|(_, end)| end - 1).unwrap_or(i);
        // A hand-off between two accesses of the *same* loop body is one per
        // iteration and stays where it is. A hand-off whose other end is
        // outside the loop happens once for the whole loop, so it belongs
        // outside: the post after `EndLoop`, the take before the `Loop`. (The
        // final accumulator->pack edge is the one that needs this; a
        // per-iteration tile hand-off must not be hoisted, or the reader sees
        // only the last iteration's tile.)
        let loop_end = |writer: usize, reader: usize| -> usize {
            let w = after_replay(writer);
            match (loop_of.get(&w), loop_of.get(&after_replay(reader))) {
                (Some(&(ws, _)), Some(&(rs, _))) if ws == rs => w,
                (Some(&(_, we)), _) => we,
                _ => w,
            }
        };
        let loop_begin = |writer: usize, reader: usize| -> usize {
            let r = before_replay(reader);
            match (loop_of.get(&r), loop_of.get(&before_replay(writer))) {
                (Some(&(rs, _)), Some(&(ws, _))) if ws == rs => r,
                (Some(&(rs, _)), _) => rs,
                _ => r,
            }
        };
        let before = |role: u8| match role {
            0 => TT_BEFORE_UNPACKER,
            2 => TT_BEFORE_PACKER,
            _ => TT_BEFORE_ALL,
        };
        // A seam names the ordered `(producer, consumer)` edge it carries; the
        // semaphore index is assigned once all edges are known.
        let mut posts: Vec<(usize, (u8, u8))> = Vec::new();
        // Edges whose writer waits for the reader (a loop reuse), counted per
        // loop so the semaphore starts with one token per hand-off the body
        // performs. One token per *edge* is only enough when the body hands
        // off one tile: with two, the writer's second take waits for a post
        // that the reader can only make after its own second wait, which the
        // writer's second post gates (measured as a three-role timeout on
        // softmax, whose loop re-unpacks the loop-invariant max vector next to
        // the input tile).
        let mut reuse: Map<((u8, u8), usize), u8> = Map::default();

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
                    posts.push((loop_end(*write_idx, first), (*writer, reader)));
                    takes.push((loop_begin(*write_idx, first), (*writer, reader)));
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
                // A loop's per-iteration pacing: one write in the IR is `trip`
                // writes at run time, and the writer must not overwrite the
                // tile until the reader has taken the previous iteration's.
                // Only when both ends are in the *same* loop body: a reader
                // after the loop reads the last iteration once and the writer
                // never rewrites, so its edge is the hoisted one above (a take
                // for it inside the loop waits on a post that only comes after
                // the loop, which deadlocks).
                let same_loop = |a: usize, b: usize| {
                    matches!(
                        (loop_of.get(&a), loop_of.get(&b)),
                        (Some(&(x, _)), Some(&(y, _))) if x == y
                    )
                };
                for &(reader, _, last) in readers {
                    if reader != *writer && same_loop(*write_idx, last) {
                        posts.push((after_replay(last), (reader, *writer)));
                        takes.push((before_replay(*write_idx), (reader, *writer)));
                        let body = loop_of.get(&after_replay(*write_idx)).map(|(start, _)| *start).unwrap_or(usize::MAX);
                        *reuse.entry(((reader, *writer), body)).or_default() += 1;
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
            // A reuse edge starts with one token per hand-off its loop body
            // performs: the first iteration's writes have nothing to wait for.
            // An edge used by two different loops is sized by the busier one;
            // tokens are only ever returned by a post that follows a read.
            let value =
                reuse.iter().filter(|((edge, _), _)| *edge == edges[sem as usize]).map(|(_, count)| *count).max().unwrap_or(0);
            assert!(
                value <= tt_isa::sync::MAX_VALUE,
                "tt_reg_sync: edge {edge:?} performs {value} loop hand-offs per iteration, more than the {} a semaphore can start with",
                tt_isa::sync::MAX_VALUE,
                edge = edges[sem as usize]
            );
            self.insert_after(end_reader, Op::TT(TTOp::Semset { sem, value }));
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
        if std::env::var_os("ZYX_SYNC_DUMP").is_some() {
            let mut id = self.head;
            let mut n = 0;
            while !id.is_null() && n < 4000 {
                match self.at(id) {
                    Op::Loop { .. }
                    | Op::EndLoop
                    | Op::TT(TTOp::Post { .. })
                    | Op::TT(TTOp::Wait { .. })
                    | Op::TT(TTOp::WaitBelowMax { .. })
                    | Op::TT(TTOp::Semset { .. }) => eprintln!("S{n} {id:?}: {:?}", self.at(id)),
                    _ => {}
                }
                id = self.next_op(id);
                n += 1;
            }
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
            // The DST file is 8 rows (16 for 16-bit tiles). Allocating past it
            // would write outside the file: a compile error, not silent extra
            // state.
            let budget = if dtype.bit_size() > 16 { 8 } else { 16 };
            let rows = physical.values().copied().collect::<Set<OpId>>().len();
            assert!(
                rows < budget,
                "tt_dst_reuse: the register file is full ({rows} of {budget} rows) and {logical:?} needs another"
            );
            self.tt_new_dst(dtype, prologue)
        };
        physical.insert(logical, phys);
        active.push((logical, phys, last_use.get(&logical).copied().unwrap_or(0)));
    }

    /// True when an index already names **physical tiles**: it belongs to the
    /// reduce template `tt_reindex` rewrote — a reduce over more than one
    /// tile, so a grid axis **and** the reduce's tile loop are both present,
    /// and every grid axis advances a whole number of 1024-element tiles.
    ///
    /// The loop is the discriminator: an elementwise index never has one, and
    /// its logical grid terms are rebased from the group extents instead
    /// (`strip_unrolls`, `tt_slice`).
    pub(crate) fn tt_index_is_tile_units(&self, idx: OpId) -> bool {
        let Some(strides) = self.get_strides(idx) else { return false };
        let (mut grid, mut looped) = (false, false);
        for (var, (_, stride)) in strides.iter() {
            if *var == OpId::NULL {
                continue;
            }
            match self.ops[*var].op {
                Op::Range { kind: RangeKind::Group(_), .. } => {
                    grid = true;
                    if *stride <= 0 || *stride % 1024 != 0 {
                        return false;
                    }
                }
                Op::Loop { .. } => looped = true,
                _ => {}
            }
        }
        grid && looped
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
    fn strip_unrolls(&mut self, idx: OpId, before: OpId, strip_loops: bool) -> Option<OpId> {
        // Clone memo per call (index math is small).
        let mut memo: Map<OpId, OpId> = Map::default();
        // A reindexed index already names tiles; its group terms are the
        // physical offsets and must not be rebased again.
        let tile_units = self.tt_index_is_tile_units(idx);
        // The group axes **this index names**. A parameter's tile pitch only
        // counts the extents of the grid axes that index it: the `[32]` row
        // vector of `exp(x - max)` is addressed by the row group alone, so its
        // stride is one tile even though the kernel has a second (column)
        // group axis.
        let index_axes: Option<Set<u32>> = self.get_strides(idx).map(|s| {
            s.keys()
                .filter_map(|v| {
                    if v.is_null() {
                        return None;
                    }
                    match self.ops[*v].op {
                        Op::Range { axis, kind: RangeKind::Group(_), .. } => Some(axis),
                        _ => None,
                    }
                })
                .collect()
        });
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
                // A loop var is a scalar leaf. It stays in a stripped index
                // that the emitter expands per iteration; a hoisted block
                // transport zeroes it (the block starts at its base).
                Op::Const(_) | Op::Range { .. } | Op::Param { .. } => {
                    memo.insert(n, n);
                }
                Op::Loop { .. } => {
                    memo.insert(n, if strip_loops { zero } else { n });
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
                        if !tile_units && (gx.is_some() || gy.is_some()) && strides.is_none() {
                            let mut axes: Vec<(u32, Dim)> = Vec::new();
                            for (_, op) in self.iter_unordered() {
                                if let Op::Range { axis, kind: RangeKind::Group(len) } = op {
                                    if index_axes.as_ref().is_some_and(|set| !set.contains(axis)) {
                                        continue;
                                    }
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
                        if !tile_units {
                            match bop {
                                BOp::Mul => {
                                    // `c * g` -> `g * stride`: the group steps
                                    // whole tiles, so the element coefficient
                                    // `c` is *replaced* by the physical stride.
                                    // Keeping it multiplies the live range twice
                                    // (softmax's `[32]` max row: `32 * (1024*g)`).
                                    if let Some(a) = gx {
                                        if gy.is_none() && matches!(self.ops[ny].op, Op::Const(_)) {
                                            let s = strides.as_ref().expect("strip_unrolls: stride map missing")[&a];
                                            ay = self.insert_before(before, Op::Const(Constant::idx(s)));
                                        }
                                    }
                                    if let Some(a) = gy {
                                        if gx.is_none() && matches!(self.ops[nx].op, Op::Const(_)) {
                                            let s = strides.as_ref().expect("strip_unrolls: stride map missing")[&a];
                                            ax = self.insert_before(before, Op::Const(Constant::idx(s)));
                                        }
                                    }
                                    // A const times a **group-derived**
                                    // expression is the logical row stride
                                    // (`(32*g0 + lane) * 256`): the physical
                                    // stride the group just took already carries
                                    // the inner extents, so the multiplier is
                                    // dropped rather than applied a second time.
                                    if gx.is_none() && gy.is_none() {
                                        let (keep, other) = if matches!(self.ops[ny].op, Op::Const(_)) {
                                            (nx, ny)
                                        } else {
                                            (ny, nx)
                                        };
                                        if matches!(self.ops[other].op, Op::Const(_))
                                            && self.get_strides(keep).is_some_and(|s| {
                                                s.keys().any(|v| {
                                                    !v.is_null()
                                                        && matches!(self.ops[*v].op, Op::Range { kind: RangeKind::Group(_), .. })
                                                })
                                            })
                                        {
                                            memo.insert(n, keep);
                                            continue;
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
                Op::TT(TTOp::Replay { len, count, .. }) => {
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
        if !seen_reader {
            eprintln!("UNORDERED ({} ops):", self.iter_unordered().count());
            for (id, op) in self.iter_unordered() {
                eprintln!("OP {id:?}: {op:?}");
            }
            eprintln!("CHAIN:");
            let mut d = self.head;
            let mut n = 0;
            while !d.is_null() && n < 400 {
                eprintln!("C{n} {d:?}: {:?}", self.at(d));
                d = self.next_op(d);
                n += 1;
            }
        }
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
    use crate::codegen::tenstorrent::sfpu::interp::Vector;
    use crate::codegen::tenstorrent::sfpu::kernel::{A_ROW, B_ROW, OUT_ROW};
    use crate::codegen::tenstorrent::sfpu::ops::{self, kind_sfpu as sf};
    use crate::kernel::Dev;
    use std::collections::HashMap;
    use tt_isa::isa::Instruction;

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
            crate::codegen::tenstorrent::kind::ADD,
            crate::codegen::tenstorrent::kind::SUB,
            crate::codegen::tenstorrent::kind::MUL,
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
            crate::codegen::tenstorrent::kind::ADD,
            crate::codegen::tenstorrent::kind::SUB,
            crate::codegen::tenstorrent::kind::MUL,
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
    use crate::codegen::tenstorrent::sfpu::interp::Vector;
    use crate::codegen::tenstorrent::sfpu::kernel::{A_ROW, B_ROW, OUT_ROW};
    use crate::kernel::Dev;
    use std::collections::HashMap;
    use tt_isa::isa::Instruction;

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

#[cfg(test)]
mod lreg_spill {
    use super::*;

    fn body_that_fits() -> Vec<(Set<u16>, Set<u16>)> {
        vec![
            (Set::default(), Set::from_iter([16])),
            (Set::from_iter([16]), Set::from_iter([17])),
            (Set::from_iter([17]), Set::default()),
        ]
    }

    /// Nine live at once, with the *last* definition used first and the first
    /// definition used last: spilling the farthest-ending value (16) leaves the
    /// reload for a point where everything else is dead, so one spill is enough.
    fn body_that_spills_once() -> Vec<(Set<u16>, Set<u16>)> {
        let mut body: Vec<(Set<u16>, Set<u16>)> = Vec::new();
        for name in 16..=24u16 {
            body.push((Set::default(), Set::from_iter([name])));
        }
        for name in (16..=24u16).rev() {
            body.push((Set::from_iter([name]), Set::default()));
        }
        body
    }

    #[test]
    fn a_body_that_fits_needs_no_spill() {
        let (assign, spills) = tt_alloc_spill(&body_that_fits(), 8).expect("fits");
        assert!(spills.is_empty());
        assert_eq!(assign.get(&16), Some(&0));
        assert_ne!(assign.get(&16), assign.get(&17), "two values cannot share a register");
    }

    #[test]
    fn nine_live_values_spill_the_one_that_ends_last() {
        let (_, spills) = tt_alloc_spill(&body_that_spills_once(), 8).expect("one spill fits the cap");
        assert_eq!(spills, vec![16], "the value used last is the one that goes to memory");
    }

    #[test]
    fn planning_materializes_a_spill_and_converges() {
        let (words, names) = tt_plan_spills(&body_that_spills_once(), 4).expect("one spill is enough");
        assert_eq!(names, vec![16]);
        assert_eq!(words, 2, "one store at the definition and one reload at the use");
    }

    #[test]
    fn the_spill_cap_is_a_loud_refusal() {
        assert!(tt_alloc_spill(&body_that_spills_once(), 0).is_none());
    }
}

// ── Circular-buffer transactions ─────────────────────────────────────────────
//
// Ported from `zyx/src/codegen/tenstorrent.rs` at `588110a9` ("multitile cbs
// continuation"), where this was the state behind the tt-metal
// `reserve_back` / `push_back` / `wait_front` / `pop_front` calls. The old
// emitter also chose the batches (which traffic op becomes a block spanning a
// loop) and anchored the events; this is the machine those events drive, kept
// separate so it is testable without emitting anything.

/// One circular buffer's state in a pass over the emitted traffic.
///
/// A CB is a FIFO of tiles: a producer `reserve`s a block, writes every tile,
/// then `push`es it; a consumer `wait`s for a block, reads every tile, then
/// `pop`s it. The machine follows **one** pass over the traffic text: a loop
/// body runs that text once per iteration on the device, so a transaction
/// that spans a loop is opened before the `Loop` and closed after its
/// `EndLoop`.
///
/// `avail` is the ring occupancy, tiles pushed minus tiles popped. It
/// persists across transactions so a reader can run ahead of its consumer;
/// `Popped` is exactly `avail == 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(dead_code, reason = "the batching pre-pass wires this in next")]
pub(crate) enum CbState {
    /// Free: no live transaction, no unread tiles.
    #[default]
    Popped,
    /// `reserve_back(n)` emitted; `filled` of the block's tiles written.
    Reserved { n: u32, filled: u32, avail: u32 },
    /// `avail` tiles sit unconsumed at the front.
    Pushed { avail: u32 },
    /// `wait_front(m)` emitted; `filled` of the `m` waited tiles read.
    Waiting { waited: u32, filled: u32, avail: u32 },
}

#[allow(dead_code, reason = "the batching pre-pass wires this in next")]
impl CbState {
    /// `reserve_back(n)`: from the free state, or stacked on unread tiles
    /// (`Pushed` — the ring wraps, the producer runs ahead).
    #[track_caller]
    pub(crate) fn reserve(self, cb: OpId, n: u32) -> Self {
        let avail = match self {
            Self::Popped => 0,
            Self::Pushed { avail } => avail,
            s => panic!("CB{cb:?}: reserve_back in {s:?}, must be free or pushed"),
        };
        assert!(n > 0, "CB{cb:?}: reserve_back(0) is not a transaction");
        Self::Reserved { n, filled: 0, avail }
    }

    /// `push_back(n)`: closes the open reservation; every tile must have
    /// been written (`moved` accounted them).
    #[track_caller]
    pub(crate) fn push(self, cb: OpId, n: u32) -> Self {
        match self {
            Self::Reserved { n: rn, filled, avail } if rn == n => {
                assert_eq!(filled, n, "CB{cb:?}: push_back with {filled}/{n} tiles written");
                Self::Pushed { avail: avail + n }
            }
            s => panic!("CB{cb:?}: push_back in {s:?}, must be Reserved with matching block size"),
        }
    }

    /// `wait_front(m)`: waits for `m` unread tiles, which must be available.
    #[track_caller]
    pub(crate) fn wait(self, cb: OpId, m: u32) -> Self {
        match self {
            Self::Pushed { avail } => {
                assert!(m <= avail, "CB{cb:?}: wait_front({m}) with only {avail} tiles available");
                assert!(m > 0, "CB{cb:?}: wait_front(0) is not a transaction");
                Self::Waiting { waited: m, filled: 0, avail }
            }
            s => panic!("CB{cb:?}: wait_front in {s:?}, must be Pushed"),
        }
    }

    /// `pop_front(m)`: drains the waited block. Tiles left over wrap back to
    /// `Pushed`; an empty front wraps to free.
    #[track_caller]
    pub(crate) fn pop(self, cb: OpId, m: u32) -> Self {
        match self {
            Self::Waiting { waited, filled, avail } if waited == m => {
                assert_eq!(filled, m, "CB{cb:?}: pop_front with {filled}/{m} tiles read");
                let rest = avail - m;
                if rest == 0 {
                    Self::Popped
                } else {
                    Self::Pushed { avail: rest }
                }
            }
            s => panic!("CB{cb:?}: pop_front in {s:?}, must be Waiting with matching block size"),
        }
    }

    /// Accounts the tiles one execution of a data-movement text moves: fills
    /// the open reservation (producer) or drains the waited block (consumer).
    #[track_caller]
    pub(crate) fn moved(self, cb: OpId, count: u32) -> Self {
        match self {
            Self::Reserved { n, filled, avail } => {
                let filled = filled + count;
                assert!(filled <= n, "CB{cb:?}: block overfill {filled}/{n}");
                Self::Reserved { n, filled, avail }
            }
            Self::Waiting { waited, filled, avail } => {
                let filled = filled + count;
                assert!(filled <= waited, "CB{cb:?}: block overdrain {filled}/{waited}");
                Self::Waiting { waited, filled, avail }
            }
            s => panic!("CB{cb:?}: a move in {s:?}, must be Reserved or Waiting"),
        }
    }

    /// A section may end only between transactions (`Popped`/`Pushed`); a
    /// `Reserved`/`Waiting` remainder is a leaked transaction.
    pub(crate) fn settled(self) -> bool {
        matches!(self, Self::Popped | Self::Pushed { .. })
    }
}

/// Every circular buffer's state in one kernel, keyed by its `Circular`
/// storage. One entry per buffer, created on first touch.
#[derive(Debug, Default)]
#[allow(dead_code, reason = "the batching pre-pass wires this in next")]
pub(crate) struct CbTransactions {
    states: Map<OpId, CbState>,
}

#[allow(dead_code, reason = "the batching pre-pass wires this in next")]
impl CbTransactions {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The current state, or `Popped` for a buffer not touched yet.
    pub(crate) fn state(&self, cb: OpId) -> CbState {
        self.states.get(&cb).copied().unwrap_or_default()
    }

    pub(crate) fn set(&mut self, cb: OpId, state: CbState) {
        self.states.insert(cb, state);
    }

    pub(crate) fn reserve(&mut self, cb: OpId, n: u32) {
        let s = self.state(cb).reserve(cb, n);
        self.set(cb, s);
    }

    pub(crate) fn push(&mut self, cb: OpId, n: u32) {
        let s = self.state(cb).push(cb, n);
        self.set(cb, s);
    }

    pub(crate) fn wait(&mut self, cb: OpId, m: u32) {
        let s = self.state(cb).wait(cb, m);
        self.set(cb, s);
    }

    pub(crate) fn pop(&mut self, cb: OpId, m: u32) {
        let s = self.state(cb).pop(cb, m);
        self.set(cb, s);
    }

    /// Accounts `count` moved tiles against the open transaction on `cb`.
    pub(crate) fn moved(&mut self, cb: OpId, count: u32) {
        let s = self.state(cb).moved(cb, count);
        self.set(cb, s);
    }

    /// End-of-section gate: no buffer may hold a mid-transaction state.
    #[track_caller]
    pub(crate) fn assert_settled(&self, section: &str) {
        for (&cb, &state) in self.states.iter() {
            assert!(state.settled(), "{section} ends with CB{cb:?} in {state:?}; a transaction leaked");
        }
    }
}

/// Which side of a CB a traffic op moves tiles on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CbTraffic {
    /// Into the buffer (reader publish, compute pack).
    Produce,
    /// Out of the buffer (compute unpack, writer drain).
    Consume,
}

/// One traffic op's transaction.
///
/// `per_op` is how many tiles one execution of the op's movement text moves.
/// `hoist` is the `Op::Loop` whose open/close brackets the block; for a
/// single-tile transaction it is the traffic op itself, which opens and
/// closes in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CbBatch {
    pub(crate) cb: OpId,
    pub(crate) per_op: u32,
    pub(crate) hoist: OpId,
    pub(crate) kind: CbTraffic,
}

/// An open or close event, anchored at an op. A producer reserves then
/// pushes; a consumer waits then pops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CbEvent {
    Produce { cb: OpId, n: u32 },
    Consume { cb: OpId, n: u32 },
}

impl CbEvent {
    fn cb(self) -> OpId {
        match self {
            Self::Produce { cb, .. } | Self::Consume { cb, .. } => cb,
        }
    }
}

/// The CB transactions of one kernel pass: which traffic op is a block, where
/// its open/close events anchor, and the per-pass tile totals.
#[derive(Debug, Default)]
#[allow(dead_code, reason = "the emitter wires this in next")]
pub(crate) struct CbPlan {
    batches: Map<OpId, CbBatch>,
    /// Open events, emitted before the anchored op's text.
    opens: Map<OpId, Vec<CbEvent>>,
    /// Close events, emitted after the anchored op's text.
    closes: Map<OpId, Vec<CbEvent>>,
    produced: Map<OpId, u32>,
    consumed: Map<OpId, u32>,
}

#[allow(dead_code, reason = "the emitter wires this in next")]
impl CbPlan {
    pub(crate) fn batch(&self, op: OpId) -> Option<CbBatch> {
        self.batches.get(&op).copied()
    }

    pub(crate) fn opens(&self, op: OpId) -> &[CbEvent] {
        self.opens.get(&op).map_or(&[], Vec::as_slice)
    }

    pub(crate) fn closes(&self, op: OpId) -> &[CbEvent] {
        self.closes.get(&op).map_or(&[], Vec::as_slice)
    }

    pub(crate) fn produced(&self, cb: OpId) -> u32 {
        self.produced.get(&cb).copied().unwrap_or(0)
    }

    pub(crate) fn consumed(&self, cb: OpId) -> u32 {
        self.consumed.get(&cb).copied().unwrap_or(0)
    }
}

/// The three traffic sections of a lowered TT kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CbSection {
    Reader,
    Compute,
    Writer,
}

/// One enclosing loop of the batching walk.
struct CbLoopFrame {
    op: OpId,
    trip: u32,
    /// CB -> the (last) traffic op in this frame moving tiles on it.
    traffic: Map<OpId, OpId>,
    /// CBs with more than one traffic op in this frame: no hoist.
    dirty: Set<OpId>,
}

/// True if the operand closure of `root` contains `target` (the loop counter
/// appearing in a tile-slot index).
#[allow(dead_code, reason = "the emitter wires this in next")]
fn deps_contain(kernel: &Kernel, root: OpId, target: OpId) -> bool {
    if root.is_null() {
        return false;
    }
    let mut stack = vec![root];
    for _ in 0..10_000 {
        let Some(id) = stack.pop() else { return false };
        if id == target {
            return true;
        }
        stack.extend(kernel.ops[id].op.parameters());
    }
    panic!("tt_cb_plan: deps_contain did not finish in 10000 steps");
}

#[allow(dead_code, reason = "the emitter wires this in next")]
impl Kernel {
    /// Group the lowered traffic into CB transactions.
    ///
    /// Walks the reader/compute/writer stream once. A traffic op is any
    /// `TTOp::Copy` touching a `Circular` storage (its `dst` = produce, its
    /// `src` = consume). Every such op starts as a single-tile transaction; at
    /// a loop's `EndLoop` it is upgraded to a block over the whole loop when
    /// the loop is its only traffic on that CB, the trip count is constant and
    /// fits the CB, and the CB-side index is the loop counter or 0.
    ///
    /// The pass also totals the tiles produced and consumed per CB and
    /// refuses a kernel where they do not balance. A tile count that does not
    /// fold to a constant is a compile error: the block is a compile-time
    /// quantity.
    pub(crate) fn tt_cb_plan(&self) -> CbPlan {
        let mut plan = CbPlan::default();
        let mut section = CbSection::Reader;
        let mut stack: Vec<CbLoopFrame> = Vec::new();
        let mut loop_end_of: Map<OpId, OpId> = Map::default();
        // Traffic op -> its CB-side index (the tile-slot operand).
        let mut traffic_index: Map<OpId, OpId> = Map::default();
        let mut scan = self.head;
        for _ in 0..250_000 {
            if scan.is_null() {
                break;
            }
            match &self.ops[scan].op {
                Op::TT(TTOp::EndReader) => {
                    assert!(stack.is_empty(), "tt_cb_plan: a loop crosses EndReader");
                    section = CbSection::Compute;
                }
                Op::TT(TTOp::EndCompute) => {
                    assert!(stack.is_empty(), "tt_cb_plan: a loop crosses EndCompute");
                    section = CbSection::Writer;
                }
                Op::Loop { len } => {
                    let trip = self.resolve_const(*len).and_then(Constant::as_dim).unwrap_or_else(|| {
                        panic!("tt_cb_plan: loop {scan:?} has a non-constant trip count; CB tile counts are compile-time")
                    });
                    let trip = u32::try_from(trip)
                        .unwrap_or_else(|_| panic!("tt_cb_plan: loop {scan:?} has negative trip count {trip}"));
                    stack.push(CbLoopFrame { op: scan, trip, traffic: Map::default(), dirty: Set::default() });
                }
                Op::EndLoop => {
                    let frame = stack.pop().expect("tt_cb_plan: EndLoop without Loop");
                    loop_end_of.insert(frame.op, scan);
                    for (&cb, &op) in frame.traffic.iter() {
                        if frame.dirty.contains(&cb) || frame.trip <= 1 {
                            continue;
                        }
                        let Some(tiles) = self.cb_tiles(cb) else { continue };
                        if frame.trip > tiles {
                            // The CB cannot hold the whole block: one tile per
                            // loop pass (streaming), which is always correct.
                            continue;
                        }
                        let index = traffic_index[&op];
                        if !deps_contain(self, index, frame.op) && !self.is_zero_index(index) {
                            panic!("tt_cb_plan: CB{cb:?} block index {index:?} is neither the loop {scan:?} counter nor 0");
                        }
                        let kind = plan.batches[&op].kind;
                        plan.batches.insert(op, CbBatch { cb, per_op: frame.trip, hoist: frame.op, kind });
                    }
                }
                Op::TT(TTOp::Copy { src, dst, src_idx, dst_idx, .. }) => {
                    let src_cb = self.is_circular(*src);
                    let dst_cb = self.is_circular(*dst);
                    let (kind, cb, index) = match (src_cb, dst_cb) {
                        (false, true) => (CbTraffic::Produce, *dst, *dst_idx),
                        (true, false) => (CbTraffic::Consume, *src, *src_idx),
                        (true, true) => panic!("tt_cb_plan: copy {scan:?} moves between two CBs"),
                        // Register traffic (DST) is not a CB transaction.
                        (false, false) => {
                            scan = self.next_op(scan);
                            continue;
                        }
                    };
                    let sect_ok = match (section, kind) {
                        (CbSection::Reader, CbTraffic::Produce)
                        | (CbSection::Compute, CbTraffic::Produce | CbTraffic::Consume)
                        | (CbSection::Writer, CbTraffic::Consume) => true,
                        _ => false,
                    };
                    assert!(sect_ok, "tt_cb_plan: {kind:?} on CB{cb:?} at {scan:?} is not in a valid section ({section:?})");
                    // Per-pass count: the product of every enclosing trip.
                    let mut count = 1u32;
                    for f in &stack {
                        count =
                            count.checked_mul(f.trip).unwrap_or_else(|| panic!("tt_cb_plan: CB{cb:?} tile count overflows u32"));
                    }
                    match kind {
                        CbTraffic::Produce => *plan.produced.entry(cb).or_insert(0) += count,
                        CbTraffic::Consume => *plan.consumed.entry(cb).or_insert(0) += count,
                    }
                    plan.batches.insert(scan, CbBatch { cb, per_op: 1, hoist: scan, kind });
                    traffic_index.insert(scan, index);
                    if let Some(f) = stack.last_mut() {
                        if f.traffic.insert(cb, scan).is_some() {
                            f.dirty.insert(cb);
                        }
                    }
                }
                _ => {}
            }
            scan = self.next_op(scan);
        }
        assert!(scan.is_null(), "tt_cb_plan: the walk did not finish in 250000 steps");
        assert!(stack.is_empty(), "tt_cb_plan: unclosed loop at the kernel end");
        // A single-tile transaction reuses slot 0; a larger block was either
        // hoisted or is streaming one tile at a time. Anything else would
        // silently address a slot the ring does not have.
        for (&op, &b) in plan.batches.iter() {
            if b.hoist == op && !self.is_zero_index(traffic_index[&op]) {
                panic!("tt_cb_plan: single-tile transaction {op:?} on CB{:?} needs slot 0", b.cb);
            }
        }
        // Producer/consumer correspondence, per pass, per CB.
        for (&cb, &p) in plan.produced.iter() {
            let c = plan.consumed.get(&cb).copied().unwrap_or(0);
            assert_eq!(p, c, "tt_cb_plan: CB{cb:?} produces {p} but consumes {c} tiles per pass");
        }
        for (&cb, &c) in plan.consumed.iter() {
            assert!(plan.produced.contains_key(&cb), "tt_cb_plan: CB{cb:?} is consumed but never produced");
            assert_eq!(c, plan.produced[&cb], "tt_cb_plan: CB{cb:?} consumption has no producer");
        }
        // Anchor the events: a single-tile transaction opens and closes at
        // its op; a hoisted block opens at the loop and closes at its
        // EndLoop. Sorted by CB for determinism.
        let mut anchored: Vec<(OpId, OpId, CbEvent)> = Vec::new();
        for (&op, &b) in plan.batches.iter() {
            let event = match b.kind {
                CbTraffic::Produce => CbEvent::Produce { cb: b.cb, n: b.per_op },
                CbTraffic::Consume => CbEvent::Consume { cb: b.cb, n: b.per_op },
            };
            let (open, close) = if b.hoist == op {
                (op, op)
            } else {
                (b.hoist, loop_end_of[&b.hoist])
            };
            anchored.push((open, close, event));
        }
        anchored.sort_by_key(|&(open, close, event)| (open, close, event.cb()));
        for (open, close, event) in anchored {
            plan.opens.entry(open).or_default().push(event);
            plan.closes.entry(close).or_default().push(event);
        }
        plan
    }

    /// Tiles a `Circular` storage holds, or `None` if `cb` is not one.
    fn cb_tiles(&self, cb: OpId) -> Option<u32> {
        match self.ops[cb].op {
            Op::Storage { scope: MemScope::Circular, len, .. } => {
                assert_eq!(len % 1024, 0, "tt_cb_plan: CB {cb:?} holds {len} elements, not whole tiles");
                u32::try_from(len / 1024).ok()
            }
            _ => None,
        }
    }

    fn is_circular(&self, op: OpId) -> bool {
        matches!(self.ops[op].op, Op::Storage { scope: MemScope::Circular, .. })
    }

    /// True if `idx` resolves to the constant 0 (the slot a single-tile
    /// transaction and a block's first slot use).
    fn is_zero_index(&self, idx: OpId) -> bool {
        self.resolve_const(idx).and_then(Constant::as_dim) == Some(0)
    }
}

#[cfg(test)]
mod cb_tests {
    use super::*;
    use crate::kernel::Dev;

    fn cb(n: usize) -> OpId {
        OpId::from(n)
    }

    fn copy(k: &mut Kernel, src: OpId, src_idx: OpId, dst: OpId, dst_idx: OpId) -> OpId {
        k.push_back(Op::TT(TTOp::Copy { src, src_idx, dst, dst_idx, bytes: 4096 }))
    }

    /// A kernel with a reader loop producing `trip` tiles into `cb` and a
    /// compute loop consuming them, both with slot index `idx`.
    fn looped(cb_elements: i64, trip: u32, idx_produce: bool, idx_consume: bool) -> (Kernel, OpId, OpId, OpId, OpId) {
        let mut k = Kernel::new(Dev::Auto);
        let src = k.param(DType::F32);
        let cb = k.storage(DType::F32, MemScope::Circular, cb_elements);
        let dst = k.param_mut(DType::F32);
        let zero = k.const_idx(0u32);
        let len = k.const_idx(trip);
        // Reader: produce one tile per pass.
        let pl = k.push_back(Op::Loop { len });
        let pidx = if idx_produce { pl } else { zero };
        let produce = copy(&mut k, src, zero, cb, pidx);
        k.push_back(Op::EndLoop);
        k.push_back(Op::TT(TTOp::EndReader));
        // Compute: consume one tile per pass.
        let cl = k.push_back(Op::Loop { len });
        let cidx = if idx_consume { cl } else { zero };
        let consume = copy(&mut k, cb, cidx, dst, zero);
        k.push_back(Op::EndLoop);
        k.push_back(Op::TT(TTOp::EndCompute));
        (k, pl, cl, produce, consume)
    }

    #[test]
    fn a_producer_loop_becomes_one_block() {
        let (k, pl, cl, produce, consume) = looped(4096, 4, false, false);
        let plan = k.tt_cb_plan();
        let cb = plan.batch(produce).expect("produce is traffic").cb;
        assert_eq!(plan.produced(cb), 4);
        assert_eq!(plan.consumed(cb), 4);
        let p = plan.batch(produce).unwrap();
        let c = plan.batch(consume).unwrap();
        assert_eq!((p.per_op, p.hoist, p.kind), (4, pl, CbTraffic::Produce));
        assert_eq!((c.per_op, c.hoist, c.kind), (4, cl, CbTraffic::Consume));
        assert_eq!(plan.opens(pl), [CbEvent::Produce { cb, n: 4 }].as_slice());
        assert_eq!(plan.opens(cl), [CbEvent::Consume { cb, n: 4 }].as_slice());
        assert_eq!(plan.closes(k.next_op(produce)), [CbEvent::Produce { cb, n: 4 }].as_slice());
        assert_eq!(plan.closes(k.next_op(consume)), [CbEvent::Consume { cb, n: 4 }].as_slice());
    }

    #[test]
    fn two_ops_on_one_cb_in_a_loop_do_not_hoist() {
        let mut k = Kernel::new(Dev::Auto);
        let src = k.param(DType::F32);
        // Eight tiles, so the consumer's eight-iteration loop can be blocked;
        // the producer's loop cannot, because two ops touch the same CB.
        let cb = k.storage(DType::F32, MemScope::Circular, 8192);
        let dst = k.param_mut(DType::F32);
        let zero = k.const_idx(0u32);
        let four = k.const_idx(4u32);
        let eight = k.const_idx(8u32);
        let pl = k.push_back(Op::Loop { len: four });
        let a = copy(&mut k, src, zero, cb, zero);
        let b = copy(&mut k, src, zero, cb, zero);
        k.push_back(Op::EndLoop);
        k.push_back(Op::TT(TTOp::EndReader));
        let cl = k.push_back(Op::Loop { len: eight });
        let _ = cl;
        let c = copy(&mut k, cb, zero, dst, zero);
        k.push_back(Op::EndLoop);
        k.push_back(Op::TT(TTOp::EndCompute));
        let plan = k.tt_cb_plan();
        // Two traffic ops in one frame make the CB dirty: no block upgrade,
        // each stays a one-tile transaction.
        assert_eq!(plan.batch(a).unwrap().per_op, 1);
        assert_eq!(plan.batch(b).unwrap().per_op, 1);
        assert_ne!(plan.batch(a).unwrap().hoist, pl);
        assert_eq!(plan.batch(c).unwrap().per_op, 8);
        assert_eq!(plan.produced(plan.batch(a).unwrap().cb), 8);
        assert_eq!(plan.consumed(plan.batch(a).unwrap().cb), 8);
    }

    #[test]
    fn a_loop_counter_index_still_hoists() {
        let (k, pl, _, produce, _) = looped(4096, 4, true, true);
        let plan = k.tt_cb_plan();
        assert_eq!(plan.batch(produce).unwrap().hoist, pl);
        assert_eq!(plan.batch(produce).unwrap().per_op, 4);
    }

    #[test]
    fn a_block_larger_than_the_cb_streams_one_tile_per_pass() {
        // One tile of capacity, four iterations: no block, one transaction
        // per iteration, anchored at the traffic op.
        let (k, _, _, produce, consume) = looped(1024, 4, false, false);
        let plan = k.tt_cb_plan();
        let p = plan.batch(produce).unwrap();
        assert_eq!((p.per_op, p.hoist), (1, produce));
        assert_eq!(plan.batch(consume).unwrap().per_op, 1);
        assert_eq!(plan.produced(p.cb), 4);
        assert_eq!(plan.consumed(p.cb), 4);
    }

    #[test]
    #[should_panic(expected = "is not in a valid section")]
    fn traffic_in_the_wrong_section_is_loud() {
        let mut k = Kernel::new(Dev::Auto);
        let src = k.param(DType::F32);
        let cb = k.storage(DType::F32, MemScope::Circular, 1024);
        let zero = k.const_idx(0u32);
        // A reader-section consume has no producer yet.
        copy(&mut k, cb, zero, src, zero);
        k.push_back(Op::TT(TTOp::EndReader));
        k.push_back(Op::TT(TTOp::EndCompute));
        let _ = k.tt_cb_plan();
    }

    #[test]
    #[should_panic(expected = "produces 1 but consumes 2")]
    fn an_unbalanced_cb_is_loud() {
        let mut k = Kernel::new(Dev::Auto);
        let src = k.param(DType::F32);
        let cb = k.storage(DType::F32, MemScope::Circular, 4096);
        let dst = k.param_mut(DType::F32);
        let zero = k.const_idx(0u32);
        let two = k.const_idx(2u32);
        copy(&mut k, src, zero, cb, zero);
        k.push_back(Op::TT(TTOp::EndReader));
        let l = k.push_back(Op::Loop { len: two });
        let _ = l;
        copy(&mut k, cb, zero, dst, zero);
        k.push_back(Op::EndLoop);
        k.push_back(Op::TT(TTOp::EndCompute));
        let _ = k.tt_cb_plan();
    }

    #[test]
    #[should_panic(expected = "single-tile transaction")]
    fn a_single_tile_transaction_needs_slot_zero() {
        let mut k = Kernel::new(Dev::Auto);
        let src = k.param(DType::F32);
        let cb = k.storage(DType::F32, MemScope::Circular, 4096);
        let dst = k.param_mut(DType::F32);
        let zero = k.const_idx(0u32);
        let slot = k.const_idx(1024u32);
        copy(&mut k, src, zero, cb, slot);
        k.push_back(Op::TT(TTOp::EndReader));
        copy(&mut k, cb, zero, dst, zero);
        k.push_back(Op::TT(TTOp::EndCompute));
        let _ = k.tt_cb_plan();
    }

    #[test]
    #[should_panic(expected = "non-constant trip count")]
    fn a_symbolic_trip_count_is_loud() {
        let mut k = Kernel::new(Dev::Auto);
        let src = k.param(DType::F32);
        let cb = k.storage(DType::F32, MemScope::Circular, 4096);
        let zero = k.const_idx(0u32);
        let len = k.load(src, zero);
        k.push_back(Op::Loop { len });
        copy(&mut k, src, zero, cb, zero);
        k.push_back(Op::EndLoop);
        k.push_back(Op::TT(TTOp::EndReader));
        copy(&mut k, cb, zero, src, zero);
        k.push_back(Op::TT(TTOp::EndCompute));
        let _ = k.tt_cb_plan();
    }

    #[test]
    fn a_transaction_cycles_back_to_free() {
        let (c, mut m) = (cb(0), CbTransactions::new());
        m.reserve(c, 2);
        assert_eq!(m.state(c), CbState::Reserved { n: 2, filled: 0, avail: 0 });
        m.moved(c, 2);
        m.push(c, 2);
        assert_eq!(m.state(c), CbState::Pushed { avail: 2 });
        m.wait(c, 2);
        m.moved(c, 2);
        m.pop(c, 2);
        assert_eq!(m.state(c), CbState::Popped);
        m.assert_settled("test");
    }

    #[test]
    fn unread_tiles_carry_the_ring_forward() {
        let (c, mut m) = (cb(0), CbTransactions::new());
        m.reserve(c, 4);
        m.moved(c, 4);
        m.push(c, 4);
        m.wait(c, 2);
        m.moved(c, 2);
        m.pop(c, 2);
        assert_eq!(m.state(c), CbState::Pushed { avail: 2 });
        // The producer runs ahead of the consumer: the next block stacks.
        m.reserve(c, 2);
        m.moved(c, 2);
        m.push(c, 2);
        assert_eq!(m.state(c), CbState::Pushed { avail: 4 });
    }

    #[test]
    fn a_block_spanning_a_loop_is_one_transaction() {
        // The hoisted form the old emitter chose: reserve/push outside the
        // loop, one `moved` per iteration.
        let (c, mut m) = (cb(0), CbTransactions::new());
        m.reserve(c, 8);
        for _ in 0..8 {
            m.moved(c, 1);
        }
        m.push(c, 8);
        assert_eq!(m.state(c), CbState::Pushed { avail: 8 });
        m.wait(c, 8);
        for _ in 0..8 {
            m.moved(c, 1);
        }
        m.pop(c, 8);
        assert_eq!(m.state(c), CbState::Popped);
    }

    #[test]
    #[should_panic(expected = "must be free or pushed")]
    fn reserving_mid_transaction_is_loud() {
        let (c, mut m) = (cb(0), CbTransactions::new());
        m.reserve(c, 1);
        m.reserve(c, 1);
    }

    #[test]
    #[should_panic(expected = "must be Reserved with matching block size")]
    fn pushing_an_unreserved_block_is_loud() {
        let (c, mut m) = (cb(0), CbTransactions::new());
        m.push(c, 1);
    }

    #[test]
    #[should_panic(expected = "block overfill")]
    fn writing_past_the_reserved_block_is_loud() {
        let (c, mut m) = (cb(0), CbTransactions::new());
        m.reserve(c, 1);
        m.moved(c, 2);
    }

    #[test]
    #[should_panic(expected = "must be Pushed")]
    fn waiting_without_pushed_tiles_is_loud() {
        let (c, mut m) = (cb(0), CbTransactions::new());
        m.wait(c, 1);
    }

    #[test]
    #[should_panic(expected = "a transaction leaked")]
    fn a_mid_transaction_section_end_is_loud() {
        let (c, mut m) = (cb(0), CbTransactions::new());
        m.reserve(c, 1);
        m.assert_settled("compute");
    }
}

#[cfg(test)]
mod reindex_tests {
    use super::acceptance::tt_kernel;
    use super::*;

    /// The post-`tt_tile` shape of softmax's max-reduce: an 8-tile grid over
    /// the output rows, a 32-lane reduce axis, and the 8-iteration loop over
    /// the reduced axis' tiles. The logical load index is
    /// `(32*g + row) * 256 + lane + 32*i`.
    #[test]
    fn a_reduce_loop_rewrites_to_tile_numbers() {
        let mut k = tt_kernel();
        let src = k.param(DType::F32);
        let dst = k.param_mut(DType::F32);
        // The tiled buffer's geometry comes from the param's shape, which
        // `linearize` keeps on a tiled device: `[256, 256]` in, `[256]` out.
        {
            let r = k.const_idx(256u32);
            let c = k.const_idx(256u32);
            let s = k.push_back(Op::Stack { ops: vec![r, c].into_boxed_slice() });
            let Op::Param { shape, .. } = &mut k.ops[src].op else {
                unreachable!()
            };
            *shape = s;
        }
        {
            let r = k.const_idx(256u32);
            let s = k.push_back(Op::Stack { ops: vec![r].into_boxed_slice() });
            let Op::Param { shape, .. } = &mut k.ops[dst].op else {
                unreachable!()
            };
            *shape = s;
        }
        let row_len = k.const_idx(256u32);
        let eight = k.const_idx(8u32);
        let thirty_two = k.const_idx(32u32);
        let k_lane = k.push_back(Op::Unroll { axis: 0, len: 32u16 });
        let row_lane = k.push_back(Op::Unroll { axis: 1, len: 32u16 });
        let g = k.push_back(Op::Range { axis: 0, kind: RangeKind::Group(eight) });
        let g32 = k.push_back(Op::Binary { x: g, y: thirty_two, bop: BOp::Mul });
        let out_idx = k.push_back(Op::Binary { x: g32, y: row_lane, bop: BOp::Add });
        let base = k.push_back(Op::Binary { x: out_idx, y: row_len, bop: BOp::Mul });
        // The loop-carried accumulator, exactly as linearize leaves it: a
        // Register tile initialized before the loop, updated inside it, and
        // loaded again for the drain.
        let acc = k.storage(DType::F32, MemScope::Register, 32);
        let l = k.push_back(Op::Loop { len: eight });
        let i32 = k.push_back(Op::Binary { x: l, y: thirty_two, bop: BOp::Mul });
        let kk = k.push_back(Op::Binary { x: k_lane, y: i32, bop: BOp::Add });
        let idx = k.push_back(Op::Binary { x: base, y: kk, bop: BOp::Add });
        let load = k.push_back(Op::Load { src, src_idx: idx });
        let red = k.push_back(Op::Reduce { x: load, reduce_axis: k_lane, rop: BOp::Max });
        k.push_back(Op::Store { dst: acc, dst_idx: row_lane, src: red });
        k.push_back(Op::EndLoop);
        let acc_out = k.push_back(Op::Load { src: acc, src_idx: row_lane });
        let store = k.push_back(Op::Store { dst, dst_idx: out_idx, src: acc_out });

        k.tt_reindex();

        // Input tile = grid*8 (8 k-tiles per row group) + loop; the lanes are
        // the shape carrier.
        let Op::Load { src_idx, .. } = k.ops[load].op else {
            panic!("load changed variant")
        };
        let s = k.get_strides(src_idx).expect("the rewritten index is affine");
        assert_eq!(s[&OpId::NULL].1, 0, "the base is the grid/loop, not a const");
        assert_eq!(s[&g].1, 8192, "a grid step is 8 tiles");
        assert_eq!(s[&l].1, 1024, "a loop step is one tile");
        assert_eq!(s[&row_lane].1, 32, "the row lane is a shape carrier");
        assert_eq!(s[&k_lane].1, 1);
        assert!(k.tt_index_is_tile_units(src_idx), "a grid + loop index is physical");

        // Output tile = the grid axis alone; the row is a shape carrier.
        let Op::Store { dst_idx, .. } = k.ops[store].op else {
            panic!("store changed variant")
        };
        let s = k.get_strides(dst_idx).expect("the rewritten index is affine");
        assert_eq!(s[&OpId::NULL].1, 0);
        assert_eq!(s[&g].1, 1024, "one output tile per grid step");
        assert_eq!(s[&row_lane].1, 32);
    }

    /// The single-tile reduce: no grid, no loop, so the base is 0 and only the
    /// shape carriers remain. The plane is `[32, 20]` in, `[32]` out.
    #[test]
    fn a_single_tile_reduce_keeps_the_row_carrier() {
        let mut k = tt_kernel();
        let src = k.param(DType::F32);
        let dst = k.param_mut(DType::F32);
        {
            let r = k.const_idx(32u32);
            let c = k.const_idx(20u32);
            let s = k.push_back(Op::Stack { ops: vec![r, c].into_boxed_slice() });
            let Op::Param { shape, .. } = &mut k.ops[src].op else {
                unreachable!()
            };
            *shape = s;
        }
        {
            let r = k.const_idx(32u32);
            let s = k.push_back(Op::Stack { ops: vec![r].into_boxed_slice() });
            let Op::Param { shape, .. } = &mut k.ops[dst].op else {
                unreachable!()
            };
            *shape = s;
        }
        let row_len = k.const_idx(20u32);
        let k_lane = k.push_back(Op::Unroll { axis: 0, len: 32u16 });
        let row_lane = k.push_back(Op::Unroll { axis: 1, len: 32u16 });
        let base = k.push_back(Op::Binary { x: row_lane, y: row_len, bop: BOp::Mul });
        let idx = k.push_back(Op::Binary { x: base, y: k_lane, bop: BOp::Add });
        let load = k.push_back(Op::Load { src, src_idx: idx });
        let red = k.push_back(Op::Reduce { x: load, reduce_axis: k_lane, rop: BOp::Add });
        let store = k.push_back(Op::Store { dst, dst_idx: row_lane, src: red });

        k.tt_reindex();

        let Op::Load { src_idx, .. } = k.ops[load].op else {
            panic!("load changed variant")
        };
        let s = k.get_strides(src_idx).expect("affine");
        assert_eq!(s[&OpId::NULL].1, 0);
        assert_eq!(s[&row_lane].1, 32);
        assert_eq!(s[&k_lane].1, 1);
        let Op::Store { dst_idx, .. } = k.ops[store].op else {
            panic!("store changed variant")
        };
        let s = k.get_strides(dst_idx).expect("affine");
        assert_eq!(s[&row_lane].1, 32);
    }
}

#[cfg(test)]
mod acceptance {
    use super::*;
    use crate::kernel::Dev;

    /// A kernel bound to a Tenstorrent device without owning one: the whole
    /// `device_epilogue` consults `device_info().tenstorrent` and nothing else,
    /// so a pure-IR test can run the real lowering with no simulator.
    pub(super) fn tt_kernel() -> Kernel {
        use crate::backend::{DTypeCapability, DeviceInfo};
        use crate::shape::Dim;
        use std::sync::Arc;
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(DeviceInfo {
            compute: 1_000_000,
            max_global_work_dims: vec![Dim::from(1_000_000_000); 3],
            max_local_threads: 1024,
            max_local_work_dims: vec![1024, 1024, 1024],
            preferred_vector_size: 8,
            local_mem_size: 8192,
            max_register_bytes: 256_000,
            tensor_cores: false,
            warp_size: 1,
            cc: [0, 0],
            dtype_capability: [DTypeCapability::all(); DType::N_DTYPES],
            has_native_exp2: false,
            supported_vec_lens: vec![2, 4, 8, 16],
            vector_depth: 1,
            tenstorrent: true,
            tile: [Dim::from(32), Dim::from(32)],
            tile_sizes: vec![[Dim::from(32), Dim::from(32)]],
            mma_variants: vec![],
            num_circular_buffers: 8,
            has_openmp: false,
        }));
        k
    }

    /// One `[32, 32]` F32 input through one activation into one output: the
    /// shape the frontend hands `device_epilogue` for a single elementwise op.
    fn kernel_of(f: impl FnOnce(&mut Kernel, OpId) -> OpId) -> Kernel {
        let mut k = tt_kernel();
        // Read-only globals come first (see `verify`).
        let src = k.param(DType::F32);
        let dst = k.param_mut(DType::F32);
        // The `[32, 32]` tile index the frontend's linearize builds: i64 unroll
        // lanes and an i64 stride (`lane_tree`'s block form uses a `u16` stride,
        // which is a scalar-only shape).
        let zero = k.const_val(0i64);
        let stride = k.const_val(32i64);
        let row = k.push_back(Op::Unroll { axis: 0, len: 32 });
        let col = k.push_back(Op::Unroll { axis: 1, len: 32 });
        let row = k.binary(row, stride, BOp::Mul);
        let idx = k.binary(zero, row, BOp::Add);
        let idx = k.binary(idx, col, BOp::Add);
        let x = k.load(src, idx);
        let y = f(&mut k, x);
        k.store(dst, y, idx);
        k.device_epilogue();
        k
    }

    /// `(replays, loads, stores)` over the whole kernel. A pure elementwise op
    /// must be one row body with one `SFPLOAD` and one `SFPSTORE`.
    fn traffic(k: &Kernel) -> (usize, usize, usize) {
        let (mut replays, mut loads, mut stores) = (0, 0, 0);
        let mut id = k.head;
        while !id.is_null() {
            match k.at(id) {
                Op::TT(TTOp::Replay { .. }) => replays += 1,
                Op::TT(TTOp::SfpLoad(_)) => loads += 1,
                Op::TT(TTOp::SfpStore(_)) => stores += 1,
                _ => {}
            }
            id = k.next_op(id);
        }
        (replays, loads, stores)
    }

    /// `Tensor::relu` as the frontend writes it: `cmpgt(x, 0) * x`, with the
    /// `float_cast` on the mask that the `Op::Cast` arm has to translate.
    fn relu_src(k: &mut Kernel, x: OpId) -> OpId {
        let zero = k.const_val(0.0f32);
        let mask = k.binary(x, zero, BOp::Cmpgt);
        let mask = k.cast(mask, DType::F32);
        k.binary(mask, x, BOp::Mul)
    }

    /// `Tensor::tanh`: `sigmoid(2x) * 2 - 1`.
    fn tanh_src(k: &mut Kernel, x: OpId) -> OpId {
        let doubled = k.binary(x, x, BOp::Add);
        let s = k.sigmoid(doubled);
        let two = k.const_val(2.0f32);
        let m = k.binary(s, two, BOp::Mul);
        let one = k.const_val(1.0f32);
        k.binary(m, one, BOp::Sub)
    }

    /// Every activation is composed from primitives, so this is the fuser's
    /// acceptance gate: no hand-written program, one row body, one tile in and
    /// one tile out. Anything that pays a DST round trip fails here.
    #[test]
    fn activations_are_one_row_body() {
        let cases: [(&str, fn(&mut Kernel, OpId) -> OpId); 8] = [
            ("sigmoid", |k, x| k.sigmoid(x)),
            ("silu", |k, x| k.silu(x)),
            ("exp2", |k, x| k.exp2(x)),
            ("log2", |k, x| k.log2(x)),
            ("tanh", tanh_src),
            ("relu", relu_src),
            ("elu", |k, x| {
                let positive = relu_src(k, x);
                let e = k.exp(x);
                let n = k.neg(e);
                let one = k.const_val(1.0f32);
                let t = k.binary(n, one, BOp::Add); // 1 - e^x
                let negative = relu_src(k, t);
                k.binary(positive, negative, BOp::Sub)
            }),
            ("gelu", |k, x| {
                let half = k.const_val(0.5f32);
                let xh = k.binary(x, half, BOp::Mul);
                let x2 = k.binary(x, x, BOp::Mul);
                let x3 = k.binary(x2, x, BOp::Mul);
                let c = k.const_val(0.044_715f32);
                let cx3 = k.binary(x3, c, BOp::Mul);
                let inner = k.binary(x, cx3, BOp::Add);
                let kk = k.const_val((2.0f32 / core::f32::consts::PI).sqrt());
                let arg = k.binary(inner, kk, BOp::Mul);
                let t = tanh_src(k, arg);
                let one = k.const_val(1.0f32);
                let t1 = k.binary(t, one, BOp::Add);
                k.binary(xh, t1, BOp::Mul)
            }),
        ];
        let mut bad: Vec<(&str, (usize, usize, usize))> = Vec::new();
        for (name, f) in cases {
            let k = kernel_of(f);
            let t = traffic(&k);
            if t != (1, 1, 1) {
                bad.push((name, t));
            }
        }
        assert!(bad.is_empty(), "expected one row body, one load, one store; got {bad:?}");
    }
}

#[cfg(test)]
mod ir_contract {
    use super::acceptance::tt_kernel;
    use super::*;

    /// A gather with arbitrary indices: the row index is loaded from a tensor,
    /// so the source address is not an arithmetic progression. Scalar accesses
    /// (no tile lanes), which is where the emitter starts.
    fn gather_fixture() -> Kernel {
        let mut k = tt_kernel();
        let src = k.param(DType::F32);
        let idx = k.param(DType::I64);
        let dst = k.param_mut(DType::F32);
        let rows = k.const_idx(64i64);
        k.loop_over(rows, |k, i| {
            let row = k.load(idx, i);
            let stride = k.const_idx(64i64);
            let base = k.binary(row, stride, BOp::Mul);
            let lane = k.const_idx(0i64);
            let at = k.binary(base, lane, BOp::Add);
            let x = k.load(src, at);
            k.store(dst, x, at);
        });
        k
    }

    /// The IR the gather must have at each boundary: textual, as the debug
    /// printer writes it, plus the kernel hash. An exact contract, updated
    /// deliberately when a pass changes shape.
    #[test]
    fn gather_pipeline_ir_is_frozen() {
        let mut k = gather_fixture();
        let entry = format!("{k}");
        let entry_hash = k.get_hash();

        k.tt_tile();
        k.tt_unfold();
        k.tt_reindex();
        k.default_epilogue();
        k.tt_add_movers();
        k.tt_sync_cbs();
        k.tt_dedup_pushes();
        k.tt_place_pops();
        k.tt_fifo_check();
        let boundary = format!("{k}");
        let boundary_hash = k.get_hash();

        assert_eq!(entry, ENTRY_IR, "tt_tile entry IR");
        assert_eq!(entry_hash, ENTRY_HASH, "tt_tile entry hash");
        assert_eq!(boundary, BOUNDARY_IR, "tt_dataflow_select entry IR");
        assert_eq!(boundary_hash, BOUNDARY_HASH, "tt_dataflow_select entry hash");
    }

    const ENTRY_IR: &str = " r0: f32 = param global shape=NULL\n r1: i64 = param global shape=NULL\n r2: f32 = param global mut shape=NULL\n r3: i64 = 64\n for r4 in 0..r3 {    // 0..=63\n   r5: i64 = load r1[r4]    // 0..=63\n   r6: i64 = 64\n   r7: i64 = r5 * r6\n   r8: i64 = 0\n   r9: i64 = r7 + r8\n   r10: f32 = load r0[r9]    // 0..=0\n   store r2[r9] = r10    // 0..=0\n }\n\n";
    const ENTRY_HASH: u64 = 0xf50f_4928_19d0_184f;
    const BOUNDARY_IR: &str = " r0: f32 = param global shape=NULL\n r1: i64 = param global shape=NULL\n r2: f32 = param global mut shape=NULL\n r3: i64 = 64\n for r4 in 0..r3 {    // 0..=63\n   r5: i64 = load r1[r4]    // 0..=63\n   r7: i64 = r3 * r5\n   r10: f32 = load r0[r7]    // 0..=0\n   store r2[r7] = r10    // 0..=0\n }\n\n";
    const BOUNDARY_HASH: u64 = 0x360d_ffdd_cc85_80dd;
}
