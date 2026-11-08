// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The reader and writer programs: the boundary IR lowered to RISC-V.
//!
//! The reader section (head .. `EndReader`) and the writer section (after
//! `EndCompute`) are ordinary SSA over parameters, CB pages, index arithmetic
//! and the four CB syncs. This module turns them into the programs that run on
//! RISCV B and NCRISC: loops are branches, the index expression is lowered as
//! written (including a load whose value is an address, the gather), and the
//! NOC/CB operations are inline sequences.
//!
//! ## Launch arguments
//!
//! `a0` points at a `u32` array in L1, filled by the launch:
//!
//! ```text
//! 0        grid x
//! 1        grid y
//! 2        CB interface base (L1, 32 bytes per CB)
//! 3        this tile's NOC address-high word
//! 4 + 2p   parameter p's DRAM address low word
//! 5 + 2p   parameter p's DRAM address high word
//! ```
//!
//! A DRAM address is `[hi:lo]` of the NOC address of the parameter's first
//! byte; the program adds the index (already a byte offset) and issues.

use super::riscv::{self, Assembler, Label, cb, noc, reg};
use crate::Map;
use crate::kernel::{BOp, Kernel, MemScope, Op, OpId, ParamKind, RangeKind, TTOp};
use tt_isa::noc::niu::{Niu, TxnId};

/// Launch argument word offsets (see the module docs).
pub const ARG_GRID_X: u32 = 0;
pub const ARG_GRID_Y: u32 = 1;
pub const ARG_IFACE: u32 = 2;
pub const ARG_ME_HI: u32 = 3;
/// Parameter `p`'s address words: `ARG_PARAM + 2 * p` and `+ 1`.
pub const ARG_PARAM: u32 = 4;

/// Emit the reader program (B): every `Copy` into a CB and everything that
/// computes its address, up to `EndReader`.
pub fn reader(k: &Kernel, cbs: &Map<OpId, u32>, params: &Map<OpId, u32>, l1: &Map<OpId, u32>) -> Vec<u32> {
    let mut e = Emitter::new(k, cbs, params, true, l1);
    e.preamble();
    let mut id = k.head;
    while !id.is_null() {
        if matches!(k.at(id), Op::TT(TTOp::EndReader)) {
            break;
        }
        e.op(id);
        id = k.next_op(id);
    }
    e.finish()
}

/// Emit the writer program (NC): from `EndCompute` to the end.
pub fn writer(k: &Kernel, cbs: &Map<OpId, u32>, params: &Map<OpId, u32>, l1: &Map<OpId, u32>) -> Vec<u32> {
    let mut e = Emitter::new(k, cbs, params, false, l1);
    e.preamble();
    let mut writing = false;
    let mut id = k.head;
    while !id.is_null() {
        if !writing && matches!(k.at(id), Op::TT(TTOp::EndCompute)) {
            writing = true;
        } else if writing {
            e.op(id);
        }
        id = k.next_op(id);
    }
    e.finish()
}

struct Emitter<'a> {
    k: &'a Kernel,
    cbs: &'a Map<OpId, u32>,
    params: &'a Map<OpId, u32>,
    /// L1 address of each locally placed page.
    l1: &'a Map<OpId, u32>,
    reader: bool,
    a: Assembler,
    /// Registers available to an expression, and the next free one.
    free: Vec<u8>,
    /// Registers that hold values live across an expression (never reused).
    pinned: Vec<u8>,
    /// The lane loop's variable, while inside a vector copy.
    lane: Option<(OpId, u8)>,
    /// Open loops: see [`OpenLoop`].
    loops: Vec<OpenLoop>,
}

/// An open loop: its `Loop` op, the iteration index register, an optional
/// remaining-count register (absent when the trip count is a constant, so the
/// constant is re-materialised at the bottom instead of held), that constant,
/// and the top label.
struct OpenLoop {
    var: OpId,
    index: u8,
    remain: Option<u8>,
    count: Option<i64>,
    top: Label,
}

impl<'a> Emitter<'a> {
    fn new(k: &'a Kernel, cbs: &'a Map<OpId, u32>, params: &'a Map<OpId, u32>, reader: bool, l1: &'a Map<OpId, u32>) -> Self {
        Self { k, cbs, params, l1, reader, a: Assembler::new(), free: Vec::new(), pinned: Vec::new(), lane: None, loops: Vec::new() }
    }

    fn preamble(&mut self) {
        // a0 = args, a6 = CB interface base, a7 = NOC initiator, a5 = this
        // tile's address-high word. A pure transfer has no CB interface; the
        // load is harmless.
        self.a.push(riscv::lw(reg::A6, reg::A0, ARG_IFACE as i32 * 4));
        self.a.push(riscv::lw(reg::A5, reg::A0, ARG_ME_HI as i32 * 4));
        noc::open(&mut self.a, reg::A7, Niu::Noc0);
    }

    fn finish(mut self) -> Vec<u32> {
        self.a.push(riscv::RET);
        self.a.finish()
    }

    // -- registers ---------------------------------------------------------

    fn reset(&mut self) {
        // A loop counter lives across statements, so it is never handed out
        // again while the loop is open.
        let held: Vec<u8> = self.loops.iter().flat_map(|l| [Some(l.index), l.remain]).flatten().collect();
        self.free = vec![
            reg::T0,
            reg::T1,
            reg::T2,
            reg::T3,
            reg::T4,
            reg::T5,
            reg::T6,
            reg::A1,
            reg::A2,
            reg::A3,
            reg::A4,
        ]
        .into_iter()
        .filter(|r| !held.contains(r))
        .collect();
        self.pinned.clear();
    }

    fn alloc(&mut self) -> u8 {
        self.free.pop().unwrap_or_else(|| {
            panic!("tenstorrent boundary: out of registers in {}", if self.reader { "reader" } else { "writer" })
        })
    }

    /// Keep a register for the rest of the statement.
    fn pin(&mut self, r: u8) {
        self.pinned.push(r);
    }

    /// Return a consumed temporary to the pool.
    ///
    /// A loop's index/remaining registers and the lane register are operands
    /// that no statement owns: an expression that reads one must not hand it
    /// back, or the next `alloc` overwrites the loop control.
    fn release(&mut self, r: u8) {
        let held = self.pinned.contains(&r)
            || self.loops.iter().any(|l| l.index == r || l.remain == Some(r))
            || self.lane.is_some_and(|(_, lane)| lane == r);
        if !held {
            self.free.push(r);
        }
    }

    // -- one op ------------------------------------------------------------

    fn op(&mut self, id: OpId) {
        match &self.k.ops[id].op {
            Op::TT(TTOp::ReserveBack { cb, n }) => {
                self.reset();
                let cb = self.cb_id(*cb);
                cb::reserve_back(&mut self.a, reg::A6, cb, u32::from(*n), [reg::T0, reg::T1, reg::T2]);
            }
            Op::TT(TTOp::PushBack { cb, n }) => {
                self.reset();
                let cb = self.cb_id(*cb);
                cb::push_back(&mut self.a, reg::A6, cb, u32::from(*n), [reg::T0, reg::T1, reg::T2, reg::T3]);
            }
            Op::TT(TTOp::WaitFront { cb, n }) => {
                self.reset();
                let cb = self.cb_id(*cb);
                cb::wait_front(&mut self.a, reg::A6, cb, u32::from(*n), [reg::T0, reg::T1, reg::T2]);
            }
            Op::TT(TTOp::PopFront { cb, n }) => {
                self.reset();
                let cb = self.cb_id(*cb);
                cb::pop_front(&mut self.a, reg::A6, cb, u32::from(*n), [reg::T0, reg::T1, reg::T2, reg::T3]);
            }
            Op::Loop { len } => {
                self.reset();
                // A constant trip count needs no register: the bottom of the
                // loop re-materialises it. A dynamic one is read once.
                let count = match self.k.at(*len) {
                    Op::Const(c) => c.as_dim(),
                    _ => None,
                };
                let remain = match count {
                    Some(_) => None,
                    None => {
                        let n = self.value_into(*len);
                        let r = self.alloc();
                        self.a.push(riscv::addi(r, n, 0));
                        self.release(n);
                        Some(r)
                    }
                };
                // The loop variable is the iteration index 0..len-1, as the
                // kernel IR uses it.
                let index = self.alloc();
                self.a.li(index, 0);
                let top = self.a.label();
                self.a.bind(top);
                self.loops.push(OpenLoop { var: id, index, remain, count, top });
            }
            Op::EndLoop => {
                // Read the loop while it is still open (its registers are held
                // out of the pool), and pop it only after the branch is out.
                let Some(OpenLoop { index, remain, count, top, .. }) = self.loops.last() else {
                    panic!("tenstorrent boundary: EndLoop without a loop");
                };
                let (index, remain, count, top) = (*index, *remain, *count, *top);
                self.reset();
                self.a.push(riscv::addi(index, index, 1));
                match (remain, count) {
                    (Some(r), _) => {
                        self.a.push(riscv::addi(r, r, -1));
                        self.a.bne(r, reg::ZERO, top);
                    }
                    (None, Some(n)) => {
                        let t = self.alloc();
                        self.a.li(t, u32::try_from(n).expect("loop length"));
                        self.a.bne(index, t, top);
                        self.release(t);
                    }
                    (None, None) => panic!("tenstorrent boundary: loop without a trip count"),
                }
                self.loops.pop();
            }
            Op::Copy { src, src_idx, dst, dst_idx } => {
                self.reset();
                self.copy(id, *src, *src_idx, *dst, *dst_idx);
            }
            Op::TT(TTOp::Copy { src, src_idx, dst, dst_idx, bytes }) => {
                self.reset();
                self.traffic(id, *src, *src_idx, *dst, *dst_idx, Some(*bytes));
            }
            _ => {}
        }
    }

    fn cb_id(&self, cb: OpId) -> u32 {
        *self.cbs.get(&cb).unwrap_or_else(|| panic!("tenstorrent boundary: CB {cb} is not allocated"))
    }

    fn copy(&mut self, id: OpId, src: OpId, src_idx: OpId, dst: OpId, dst_idx: OpId) {
        self.traffic(id, src, src_idx, dst, dst_idx, None);
    }

    /// One traffic op: a reader publish (DRAM -> CB) or a writer drain
    /// (CB -> DRAM). A vector index is one span per lane.
    fn traffic(&mut self, id: OpId, src: OpId, src_idx: OpId, dst: OpId, dst_idx: OpId, bytes: Option<u32>) {
        let is_storage = |o: OpId| matches!(self.k.at(o), Op::Storage { .. });
        let is_global = |o: OpId| matches!(self.k.at(o), Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. });
        let (param, idx, page, page_idx, publish) = if is_global(src) && is_storage(dst) {
            (src, src_idx, dst, dst_idx, true)
        } else if is_storage(src) && is_global(dst) {
            (dst, dst_idx, src, src_idx, false)
        } else {
            panic!("tenstorrent boundary: copy {id:?} is not a DRAM<->L1 page move")
        };
        let circular = matches!(self.k.at(page), Op::Storage { scope: MemScope::Circular, .. });
        if circular && self.reader != publish {
            panic!(
                "tenstorrent boundary: a {} section holds a {} copy",
                if self.reader { "reader" } else { "writer" },
                if publish { "publish" } else { "drain" }
            );
        }
        let page_datums = match self.k.at(page) {
            Op::Storage { len, .. } => u64::try_from(*len).expect("negative page length"),
            _ => unreachable!(),
        };
        let el = self.k.dtype(page).bit_size() as u64 / 8;
        let lanes: u64 = self.k.shape(idx).iter().map(|d| *d as u64).product::<u64>().max(1);
        assert_eq!(page_datums % lanes, 0, "tenstorrent boundary: page {page_datums} is not {lanes} whole spans");
        let span = u32::try_from(page_datums / lanes * el).expect("span too large");
        if std::env::var_os("ZYX_TT_DUMP").is_some() {
            eprintln!(
                "[tt boundary] copy {id:?} page={page:?} page_datums={page_datums} lanes={lanes} span={span} bytes={bytes:?} publish={publish}"
            );
        }
        if let Some(bytes) = bytes {
            assert_eq!(u64::from(bytes), page_datums * el, "tenstorrent boundary: copy {id:?} bytes disagree with its page");
        }
        // The L1 side of the move: a CB's write/read pointer, or the placed
        // address of a plain L1 page. Slot 0 at depth 1.
        let ptr = self.alloc();
        self.pin(ptr);
        if circular {
            let cb_id = self.cb_id(page);
            let field = if publish { cb::WR_PTR } else { cb::RD_PTR };
            self.a.push(riscv::lw(ptr, reg::A6, (cb_id * cb::STRIDE) as i32 + field));
        } else {
            let at = *self.l1.get(&page).unwrap_or_else(|| panic!("tenstorrent boundary: page {page} is not placed"));
            self.a.li(ptr, at);
        }
        // The DRAM side: the parameter's low word.
        let p =
            *self.params.get(&param).unwrap_or_else(|| panic!("tenstorrent boundary: parameter {param} is not in the launch"));
        let lo = self.alloc();
        self.pin(lo);
        self.a.push(riscv::lw(lo, reg::A0, (ARG_PARAM + 2 * p) as i32 * 4));
        let hi = self.alloc();
        self.pin(hi);
        self.a.push(riscv::lw(hi, reg::A0, (ARG_PARAM + 2 * p + 1) as i32 * 4));
        // A slot index other than 0 is a second page in the ring: not at depth 1.
        if let Some(slot) = self.k.resolve_const(page_idx).and_then(|c| c.as_dim())
            && slot != 0
        {
            panic!("tenstorrent boundary: copy {id:?} uses page slot {slot} on a depth-1 ring");
        }
        if lanes == 1 {
            let dram = self.value(idx);
            self.a.push(riscv::add(dram, dram, lo));
            self.issue(dram, hi, ptr, span, publish);
        } else {
            let count = self.alloc();
            self.pin(count);
            self.a.li(count, u32::try_from(lanes).expect("lane count"));
            let counter = self.alloc();
            self.pin(counter);
            self.a.li(counter, 0);
            let span_reg = self.alloc();
            self.pin(span_reg);
            self.a.li(span_reg, span);
            let top = self.a.label();
            self.a.bind(top);
            self.lane = Some((lane_var(self.k, idx), counter));
            let dram = self.value(idx);
            self.lane = None;
            let off = self.alloc();
            self.a.push(riscv::mul(off, counter, span_reg));
            let l1 = self.alloc();
            self.a.push(riscv::add(l1, ptr, off));
            self.a.push(riscv::add(dram, dram, lo));
            self.issue(dram, hi, l1, span, publish);
            self.a.push(riscv::addi(counter, counter, 1));
            self.a.bltu(counter, count, top);
        }
    }

    fn issue(&mut self, dram: u8, hi: u8, l1: u8, len: u32, publish: bool) {
        // A publish reads DRAM -> L1; a drain writes L1 -> DRAM. The high words
        // are launch arguments: the DRAM endpoint and this tile.
        let txn = TxnId::new(0).expect("transaction 0");
        let t0 = self.alloc();
        if publish {
            noc::issue_addr(
                &mut self.a,
                reg::A7,
                noc::Addr::Reg(dram),
                noc::Addr::Reg(hi),
                noc::Addr::Reg(l1),
                noc::Addr::Reg(reg::A5),
                0,
                len,
                false,
                t0,
            );
        } else {
            noc::issue_addr(
                &mut self.a,
                reg::A7,
                noc::Addr::Reg(l1),
                noc::Addr::Reg(reg::A5),
                noc::Addr::Reg(dram),
                noc::Addr::Reg(hi),
                0,
                len,
                true,
                t0,
            );
        }
        noc::drain(&mut self.a, reg::A7, txn, t0);
    }

    fn value(&mut self, id: OpId) -> u8 {
        self.value_into(id)
    }

    /// Evaluate one index expression into a register, as written.
    fn value_into(&mut self, id: OpId) -> u8 {
        match &self.k.ops[id].op {
            Op::Const(c) => {
                let v = c.as_dim().unwrap_or_else(|| panic!("tenstorrent boundary: non-integer const {id:?}"));
                let r = self.alloc();
                self.a.li(r, v as u32);
                r
            }
            Op::Range { axis, kind: RangeKind::Group(_) } => {
                let r = self.alloc();
                self.a.push(riscv::lw(r, reg::A0, (ARG_GRID_X + axis) as i32 * 4));
                r
            }
            Op::Unroll { .. } => match self.lane {
                Some((var, r)) if var == id => r,
                _ => panic!("tenstorrent boundary: unroll {id:?} outside its lane loop"),
            },
            Op::Loop { .. } => match self.loops.iter().rev().find(|l| l.var == id) {
                Some(l) => l.index,
                None => panic!("tenstorrent boundary: loop counter {id:?} outside its loop"),
            },
            Op::Binary { x, y, bop } => {
                let a = self.value_into(*x);
                let b = self.value_into(*y);
                let r = self.alloc();
                self.a.push(match bop {
                    BOp::Add => riscv::add(r, a, b),
                    BOp::Sub => riscv::sub(r, a, b),
                    BOp::Mul => riscv::mul(r, a, b),
                    BOp::Div => riscv::divu(r, a, b),
                    BOp::Mod => riscv::remu(r, a, b),
                    BOp::BitAnd => riscv::and(r, a, b),
                    BOp::BitOr => riscv::or(r, a, b),
                    BOp::BitXor => riscv::xor(r, a, b),
                    other => panic!("tenstorrent boundary: {other:?} is not an integer index op"),
                });
                self.release(a);
                self.release(b);
                r
            }
            Op::Cast { x, dtype } => {
                assert!(dtype.is_int(), "tenstorrent boundary: cast to {dtype:?} in an index");
                self.value_into(*x)
            }
            Op::Load { src, src_idx } => {
                assert!(
                    matches!(self.k.at(*src), Op::Storage { .. }),
                    "tenstorrent boundary: an index load is not from an L1 page"
                );
                let el = self.k.dtype(*src).bit_size() as i32 / 8;
                let idx = self.value_into(*src_idx);
                let rd = self.alloc();
                match self.k.at(*src) {
                    Op::Storage { scope: MemScope::Circular, .. } => {
                        let cb = self.cb_id(*src);
                        self.a.push(riscv::lw(rd, reg::A6, (cb * cb::STRIDE) as i32 + cb::RD_PTR));
                    }
                    _ => {
                        let at = *self.l1.get(src).unwrap_or_else(|| panic!("tenstorrent boundary: page {src} is not placed"));
                        self.a.li(rd, at);
                    }
                }
                let scaled = self.alloc();
                self.a.li(scaled, el as u32);
                self.a.push(riscv::mul(scaled, scaled, idx));
                self.a.push(riscv::add(rd, rd, scaled));
                let v = self.alloc();
                self.a.push(riscv::lw(v, rd, 0));
                self.release(idx);
                self.release(rd);
                self.release(scaled);
                v
            }
            Op::Select { cond, if_true, if_false } => {
                let c = self.value_into(*cond);
                let r = self.alloc();
                let other = self.a.label();
                let end = self.a.label();
                self.a.beq(c, reg::ZERO, other);
                self.release(c);
                let t = self.value_into(*if_true);
                self.a.push(riscv::addi(r, t, 0));
                self.release(t);
                self.a.jal(reg::ZERO, end);
                self.a.bind(other);
                let f = self.value_into(*if_false);
                self.a.push(riscv::addi(r, f, 0));
                self.release(f);
                self.a.bind(end);
                r
            }
            other => panic!("tenstorrent boundary: {other:?} at {id:?} is not an index op"),
        }
    }
}

/// The lane variable an expression is vectorised over.
fn lane_var(k: &Kernel, idx: OpId) -> OpId {
    let mut stack = vec![idx];
    for _ in 0..250_000 {
        let Some(n) = stack.pop() else { break };
        match &k.ops[n].op {
            Op::Unroll { .. } => return n,
            Op::Binary { x, y, .. } => {
                stack.push(*x);
                stack.push(*y);
            }
            Op::Unary { x, .. } | Op::Cast { x, .. } => stack.push(*x),
            Op::Select { cond, if_true, if_false } => {
                stack.push(*cond);
                stack.push(*if_true);
                stack.push(*if_false);
            }
            _ => {}
        }
    }
    panic!("tenstorrent boundary: a vector index has no lane variable")
}
