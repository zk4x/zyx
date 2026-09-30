// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tenstorrent kernel passes: CB sync ops and their insertion.
//!
//! Slice 1 covers the CB sync family (`TTOp::ReserveBack`/`PushBack`/
//! `WaitFront`/`PopFront`) plus `tt_sync_cbs`, which wraps CB traffic
//! with straight-line single-tile syncs. No pops yet (wait-front only);
//! pop placement, NOC movement, and tile compute land in later slices.

use crate::Map;
use crate::kernel::{Kernel, MemScope, Op, OpId, TTOp};

impl Kernel {
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
    /// slots. No producer in slice 1 (waits only); lands with pop
    /// placement.
    pub fn tt_pop_front(&mut self, cb: OpId, n: u8) -> OpId {
        debug_assert!(
            matches!(self.at(cb), Op::Storage { scope: MemScope::Circular, .. }),
            "tt_pop_front: cb {cb} is not a Circular storage"
        );
        debug_assert!(n != 0, "tt_pop_front: n must be positive");
        self.push_back(Op::TT(TTOp::PopFront { cb, n }))
    }

    /// CB sync insertion: wrap every CB traffic op with straight-line
    /// single-tile syncs (`n = 1`; hoisted batches land with batching).
    /// Direction reads off which copy side is circular:
    /// - publish (`dst` GEP over a Circular storage): `ReserveBack`
    ///   immediately before, `PushBack` immediately after;
    /// - drain (`src` GEP over a Circular storage): `WaitFront`
    ///   immediately before, no pop yet;
    /// - CB-to-CB copy: `TileCopy` rule — `WaitFront` on the read side
    ///   immediately before, no pop yet.
    /// A circular `Load` is the compute-consume read (the kernel Load IS
    /// the CB read; codegen bundles read+consume, so its wait sits at the
    /// consumer — here it sits before the Load): `WaitFront` immediately
    /// before iff every user is a tile compute op, loud `panic!`
    /// otherwise. A circular buffer reached outside a GEP is malformed
    /// CB traffic — loud, never silently skipped.
    pub(crate) fn tt_sync_cbs(&mut self) {
        // Linear user map: users[v] = ops taking v as a data operand.
        let mut users: Map<OpId, Vec<OpId>> = Map::default();
        let mut scan = self.head;
        while !scan.is_null() {
            for p in self.at(scan).parameters() {
                users.entry(p).or_default().push(scan);
            }
            scan = self.next_op(scan);
        }

        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match self.ops[op_id].op {
                Op::Copy { src, dst } => {
                    let src_cb = match self.ops[src].op {
                        Op::GEP { x, .. }
                            if matches!(self.ops[x].op, Op::Storage { scope: MemScope::Circular, .. }) =>
                        {
                            Some(x)
                        }
                        Op::GEP { .. } => None,
                        Op::Storage { scope: MemScope::Circular, .. } => {
                            panic!("tt_sync_cbs: copy src {src:?} names a Circular storage directly, must go through a GEP")
                        }
                        _ => None,
                    };
                    let dst_cb = match self.ops[dst].op {
                        Op::GEP { x, .. }
                            if matches!(self.ops[x].op, Op::Storage { scope: MemScope::Circular, .. }) =>
                        {
                            Some(x)
                        }
                        Op::GEP { .. } => None,
                        Op::Storage { scope: MemScope::Circular, .. } => {
                            panic!("tt_sync_cbs: copy dst {dst:?} names a Circular storage directly, must go through a GEP")
                        }
                        _ => None,
                    };
                    match (src_cb, dst_cb) {
                        (None, None) => {}
                        (None, Some(cb)) => {
                            self.insert_before(op_id, Op::TT(TTOp::ReserveBack { cb, n: 1 }));
                            self.insert_after(op_id, Op::TT(TTOp::PushBack { cb, n: 1 }));
                        }
                        (Some(cb), None) => {
                            self.insert_before(op_id, Op::TT(TTOp::WaitFront { cb, n: 1 }));
                        }
                        (Some(scb), Some(_dcb)) => {
                            self.insert_before(op_id, Op::TT(TTOp::WaitFront { cb: scb, n: 1 }));
                        }
                    }
                }
                Op::Load { src } => {
                    // Non-CB reads are not sync business. Circularity gates
                    // the whole arm as a nested `if` — a `continue` here
                    // would skip the cursor advance at the loop bottom and
                    // spin forever.
                    if let Op::GEP { x: cb, .. } = self.ops[src].op
                        && matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. })
                    {
                        match users.get(&op_id) {
                            // Dead load: no consumer reads, no wait needed.
                            None => {}
                            Some(use_list) => {
                                for &u in use_list {
                                    if !matches!(
                                        self.ops[u].op,
                                        Op::TT(
                                            TTOp::MatmulTile { .. }
                                                | TTOp::TransposeTile { .. }
                                                | TTOp::ReduceTile { .. }
                                                | TTOp::BroadcastTile { .. }
                                        )
                                    ) {
                                        panic!(
                                            "tt_sync_cbs: circular load {op_id:?} feeds non-compute {u:?}"
                                        );
                                    }
                                }
                                self.insert_before(op_id, Op::TT(TTOp::WaitFront { cb, n: 1 }));
                            }
                        }
                    }
                }
                _ => {}
            }
            op_id = next;
        }

        self.verify();
    }
}
