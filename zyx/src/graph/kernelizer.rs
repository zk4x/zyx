// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! E-graph kernelizer.
//!
//! Walks the e-graph's e-classes and inserts `ENode::Kernel` variants for
//! different fusion granularities. Each `Kernel` enode represents "this
//! chain of ops can be compiled into one GPU kernel." The e-graph's
//! extraction DP then picks the cheapest combination.

use crate::{
    backend::ProgramId,
    graph::search::{ClassId, EGraph, ENode, NodeId},
};

const HIGH_COST: u64 = 1_000_000;

fn kernel_cost(ops_count: u32) -> u64 {
    ((1.0 + ops_count as f64).ln() * HIGH_COST as f64) as u64
}

impl EGraph {
    /// Insert kernel alternatives into every e-class that doesn't have one.
    pub(crate) fn kernelize_all(&mut self) {
        let classes: Vec<ClassId> = self.classes.ids().collect();

        // Phase 1: single-op kernels for every non-kernel, non-transform, non-leaf enode.
        // Transforms (Cast, Expand, Permute, Reshape, Pad) have no kernel in Phase 1
        // — they get fused kernels in Phase 2.
        // Leaf has no kernel — it's an implicit input buffer, fused into compute kernels.
        for &cid in &classes {
            if !self.classes.contains_key(cid) {
                continue;
            }
            let nodes: Vec<NodeId> = self.classes[cid].nodes.clone();
            for &nid in &nodes {
                if self.nodes[nid].is_kernel() || self.nodes[nid].is_transform() || matches!(&self.nodes[nid], ENode::Leaf(_)) {
                    continue;
                }
                let mut inputs: Vec<ClassId> = self.nodes[nid].child_classes().to_vec();
                inputs.retain(|&c| {
                    !self.classes[self.find_class(c)].nodes.iter().any(|&n| matches!(&self.nodes[n], ENode::Leaf(_)))
                });
                inputs.sort();
                let inputs: Box<[ClassId]> = inputs.into_boxed_slice();
                let outputs: Box<[ClassId]> = vec![cid].into_boxed_slice();
                let (knid, _) = self.make(ENode::Kernel(inputs, outputs, ProgramId::NULL));
                self.costs.insert(knid, kernel_cost(1));
                self.ops_count.insert(knid, 1);
                self.add_to_class(knid, cid);
            }
        }

        // Phase 2: extend kernels forward through transforms and ops.
        for _ in 0..5 {
            let mut changed = false;
            let classes: Vec<ClassId> = self.classes.ids().collect();

            for &cid in &classes {
                if !self.classes.contains_key(cid) {
                    continue;
                }
                let nodes: Vec<NodeId> = self.classes[cid].nodes.clone();
                for &nid in &nodes {
                    let op = &self.nodes[nid];
                    if op.is_kernel() {
                        continue;
                    }

                    let child_classes: Vec<ClassId> = op.child_classes();
                    if child_classes.is_empty() {
                        continue;
                    }

                    for &child in &child_classes {
                        if !self.classes.contains_key(child) {
                            continue;
                        }
                        let child_nodes: Vec<NodeId> = self.classes[child].nodes.clone();
                        for &cnid in &child_nodes {
                            let ENode::Kernel(child_inputs, child_outputs, _) = &self.nodes[cnid] else {
                                continue;
                            };
                            if !child_outputs.contains(&child) {
                                continue;
                            }

                            let mut new_inputs: Vec<ClassId> = child_inputs.to_vec();
                            for &other_child in &child_classes {
                                if other_child != child {
                                    if self.classes[self.find_class(other_child)].nodes.iter().any(|&n| matches!(&self.nodes[n], ENode::Leaf(_))) {
                                        continue;
                                    }
                                    new_inputs.push(other_child);
                                }
                            }
                            new_inputs.sort();
                            new_inputs.dedup();
                            let outputs: Vec<ClassId> = vec![cid];
                            let (knid, _) = self.make(ENode::Kernel(
                                new_inputs.into_boxed_slice(),
                                outputs.clone().into_boxed_slice(),
                                ProgramId::NULL,
                            ));

                            let child_ops = self.ops_count.get(&cnid).copied().unwrap_or(1);
                            let total_ops = 1 + child_ops;
                            self.costs.insert(knid, kernel_cost(total_ops));
                            self.ops_count.insert(knid, total_ops);

                            for &out in &outputs {
                                self.add_to_class(knid, out);
                            }
                            changed = true;
                        }
                    }
                }
            }

            if !changed {
                break;
            }
        }
    }
}
