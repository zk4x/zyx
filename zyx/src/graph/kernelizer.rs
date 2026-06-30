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

impl EGraph {
    /// Insert kernel alternatives into every e-class that doesn't have one.
    pub(crate) fn kernelize_all(&mut self) {
        // Build a topological order of classes
        let classes: Vec<ClassId> = self.classes.ids().collect();

        // Phase 1: insert single-op kernels for every non-kernel enode
        for &cid in &classes {
            if !self.classes.contains_key(cid) {
                continue;
            }
            let nodes: Vec<NodeId> = self.classes[cid].nodes.clone();
            for &nid in &nodes {
                if self.nodes[nid].is_kernel() || self.nodes[nid].is_transform() {
                    continue;
                }
                let mut inputs: Vec<ClassId> = self.nodes[nid].child_classes().to_vec();
                inputs.sort();
                let inputs: Box<[ClassId]> = inputs.into_boxed_slice();
                let outputs: Box<[ClassId]> = vec![cid].into_boxed_slice();
                let (knid, _) = self.make(ENode::Kernel(inputs, outputs, ProgramId::NULL));
                self.add_to_class(knid, cid);
            }
        }

        // Phase 2: extend kernels forward (fuse chains)
        // Walk classes in topological order, try to extend each kernel
        // by one op.
        for _ in 0..5 {
            // Iterate to fixpoint (small limit for safety)
            let mut changed = false;
            let classes: Vec<ClassId> = self.classes.ids().collect();

            for &cid in &classes {
                if !self.classes.contains_key(cid) {
                    continue;
                }
                // This class has enodes. Look at its op enodes (not kernels).
                let nodes: Vec<NodeId> = self.classes[cid].nodes.clone();
                for &nid in &nodes {
                    let op = &self.nodes[nid];
                    if op.is_kernel() || op.is_transform() {
                        continue;
                    }

                    // All children of this op are already computed by kernels.
                    // Try to extend: create a kernel that fuses us with any
                    // single-op kernel from our children.
                    let child_classes: Vec<ClassId> = op.child_classes();
                    if child_classes.is_empty() {
                        continue;
                    }

                    // Find kernel alternatives in each child class
                    for &child in &child_classes {
                        if !self.classes.contains_key(child) {
                            continue;
                        }
                        let child_nodes: Vec<NodeId> = self.classes[child].nodes.clone();
                        for &cnid in &child_nodes {
                            let ENode::Kernel(child_inputs, child_outputs, _) = &self.nodes[cnid] else {
                                continue;
                            };
                            // child_outputs must include child
                            if !child_outputs.contains(&child) {
                                continue;
                            }

                            // Fuse: new kernel = (child_inputs + current other-inputs, outputs)
                            let mut new_inputs: Vec<ClassId> = child_inputs.to_vec();
                            // If child kernel has no compute inputs (leaf/const), the fused
                            // kernel still needs the child's buffer as an input.
                            if child_inputs.is_empty() {
                                new_inputs.push(child);
                            }
                            for &other_child in &child_classes {
                                if other_child != child {
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
