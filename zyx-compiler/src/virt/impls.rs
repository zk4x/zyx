use std::collections::{BTreeMap, BTreeSet};
use std::prelude::rust_2015::Vec;
use zyx_core::error::ZyxError;
use zyx_core::node::Node;
use zyx_core::runtime::RuntimeBackend;
use zyx_core::scalar::Scalar;
use zyx_core::tensor::Id;
use crate::{CompiledBackend, Compiler};
use crate::virt::VirtKernel;
use alloc::vec;

/// Compiled graph
pub struct CompiledGraph<Program> {
    args: Vec<Id>,
    program: Program,
    flop: usize,
    bytes: usize,
}

impl<C: Compiler> RuntimeBackend for CompiledBackend<C> {
    type CompiledGraph = CompiledGraph<C::Program>;

    fn is_empty(&self, x: Id) -> bool {
        !self.buffers.contains_key(&x)
    }

    fn evaluated_nodes(&self) -> BTreeSet<Id> {
        self.buffers.keys().copied().collect()
    }

    fn store<T: Scalar, IT>(&mut self, x: Id, iter: IT) -> Result<(), ZyxError>
        where
            IT: IntoIterator<Item=T>,
            IT::IntoIter: ExactSizeIterator,
    {
        //std::println!("Storing {x}");
        let data = iter.into_iter();
        let mut buffer = self.compiler.allocate_mem(data.len(), T::dtype())?;
        self.compiler.store_mem(&mut buffer, data)?;
        self.buffers.insert(x, buffer);
        Ok(())
    }

    fn load<T: Scalar>(&mut self, x: Id, numel: usize) -> Result<Vec<T>, ZyxError> {
        //std::println!("Loading {x}");
        if let Some(buffer) = self.buffers.get(&x) {
            self.compiler.load_mem(buffer, numel)
        } else {
            panic!("Buffer not evaluated");
        }
    }

    fn remove(&mut self, x: Id) -> Result<(), ZyxError> {
        // TODO we can later optimize this by not deallocating memory if it can be reused later
        if let Some(mut buffer) = self.buffers.remove(&x) {
            //std::println!("Dropping buffer {p} out of total {} buffers", self.buffers.len());
            self.compiler.deallocate_mem(&mut buffer)?;
        }
        Ok(())
    }

    fn compile_graph(&mut self, global_rcs: &[u32], nodes: &[Node], to_eval: &BTreeSet<Id>) -> Result<Self::CompiledGraph, ZyxError> {
        // Find the best order of execution of nodes
        let (order, rcs) = {
            let mut temp_rcs: BTreeMap<Id, u32> = BTreeMap::new();
            let mut params: Vec<Id> = to_eval.iter().copied().collect();
            params.reserve(100);
            while let Some(nid) = params.pop() {
                //std::println!("{nid} is evaluated: {}", self.runtime_backend.is_evaluated(nid));
                temp_rcs
                    .entry(nid)
                    .and_modify(|rc| *rc += 1)
                    .or_insert_with(|| {
                        if self.is_empty(nid) {
                            params.extend(nodes[nid.i()].parameters());
                        }
                        1
                    });
            }
            // Order them using rcs reference counts.
            let mut order = Vec::new();
            let mut rcs: BTreeMap<Id, u32> = BTreeMap::new();
            let mut params: Vec<Id> = to_eval.iter().copied().collect();
            params.reserve(100);
            while let Some(nid) = params.pop() {
                if let Some(temp_rc) = temp_rcs.get(&nid) {
                    let rc = rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
                    if *temp_rc == *rc {
                        order.push(nid);
                        params.extend(nodes[nid.i()].parameters());
                    }
                }
            }
            order.reverse();
            (order, rcs)
        };

        // First reorder nodes in such a way, that movement ops are as late as possible,
        // after all unary ops just before reduce ops. (Do not reorder it after binary ops though.)
        // Permutes after reduce ops can be also moved before reduce ops!

        // Kernel
        let mut kernel = VirtKernel {
            indices: vec![],
            mems: [vec![], vec![], vec![]],
            instructions: vec![],
        };

        Ok(CompiledGraph {
            args: vec![],
            program: self.compiler.compile_program(&kernel)?,
            flop: 0,
            bytes: 0,
        })
    }

    fn launch_graph(&mut self, graph: &Self::CompiledGraph) -> Result<(), ZyxError> {
        let buffers: Vec<&C::Buffer> = graph.args.iter().map(|id| self.buffers.get(id).unwrap()).collect();
        self.compiler.launch_program(&graph.program, &buffers, graph.flop, graph.bytes)
    }
}

// Higher level abstraction, basically with tiles, views, bound dimensions and loops
// On this level we can do ops reordering, reshapes, permutes and pads can be moved even
// with binary ops, while expands can be only reordered between unary ops.
//
// So we need to add loops,
// mark movement of leafs between global, local and register tiles
// Add loops for expands bigger than work size,
// those must loops end with reduce ops, since initial work size (global and local loops)
// is calculated with output size in mind.
// Apply movement ops directly on tiles,
// Leave unary and binary ops as they are.
// Here we can do all the local and register tile optimizations and have special instructions
// for things like 4x4x4 matmuls, where we can just use tensor cores, wmma cores or like strassen

// For more low level (this needs to be rewritten once the more higher level approach is finalized
// Nodes like this (don't worry to make some assumptions)
//
// If values is in global scope
//   -> create new register mem
//   -> copy data from global into register (TODO use wide loads if it increases performance)
// Exp, Tanh, Neg, Sin, Cos
//   -> create new register mem
//   -> add instruction for unary op
//   -> decrease reference count from mems
// ReLU
//   -> same as unary, just rewrite as binary max with const
// Add, Sub, Mul, Div, Pow, Cmplt
//   -> same as unary
// Sum, Max
//   -> mark reduce start before first movement op on any of the leafs
//   -> create reduce loop
//   -> apply
//
