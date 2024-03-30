use std::collections::{BTreeMap, BTreeSet};
use std::prelude::rust_2015::Vec;
use zyx_core::error::ZyxError;
use zyx_core::node::Node;
use zyx_core::runtime::RuntimeBackend;
use zyx_core::scalar::Scalar;
use zyx_core::tensor::Id;
use crate::{CompiledBackend, Compiler};

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
        if let Some(mut buffer) = self.buffers.remove(&x) {
            //std::println!("Dropping buffer {p} out of total {} buffers", self.buffers.len());
            self.compiler.deallocate_mem(&mut buffer)?;
        }
        Ok(())
    }

    fn compile_graph(&mut self, global_rcs: &[u32], nodes: &[Node], to_eval: &BTreeSet<Id>) -> Result<Self::CompiledGraph, ZyxError> {
        // Find the best order of execution of nodes
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
    }

    fn launch_graph(&mut self, graph: &Self::CompiledGraph) -> Result<(), ZyxError> {
        let buffers: Vec<&C::Buffer> = graph.args.iter().map(|id| self.buffers.get(id).unwrap()).collect();
        self.compiler.launch_program(&graph.program, &buffers, graph.flop, graph.bytes)
    }
}
