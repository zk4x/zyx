#![allow(unused)]

use crate::{tensor::{Tensor, TensorId}, Set, RT};

pub struct StaticGraph {
    inputs: Set<TensorId>,
    outputs: Set<TensorId>,
    graph: Vec<GraphOp>,
}

impl Drop for StaticGraph {
    fn drop(&mut self) {
        let mut rt = RT.lock();
        for &tid in self.inputs.union(&self.outputs) {
            rt.release(tid);
        }
    }
}

impl StaticGraph {
    /// Create new static graph using inputs and outputs.
    /// Inputs are tensors that can be changed during each forward pass.
    /// Outputs are tensors that get realized during forward pass.
    pub fn new(inputs: impl IntoIterator<Item = Tensor>, outputs: impl IntoIterator<Item = Tensor>) -> Self {

        // TODO keep order of inputs and resolve the fact, that input IDs can change, so there needs to be some
        // perhaps some interior mutability to keep the graph valid.
        // But actually we don't need to do that. We only need to work on the level of buffer IDs.
        // The inputs need to be realized before passing them through the compiler and forward pass
        // and we only need to map buffers correctly, once we are compiled down to kernels, only buffers matter,
        // not tensors.

        let inputs: Set<TensorId> = inputs.into_iter().map(|t| t.id).collect();
        let outputs: Set<TensorId> = outputs.into_iter().map(|t| t.id).collect();
        let mut rt = RT.lock();
        for &tid in inputs.union(&outputs) {
            rt.retain(tid);
        }
        let graph = rt.compile_graph(&inputs, &outputs);
        Self {
            inputs,
            outputs,
            graph,
        }
    }

    /// Launch the graph with given inputs.
    #[allow(clippy::needless_pass_by_value)]
    pub fn forward(&mut self, inputs: impl IntoIterator<Item = Tensor>) {
        let _ = inputs;
        todo!()
    }
}

pub enum GraphOp {
    MemoryAllocate,
    MemoryFree,
    MemoryCopy,
    KernelLaunch,
}
