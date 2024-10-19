use std::fmt::Display;

use crate::dtype::Constant;

use super::ir::{IRDType, IROp, Var};

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct View(Vec<Vec<RDim>>);

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RDim {
    a: usize, // axis
    d: usize, // dim
    st: usize, // stride
    lp: isize, // left pad
    rp: isize, // right pad
}

impl View {
    /// Create empty view for scalars
    pub(crate) fn none() -> View {
        View(Vec::new())
    }

    pub(crate) fn new(shape: &[usize]) -> View {
        todo!()
    }

    pub(crate) fn binded(shape: &[usize], axes: &[usize]) -> View {
        todo!()
    }

    pub(crate) fn rank(&self) -> usize {
        todo!()
    }

    pub(crate) fn shape(&self) -> Vec<usize> {
        todo!()
    }

    pub(crate) fn original_numel(&self) -> usize {
        todo!()
    }

    pub(crate) fn numel(&self) -> usize {
        todo!()
    }

    pub(crate) fn is_contiguous(&self) -> bool {
        todo!()
    }

    pub(crate) fn used_axes(&self) -> Vec<usize> {
        todo!()
    }

    /// Inserts new loop, shifts all axes greater than axis up
    pub(crate) fn insert_loop(&mut self, axis: usize) {
        todo!()
    }

    pub(crate) fn reshape(&mut self, shape: &[usize]) {
        todo!()
    }

    pub(crate) fn split(&mut self, axis: usize, dimensions: &[usize]) {
        todo!()
    }

    pub(crate) fn permute(&mut self, axes: &[usize]) {
        todo!()
    }

    pub(crate) fn expand(&mut self, axis: usize, dim: usize) {
        todo!()
    }

    pub(crate) fn pad(&mut self, axis: usize, left_pad: isize, right_pad: isize) {
        todo!()
    }

    /// Load constant into variable or directly return it if view isn't padded
    pub(crate) fn ir_for_constant_load(&self,
        constant: Constant,
        registers: &mut Vec<(IRDType, u32)>,
    ) -> (Vec<IROp>, Var) {
        todo!()
    }

    /// Load from address into variable
    pub(crate) fn ir_for_indexed_load(&self,
        address: u16,
        registers: &mut Vec<(IRDType, u32)>,
        ops: &mut Vec<IROp>,
    ) -> Var {
        todo!()
    }

    /// Store from variable into address
    pub(crate) fn ir_for_indexed_store(&self,
        address: u16,
        var: Var,
        registers: &mut Vec<(IRDType, u32)>,
        ops: &mut Vec<IROp>,
    ) {
        todo!()
    }
}

impl Display for View {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}
