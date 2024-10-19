use std::{collections::BTreeMap, fmt::Display};

use crate::{dtype::Constant, shape::Axis};

use super::ir::{IRDType, IROp, Var};

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct View(Vec<BTreeMap<Axis, RDim>>);

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RDim {
    d: usize,  // dim
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
        let mut stride = 1;
        View(vec![shape
            .iter()
            .enumerate()
            .rev()
            .map(|(axis, dim)| {
                let temp = stride;
                stride *= dim;
                (
                    axis,
                    RDim {
                        st: temp,
                        d: *dim,
                        lp: 0,
                        rp: 0,
                    },
                )
            })
            .collect()])
    }

    pub(crate) fn binded(shape: &[usize], axes: &[usize]) -> View {
        todo!()
    }

    pub(crate) fn rank(&self) -> usize {
        if let Some(inner) = self.0.last() {
            inner.len()
        } else {
            1
        }
    }

    pub(crate) fn shape(&self) -> Vec<usize> {
        if let Some(inner) = self.0.last() {
            inner.values().map(|dim| dim.d).collect()
        } else {
            vec![1]
        }
    }

    pub(crate) fn original_numel(&self) -> usize {
        if let Some(inner) = self.0.first() {
            inner
                .values()
                .map(|dim| {
                    if dim.st != 0 {
                        (dim.d as isize + dim.lp + dim.rp) as usize
                    } else {
                        1
                    }
                })
                .product()
        } else {
            1
        }
    }

    pub(crate) fn numel(&self) -> usize {
        if let Some(inner) = self.0.last() {
            inner.values().map(|dim| dim.d).product()
        } else {
            1
        }
    }

    pub(crate) fn is_contiguous(&self) -> bool {
        todo!()
    }

    pub(crate) fn used_axes(&self) -> Vec<usize> {
        if let Some(inner) = self.0.last() {
            inner.keys().copied().collect()
        } else {
            Vec::new()
        }
    }

    /// Inserts new loop, shifts all axes greater than axis up
    pub(crate) fn insert_loop(&mut self, axis: usize) {
        todo!()
    }

    // TODO this will be used if split is not possible
    /*pub(crate) fn reshape(&mut self, shape: &[usize]) {
        todo!()
    }*/

    pub(crate) fn split(&mut self, axis: usize, dimensions: &[usize]) {
        // if axis contains padding, we have to reshape, otherwise just split
        todo!()
    }

    pub(crate) fn permute(&mut self, axes: &[usize]) {
        // Move around strides, dim, rp and lp
        todo!()
    }

    pub(crate) fn expand(&mut self, axis: usize, ndim: usize) {
        if let Some(inner) = self.0.last_mut() {
            if let Some(dim) = inner.get_mut(&axis) {
                assert_eq!(dim.d, 1);
                dim.d = ndim;
                dim.st = 0;
            }
        }
    }

    pub(crate) fn pad(&mut self, axis: usize, left_pad: isize, right_pad: isize) {
        todo!()
    }

    /// Load constant into variable or directly return it if view isn't padded
    pub(crate) fn ir_for_constant_load(
        &self,
        constant: Constant,
        registers: &mut Vec<(IRDType, u32)>,
    ) -> (Vec<IROp>, Var) {
        todo!()
    }

    /// Load from address into variable
    pub(crate) fn ir_for_indexed_load(
        &self,
        address: u16,
        registers: &mut Vec<(IRDType, u32)>,
        ops: &mut Vec<IROp>,
    ) -> Var {
        todo!()
    }

    /// Store from variable into address
    pub(crate) fn ir_for_indexed_store(
        &self,
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
