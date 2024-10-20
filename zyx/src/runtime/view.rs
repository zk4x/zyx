use std::{collections::BTreeMap, fmt::Display};
use crate::{dtype::Constant, shape::Axis};
use super::ir::{get_empty_register, IRDType, IROp, IRVec, Var};

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

    pub(crate) fn contiguous(shape: &[usize]) -> View {
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
        let mut stride = 1;
        View(vec![shape
            .iter()
            .zip(axes)
            .rev()
            .map(|(&axis, dim)| {
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
        if let Some(inner) = self.0.last() {
            let stride = 1;
            inner
                .values()
                .all(|dim| dim.lp == 0 && dim.rp == 0 && dim.st == stride)
        } else {
            true
        }
    }

    pub(crate) fn used_axes(&self) -> Vec<usize> {
        if let Some(inner) = self.0.last() {
            inner.keys().copied().collect()
        } else {
            Vec::new()
        }
    }

    /// Inserts new loop, shifts all axes greater than axis up by one
    pub(crate) fn insert_loop(&mut self, axis: usize) {
        if let Some(inner) = self.0.last_mut() {
            let keys: Vec<Axis> = inner.keys().filter(|&&a| a > axis).copied().collect();
            for a in keys {
                let dim = inner.remove(&a).unwrap();
                inner.insert(a + 1, dim);
            }
        }
    }

    // TODO this will be used if split or merge are not possible
    /*pub(crate) fn reshape(&mut self, shape: &[usize]) {
        todo!()
    }*/

    pub(crate) fn split(&mut self, mut axis: usize, dimensions: &[usize]) {
        // if axis contains padding, we have to reshape, otherwise just split
        if let Some(inner) = self.0.last_mut() {
            if let Some(dim) = inner.get_mut(&axis) {
                if dim.lp != 0 || dim.rp != 0 {
                    todo!("Reshape padded view.");
                } else {
                    // First shift axes > axis by dimensions.len()
                    let keys: Vec<Axis> = inner.keys().copied().collect();
                    for a in keys {
                        if a > axis {
                            let dim = inner.remove(&a).unwrap();
                            inner.insert(a + dimensions.len(), dim);
                        }
                    }
                    // Then remove axis and get it's stride
                    let mut stride = inner.remove(&axis).unwrap().st;
                    // At last insert all new dimensions
                    for &d in dimensions.iter().rev() {
                        inner.insert(
                            axis,
                            RDim {
                                d,
                                st: stride,
                                lp: 0,
                                rp: 0,
                            },
                        );
                        stride *= d;
                        if axis > 0 {
                            axis -= 1;
                        }
                    }
                }
            }
        }
    }

    // TODO function for merging multiple axes together, must be used in case of very
    // high dimensionality of tensors. It can also be used to make reduce over multiple axes
    // reduce over single axis and such.
    //pub(crate) fn merge(&mut self, axes: &[usize]) {}

    pub(crate) fn permute(&mut self, axes: &[usize]) {
        // Move around strides, dim, rp and lp
        let inner = self.0.last_mut().unwrap();
        let keys: Vec<Axis> = inner.keys().copied().collect();
        assert_eq!(keys.len(), axes.len());
        let mut new = BTreeMap::new();
        for (a, k) in axes.iter().zip(keys) {
            let dim = inner.remove(a).unwrap();
            new.insert(k, dim);
        }
        *inner = new;
    }

    pub(crate) fn expand(&mut self, axis: usize, ndim: usize) {
        if let Some(inner) = self.0.last_mut() {
            if let Some(dim) = inner.get_mut(&axis) {
                assert_eq!(dim.d, 1);
                assert_eq!(dim.lp, 0);
                assert_eq!(dim.rp, 0);
                dim.d = ndim;
                dim.st = 0;
            }
        }
    }

    pub(crate) fn pad(&mut self, axis: usize, left_pad: isize, right_pad: isize) {
        if let Some(inner) = self.0.last_mut() {
            if let Some(dim) = inner.get_mut(&axis) {
                dim.d = (dim.d as isize + left_pad + right_pad) as usize;
                dim.lp = left_pad;
                dim.rp = right_pad;
            }
        }
    }

    /// Load constant into variable or directly return it if view isn't padded
    pub(crate) fn ir_for_constant_load(
        &self,
        constant: Constant,
        registers: &mut Vec<(IRDType, u32)>,
    ) -> (Vec<IROp>, Var) {
        let _ = constant;
        let _ = registers;
        todo!()
    }

    /// Load from address into variable
    pub(crate) fn ir_for_indexed_load(
        &self,
        address: u16,
        ir_dtype: IRDType,
        rc: u32,
        registers: &mut Vec<(IRDType, u32)>,
        ops: &mut Vec<IROp>,
    ) -> Var {
        let id = get_empty_register(registers, ir_dtype, rc);
        let offset = get_empty_register(registers, IRDType::U32(IRVec::Scalar), 0);
        ops.push(IROp::Set {
            z: offset,
            value: Constant::U32(0),
        });
        let offset = Var::Id(offset);
        if let Some(inner) = self.0.last() {
            for (a, dim) in inner {
                if dim.st != 0 {
                    ops.push(IROp::MAdd {
                        z: offset,
                        a: Var::Id(*a as u16),
                        b: Var::Const(Constant::U32(dim.st as u32)),
                        c: offset,
                    });
                }
            }
        }
        let z = Var::Id(id);
        ops.push(IROp::Load {
            z,
            address,
            offset,
        });
        z
    }

    /// Store from variable into address
    pub(crate) fn ir_for_indexed_store(
        &self,
        address: u16,
        var: Var,
        registers: &mut Vec<(IRDType, u32)>,
        ops: &mut Vec<IROp>,
    ) {
        let offset = get_empty_register(registers, IRDType::U32(IRVec::Scalar), 0);
        ops.push(IROp::Set {
            z: offset,
            value: Constant::U32(0),
        });
        let offset = Var::Id(offset);
        if let Some(inner) = self.0.last() {
            for (a, dim) in inner {
                if dim.st != 0 {
                    ops.push(IROp::MAdd {
                        z: offset,
                        a: Var::Id(*a as u16),
                        b: Var::Const(Constant::U32(dim.st as u32)),
                        c: offset,
                    });
                }
            }
        }
        ops.push(IROp::Store {
            address,
            x: var,
            offset,
        });
    }
}

impl Display for View {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(inner) = self.0.last() {
            f.write_fmt(format_args!(
                "V:S ax{:?} sh{:?} st{:?} pd{:?}",
                inner.keys().map(|&a| a).collect::<Vec<usize>>(),
                inner.values().map(|d| d.d).collect::<Vec<usize>>(),
                inner.values().map(|d| d.st).collect::<Vec<usize>>(),
                inner
                    .values()
                    .map(|d| (d.lp, d.rp))
                    .collect::<Vec<(isize, isize)>>()
            ))
        } else {
            f.write_str("none")
        }
    }
}
