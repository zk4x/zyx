use super::ir::{IRCompiler, IROp, Reg};
use crate::{dtype::Constant, shape::Axis, DType};
use std::{collections::BTreeMap, fmt::Display};

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
            .map(|(&d, &axis)| {
                let temp = stride;
                stride *= d;
                (
                    axis,
                    RDim {
                        st: temp,
                        d,
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

    pub(crate) fn split(&mut self, axis: usize, dimensions: &[usize]) {
        fn split_inner(inner: &mut BTreeMap<usize, RDim>, mut axis: usize, dimensions: &[usize]) {
            let keys: Vec<Axis> = inner.keys().copied().collect();
            for a in keys.into_iter().rev() {
                if a > axis {
                    let dim = inner.remove(&a).unwrap();
                    inner.insert(a + dimensions.len() - 1, dim);
                }
            }
            //println!("inner {inner:?}");
            // Then remove axis and get it's stride
            let mut stride = inner.remove(&axis).unwrap().st;
            //println!("inner {inner:?}");
            // At last insert all new dimensions
            axis += dimensions.len() - 1;
            for &d in dimensions.iter().rev() {
                assert!(inner
                    .insert(
                        axis,
                        RDim {
                            d,
                            st: stride,
                            lp: 0,
                            rp: 0,
                        },
                    )
                    .is_none());
                stride *= d;
                if axis > 0 {
                    axis -= 1;
                }
            }
        }
        //println!("Splitting {} axis {axis} into {dimensions:?}", self);
        // if axis contains padding, we have to reshape, otherwise just split
        if let Some(inner) = self.0.last_mut() {
            if let Some(dim) = inner.get_mut(&axis) {
                if dim.lp != 0 || dim.rp != 0 {
                    //todo!("Reshape padded view.");
                    let mut inner = inner
                        .iter()
                        .map(|(&a, dim)| {
                            (
                                a,
                                RDim {
                                    d: dim.d,
                                    st: dim.st,
                                    lp: 0,
                                    rp: 0,
                                },
                            )
                        })
                        .collect();
                    split_inner(&mut inner, axis, dimensions);
                    self.0.push(inner);
                } else {
                    //println!("inner {inner:?}");
                    // First shift axes > axis by dimensions.len()
                    split_inner(inner, axis, dimensions);
                    //println!("done {inner:?}");
                }
            }
        }
        //println!("Result {}", self);
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
    pub(crate) fn ir_for_constant_load(&self, c: &mut IRCompiler, constant: Reg) -> Reg {
        let _ = constant;
        let _ = c;
        todo!()
    }

    /// Load from address into variable
    pub(crate) fn ir_for_indexed_load(
        &self,
        c: &mut IRCompiler,
        address: u16,
        dtype: DType,
    ) -> u16 {
        // With padding, right padding does not affect offset
        // offset = (a0-lp0)*st0 + a1*st1
        // Padding condition, negative right padding does not affect it
        // pc = a0 > lp0-1 && a0 < d0-rp0
        // pc = pc.cast(dtype)
        // x = pc * value[offset]
        // Last view
        let mut offset = 0;
        let mut pc = 0;
        if let Some(inner) = self.0.last() {
            for (&a, dim) in inner {
                // Offset
                if dim.st != 0 {
                    let t = if dim.lp != 0 {
                        let lp = Reg::Const(Constant::U32(
                            if dim.lp > 0 { dim.lp } else { -dim.lp } as u32,
                        ));
                        c.sub(Reg::Var(a as u16), lp)
                    } else {
                        a as u16
                    };
                    let stride = Reg::Const(Constant::U32(dim.st as u32));
                    offset = c.mad(Reg::Var(t), stride, if offset != 0 { Reg::Var(offset) } else { Reg::Const(Constant::U32(0)) });
                }
                // Padding condition
                if dim.lp > 0 {
                    let lp = Reg::Const(Constant::U32((dim.lp - 1) as u32));
                    let t = c.cmplt(Reg::Var(a as u16), lp);
                    pc = c.and(Reg::Var(t), if pc != 0 { Reg::Var(pc) } else { Reg::Const(Constant::Bool(true)) });
                }
                if dim.rp > 0 {
                    let rp = Reg::Const(Constant::U32((dim.d as isize - dim.rp) as u32));
                    let t = c.cmpgt(Reg::Var(a as u16), rp);
                    pc = c.and(Reg::Var(t), Reg::Var(pc));
                }
            }
        }
        // All previous views
        if self.0.len() > 1 {
            for inner in &self.0[..self.0.len()-2] {
                // a = offset / ost % dim
                let mut ost = 1;
                for (_, &RDim { d, st, lp, rp }) in inner.iter().rev() {
                    let a = c.div(offset, Reg::Const(Constant::U32(0)));
                    let a = c.mod(a, d);
                    ost *= d;
                }
                todo!()
            }
        }
        if pc != 0 {
            let pcu32 = c.cast(Reg::Var(pc), DType::U32);
            offset = c.mul(pcu32, Reg::Var(offset));
        }
        let mut z = c.load(address, Reg::Var(offset), dtype);
        if pc != 0 {
            let pcd = c.cast(Reg::Var(pc), dtype);
            // Nullify z if padding condition is false (if there is padding at that index)
            z = c.mul(pcd, Reg::Var(z));
        }
        z
    }

    /// Store from variable into address
    pub(crate) fn ir_for_indexed_store(&self, c: &mut IRCompiler, address: u16, var: Reg) {
        let mut offset = 0;
        if let Some(inner) = self.0.last() {
            for (&a, dim) in inner {
                if dim.st != 0 {
                    let stride = Reg::Const(Constant::U32(dim.st as u32));
                    offset = c.mad(Reg::Var(a as u16), stride, if offset != 0 { Reg::Var(offset) } else { Reg::Const(Constant::U32(0)) });
                }
            }
        }
        c.ops.push(IROp::Store {
            address,
            x: var,
            offset: if offset != 0 { Reg::Var(offset) } else { Reg::Const(Constant::U32(0)) },
        });
    }
}

impl Display for View {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(inner) = self.0.last() {
            f.write_fmt(format_args!(
                "V(ax{:?} sh{:?} st{:?} pd{:?})",
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

#[test]
fn view_split() {
    let mut view = View::contiguous(&[3, 1, 4, 2]);
    view.split(2, &[2, 2, 1]);
    assert_eq!(view.shape(), [3, 1, 2, 2, 1, 2]);
    view.split(0, &[1, 3, 1]);
    assert_eq!(view.shape(), [1, 3, 1, 1, 2, 2, 1, 2]);
}

#[test]
fn view_binded() {
    let view = View::binded(&[4, 2, 3], &[5, 1, 2]);
    println!("{view:?}");
    assert_eq!(view.rank(), 3);
    assert_eq!(view.used_axes(), [1, 2, 5]);
    assert_eq!(view.shape(), [2, 3, 4]);
}
