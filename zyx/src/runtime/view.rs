use super::ir::{IRCompiler, IROp, Reg};
use crate::{dtype::Constant, shape::Axis, DType};
use std::{collections::BTreeMap, fmt::Display, ops::Range};

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct View(Vec<BTreeMap<Axis, RDim>>);

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
    pub(crate) const fn none() -> View {
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
        self.0.last().map_or(1, BTreeMap::len)
    }

    pub(crate) fn shape(&self) -> Vec<usize> {
        self.0.last().map_or_else(
            || vec![1],
            |inner| inner.values().map(|dim| dim.d).collect(),
        )
    }

    pub(crate) fn original_numel(&self) -> usize {
        self.0.first().map_or(1, |inner| {
            inner
                .values()
                .map(|dim| {
                    if dim.st != 0 {
                        usize::try_from(isize::try_from(dim.d).unwrap() - dim.lp - dim.rp).unwrap()
                    } else {
                        1
                    }
                })
                .product()
        })
    }

    pub(crate) fn numel(&self) -> usize {
        self.0
            .last()
            .map_or(1, |inner| inner.values().map(|dim| dim.d).product())
    }

    pub(crate) fn is_contiguous(&self) -> bool {
        self.0.last().map_or(true, |inner| {
            let stride = 1;
            inner
                .values()
                .all(|dim| dim.lp == 0 && dim.rp == 0 && dim.st == stride)
        })
    }

    pub(crate) fn used_axes(&self) -> Vec<usize> {
        self.0
            .last()
            .map_or_else(Vec::new, |inner| inner.keys().copied().collect())
    }

    /// Inserts new loop, shifts all axes greater than axis up by one
    pub(crate) fn insert_loop(&mut self, axis: usize) {
        //println!("Inserting loop at axis {axis}");
        if let Some(inner) = self.0.last_mut() {
            let keys: Vec<Axis> = inner.keys().filter(|&&a| a >= axis).copied().collect();
            for a in keys.into_iter().rev() {
                let dim = inner.remove(&a).unwrap();
                inner.insert(a + 1, dim);
            }
        }
        //println!("After insert loop {self:?}");
    }

    // This will be used if split is not possible.
    // This is used for both reshape and merge
    pub(crate) fn reshape(&mut self, axes: Range<usize>, shape: &[usize]) {
        if let Some(inner) = self.0.last_mut() {
            // TODO if axes range is contiguous, reshape inplace, otherwise create new inner
            let mut ost = 1;
            let mut a = *inner.last_key_value().unwrap().0;
            let mut new_inner = BTreeMap::new();
            let mut axes_i = shape.len();
            while a > 0 {
                if axes.contains(&a) {
                    axes_i -= 1;
                    let st = ost;
                    ost *= shape[axes_i];
                    new_inner.insert(
                        a,
                        RDim {
                            d: shape[axes_i],
                            st,
                            lp: 0,
                            rp: 0,
                        },
                    );
                } else if let Some(dim) = inner.get(&a) {
                    let st = ost;
                    ost *= dim.d;
                    new_inner.insert(
                        a,
                        RDim {
                            d: dim.d,
                            st,
                            lp: 0,
                            rp: 0,
                        },
                    );
                }
                a -= 1;
            }
            self.0.push(new_inner);
        }
    }

    pub(crate) fn split(&mut self, axis: usize, dimensions: &[usize]) {
        // TODO also check if inners can be merged after applying split.
        // For example if axes were merged and then split again, we may be able to remove
        // the last inner view.
        fn split_inner(inner: &mut BTreeMap<usize, RDim>, mut axis: usize, dimensions: &[usize]) {
            //println!("inner {inner:?}, split axis {axis}, dims {dimensions:?}");
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
                axis = axis.saturating_sub(1);
            }
        }
        //println!("Splitting {} axis {axis} into {dimensions:?}", self);
        // if axis contains padding, we have to reshape, otherwise just split
        if let Some(inner) = self.0.last_mut() {
            if let Some(dim) = inner.get_mut(&axis) {
                if dim.lp != 0 || dim.rp != 0 {
                    //todo!("Reshape padded view.");
                    let mut ost = 1;
                    let mut inner = inner
                        .iter()
                        .rev()
                        .map(|(&a, dim)| {
                            let st = ost;
                            ost *= dim.d;
                            (
                                a,
                                RDim {
                                    d: dim.d,
                                    st,
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
            } else {
                let keys: Vec<Axis> = inner.keys().filter(|&&a| a > axis).copied().collect();
                for a in keys.iter().rev() {
                    let dim = inner.remove(&a).unwrap();
                    inner.insert(a + dimensions.len() - 1, dim);
                }
            }
        }
        //println!("Result {}", self);
    }

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
                dim.d = usize::try_from(isize::try_from(dim.d).unwrap() + left_pad + right_pad)
                    .unwrap();
                dim.lp = left_pad;
                dim.rp = right_pad;
            }
        }
    }

    /// Load constant into variable or directly return it if view isn't padded
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(crate) fn ir_for_constant_load(&self, c: &mut IRCompiler, constant: Constant) -> Reg {
        let mut pc = 0;
        let mut old_offset = None;
        //println!("Self {self:?}");
        for inner in self.0.iter().rev() {
            //println!("\n{inner:?}");
            // a = offset / ost % dim
            let mut ost = 1;
            let mut offset = 0;
            for (&a, dim) in inner.iter().rev() {
                let a = Reg::Var(old_offset.map_or_else(
                    || u16::try_from(a).unwrap(),
                    |old_offset| {
                        let a = c.div(Reg::Var(old_offset), Reg::Const(Constant::U32(ost)));
                        ost *= u32::try_from(dim.d).unwrap();
                        c.mod_(
                            Reg::Var(a),
                            Reg::Const(Constant::U32(u32::try_from(dim.d).unwrap())),
                        )
                    },
                ));
                //println!("ost: {ost}, {dim:?}");
                // Offset
                //if dim.st != 0 && dim.d != 1 {
                let t = if dim.lp != 0 {
                    let lp = Reg::Const(Constant::U32(u32::try_from(dim.lp.abs()).unwrap()));
                    Reg::Var(if dim.lp > 0 {
                        c.sub(a, lp)
                    } else {
                        c.add(a, lp)
                    })
                } else {
                    a
                };
                let stride = Reg::Const(Constant::U32(u32::try_from(dim.st).unwrap()));
                offset = c.mad(
                    t,
                    stride,
                    if offset != 0 {
                        Reg::Var(offset)
                    } else {
                        Reg::Const(Constant::U32(0))
                    },
                );
                //}
                // Padding condition
                if dim.lp > 0 {
                    let lp = Reg::Const(Constant::U32(u32::try_from(dim.lp - 1).unwrap()));
                    let t = c.cmpgt(a, lp);
                    pc = c.and(
                        Reg::Var(t),
                        if pc != 0 {
                            Reg::Var(pc)
                        } else {
                            Reg::Const(Constant::Bool(true))
                        },
                    );
                }
                if dim.rp > 0 {
                    let rp = Reg::Const(Constant::U32(
                        u32::try_from(isize::try_from(dim.d).unwrap() - dim.rp).unwrap(),
                    ));
                    let t = c.cmplt(a, rp);
                    pc = c.and(Reg::Var(t), Reg::Var(pc));
                }
            }
            old_offset = Some(offset);
        }
        //if pc != 0 {
        //let pcu32 = c.cast(Reg::Var(pc), DType::U32);
        //offset = c.mul(pcu32, Reg::Var(offset));
        //}
        let dtype = constant.dtype();
        let mut z = Reg::Const(constant);
        if pc != 0 {
            let pcd = c.cast(Reg::Var(pc), dtype);
            // Nullify z if padding condition is false (if there is padding at that index)
            z = Reg::Var(c.mul(pcd, z));
        }
        z
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
        let mut pc = 0;
        let mut offset = 0;
        let mut old_offset = None;
        //println!("Self {self:?}");
        for inner in self.0.iter().rev() {
            //println!("\n{inner:?}");
            // a = offset / ost % dim
            let mut ost = 1;
            offset = 0;
            for (&a, dim) in inner.iter().rev() {
                let a = Reg::Var(old_offset.map_or_else(
                    || u16::try_from(a).unwrap(),
                    |old_offset| {
                        let a = c.div(Reg::Var(old_offset), Reg::Const(Constant::U32(ost)));
                        ost *= u32::try_from(dim.d).unwrap();
                        c.mod_(
                            Reg::Var(a),
                            Reg::Const(Constant::U32(u32::try_from(dim.d).unwrap())),
                        )
                    },
                ));
                //println!("ost: {ost}, {dim:?}");
                // Offset
                //if dim.st != 0 && dim.d != 1 {
                let t = if dim.lp != 0 {
                    let lp = Reg::Const(Constant::U32(u32::try_from(dim.lp.abs()).unwrap()));
                    Reg::Var(if dim.lp > 0 {
                        c.sub(a, lp)
                    } else {
                        c.add(a, lp)
                    })
                } else {
                    a
                };
                let stride = Reg::Const(Constant::U32(u32::try_from(dim.st).unwrap()));
                offset = c.mad(
                    t,
                    stride,
                    if offset != 0 {
                        Reg::Var(offset)
                    } else {
                        Reg::Const(Constant::U32(0))
                    },
                );
                //}
                // Padding condition
                if dim.lp > 0 {
                    let lp = Reg::Const(Constant::U32(u32::try_from(dim.lp - 1).unwrap()));
                    let t = c.cmpgt(a, lp);
                    pc = c.and(
                        Reg::Var(t),
                        if pc != 0 {
                            Reg::Var(pc)
                        } else {
                            Reg::Const(Constant::Bool(true))
                        },
                    );
                }
                if dim.rp > 0 {
                    let rp = Reg::Const(Constant::U32(
                        u32::try_from(isize::try_from(dim.d).unwrap() - dim.rp).unwrap(),
                    ));
                    let t = c.cmplt(a, rp);
                    pc = c.and(
                        Reg::Var(t),
                        if pc != 0 {
                            Reg::Var(pc)
                        } else {
                            Reg::Const(Constant::Bool(true))
                        },
                    );
                }
            }
            old_offset = Some(offset);
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
                if dim.st != 0 && dim.d != 1 {
                    let stride = Reg::Const(Constant::U32(u32::try_from(dim.st).unwrap()));
                    offset = c.mad(
                        Reg::Var(u16::try_from(a).unwrap()),
                        stride,
                        if offset != 0 {
                            Reg::Var(offset)
                        } else {
                            Reg::Const(Constant::U32(0))
                        },
                    );
                }
            }
        }
        c.ops.push(IROp::Store {
            address,
            x: var,
            offset: if offset != 0 {
                Reg::Var(offset)
            } else {
                Reg::Const(Constant::U32(0))
            },
        });
    }
}

impl Display for View {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(inner) = self.0.last() {
            f.write_fmt(format_args!(
                "V(ax{:?} sh{:?} st{:?} pd{:?})",
                inner.keys().copied().collect::<Vec<usize>>(),
                inner.values().map(|d| d.d).collect::<Vec<usize>>(),
                inner.values().map(|d| d.st).collect::<Vec<usize>>(),
                inner
                    .values()
                    .map(|d| (d.lp, d.rp))
                    .rev()
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
    //println!("{view:?}");
    assert_eq!(view.rank(), 3);
    assert_eq!(view.used_axes(), [1, 2, 5]);
    assert_eq!(view.shape(), [2, 3, 4]);
}
