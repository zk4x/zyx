use super::ir::{IRCompiler, IROp, Reg};
use crate::{dtype::Constant, DType};
use std::{fmt::Display, ops::Range};

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct View(Vec<Vec<RDim>>);

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
        let mut res: Vec<RDim> = shape
            .iter()
            .rev()
            .map(|&d| {
                let temp = stride;
                stride *= d;
                RDim {
                    st: temp,
                    d,
                    lp: 0,
                    rp: 0,
                }
            })
            .collect();
        res.reverse();
        View(vec![res])
    }

    pub(crate) fn binded(shape: &[usize], axes: &[usize], rank: usize) -> View {
        let mut stride = 1;
        assert!(axes.iter().all(|&a| a < rank));
        let mut a: usize = rank;
        let mut res = Vec::with_capacity(a);
        while a > 0 {
            a -= 1;
            if let Some(i) = axes.iter().position(|axis| *axis == a) {
                let st = stride;
                let d = shape[i];
                stride *= d;
                res.push(RDim {
                    d,
                    st,
                    lp: 0,
                    rp: 0,
                });
            } else {
                res.push(RDim {
                    d: 1,
                    st: 0,
                    lp: 0,
                    rp: 0,
                });
            }
        }
        res.reverse();
        View(vec![res])
    }

    pub(crate) fn rank(&self) -> usize {
        self.0.last().map_or(1, Vec::len)
    }

    pub(crate) fn shape(&self) -> Vec<usize> {
        self.0
            .last()
            .map_or_else(|| vec![1], |inner| inner.iter().map(|dim| dim.d).collect())
    }

    pub(crate) fn original_numel(&self) -> usize {
        self.0.first().map_or(1, |inner| {
            inner
                .iter()
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
            .map_or(1, |inner| inner.iter().map(|dim| dim.d).product())
    }

    pub(crate) fn is_contiguous(&self) -> bool {
        self.0.last().map_or(true, |inner| {
            let stride = 1;
            inner
                .iter()
                .all(|dim| dim.lp == 0 && dim.rp == 0 && dim.st == stride)
        })
    }

    pub(crate) fn used_axes(&self) -> Vec<usize> {
        self.0.last().map_or_else(Vec::new, |inner| {
            (0..inner.len()).filter(|&a| inner[a].st != 0).collect()
        })
    }

    /// Inserts new loop, shifts all axes greater than axis up by one
    pub(crate) fn insert_loop(&mut self, axis: usize) {
        //println!("Inserting loop at axis {axis}");
        if let Some(inner) = self.0.last_mut() {
            assert!(axis < inner.len());
            let st = inner[axis].st;
            inner.insert(
                axis,
                RDim {
                    d: 1,
                    st,
                    lp: 0,
                    rp: 0,
                },
            );
        }
        //println!("After insert loop {self:?}");
    }

    // This is used for both reshape and merge and split
    pub(crate) fn reshape(&mut self, axes: Range<usize>, shape: &[usize]) {
        if self.0.is_empty() {
            return;
        }
        //println!("Reshape {self} axes {axes:?} into shape {shape:?}");
        assert!(axes.end <= self.0.last().map_or(1, |inner| inner.len()));
        assert_eq!(
            self.0.last().unwrap()[axes.clone()]
                .iter()
                .map(|dim| dim.d)
                .product::<usize>(),
            shape.iter().product::<usize>()
        );
        if let Some(inner) = self.0.last_mut() {
            let mut contiguous = true;
            let mut a = inner.len();
            let mut stride = 1;
            let mut ost = 1;
            while a > axes.start {
                a -= 1;
                let dim = &inner[a];
                if a >= axes.end - 1 {
                    if dim.st != 0 {
                        stride = dim.st * dim.d;
                        ost = dim.st;
                    }
                } else {
                    let st = stride;
                    stride *= dim.d;
                    //println!("a = {a} stride = {stride} dim = {dim:?}");
                    if dim.st != st || dim.lp != 0 || dim.rp != 0 {
                        contiguous = false;
                        break;
                    }
                }
            }
            if axes.clone().any(|a| inner[a].st == 0) {
                contiguous = false;
            }
            let mut expanded_reshape = false;
            // If all reshaped axes are expanded
            if axes.clone().all(|a| inner[a].st == 0) {
                contiguous = true;
                expanded_reshape = true;
            }
            if axes.clone().any(|a| inner[a].lp != 0 || inner[a].rp != 0) {
                contiguous = false;
            }

            if contiguous {
                //println!("Reshape contiguous");
                for a in axes.clone().rev() {
                    let dim = inner.remove(a);
                    assert_eq!(dim.lp, 0);
                    assert_eq!(dim.rp, 0);
                }
                for &d in shape.iter().rev() {
                    let st = if expanded_reshape { 0 } else { ost };
                    ost *= d;
                    inner.insert(
                        axes.start,
                        RDim {
                            d,
                            st,
                            lp: 0,
                            rp: 0,
                        },
                    );
                }
            } else {
                //println!("Reshape non-contiguous");
                let mut old_shape = self.shape();
                old_shape.splice(axes, shape.iter().copied());
                let mut stride = 1;
                let mut res: Vec<RDim> = old_shape
                    .iter()
                    .rev()
                    .map(|dim| {
                        let st = stride;
                        stride *= dim;
                        RDim {
                            st,
                            d: *dim,
                            lp: 0,
                            rp: 0,
                        }
                    })
                    .collect();
                res.reverse();
                self.0.push(res);
            }
        }
        //println!("After reshape: {self}\n");
    }

    pub(crate) fn permute(&mut self, axes: &[usize]) {
        // Move around strides, dim, rp and lp
        let inner = self.0.last_mut().unwrap();
        assert_eq!(inner.len(), axes.len());
        let mut new = Vec::with_capacity(axes.len());
        for &a in axes {
            let dim = inner[a].clone();
            new.push(dim);
        }
        *inner = new;
    }

    pub(crate) fn expand(&mut self, axis: usize, ndim: usize) {
        //println!("View expand {self} axis = {axis} to ndim {ndim}");
        if let Some(inner) = self.0.last_mut() {
            if let Some(dim) = inner.get_mut(axis) {
                assert!(dim.d == ndim || dim.d == 1);
                assert_eq!(dim.lp, 0);
                assert_eq!(dim.rp, 0);
                dim.d = ndim;
                dim.st = 0;
            } else {
                unreachable!("Expand on nonexistent axis.");
            }
        }
    }

    pub(crate) fn pad(&mut self, axis: usize, left_pad: isize, right_pad: isize) {
        let mut old_shape = self.shape();
        if let Some(inner) = self.0.last_mut() {
            let mut dim = &mut inner[axis];
            dim.d =
                usize::try_from(isize::try_from(dim.d).unwrap() + left_pad + right_pad).unwrap();
            // TODO this is possible only if original padding has the same sign or is zero
            // otherwise we need to create a new inner
            if dim.lp < 0 && left_pad > 0 {
                dim.d = (dim.d as isize - left_pad) as usize;
                let mut stride = 1;
                let mut res: Vec<RDim> = old_shape
                    .iter()
                    .enumerate()
                    .rev()
                    .map(|(a, &d)| {
                        let st = stride;
                        stride *= d;
                        RDim {
                            st,
                            d: if a == axis {
                                (d as isize + left_pad) as usize
                            } else {
                                d
                            },
                            lp: if a == axis { left_pad } else { 0 },
                            rp: 0,
                        }
                    })
                    .collect();
                res.reverse();
                self.0.push(res);
                dim = &mut self.0.last_mut().unwrap()[axis];
            } else {
                dim.lp += left_pad;
            }
            if dim.rp < 0 && right_pad > 0 {
                dim.d = (dim.d as isize - right_pad) as usize;
                let old_shape = self.shape();
                let mut stride = 1;
                let mut res: Vec<RDim> = old_shape
                    .iter()
                    .enumerate()
                    .rev()
                    .map(|(a, &d)| {
                        let st = stride;
                        stride *= d;
                        RDim {
                            st,
                            d: if a == axis {
                                (d as isize + right_pad) as usize
                            } else {
                                d
                            },
                            lp: 0,
                            rp: if a == axis { right_pad } else { 0 },
                        }
                    })
                    .collect();
                res.reverse();
                self.0.push(res);
            } else {
                dim.rp += right_pad;
            }
        }
        old_shape[axis] = (old_shape[axis] as isize + left_pad + right_pad) as usize;
        assert_eq!(self.shape(), old_shape);
    }

    /// Load constant into variable or directly return it if view isn't padded
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(crate) fn ir_for_constant_load(&self, c: &mut IRCompiler, constant: Constant) -> Reg {
        let mut pc = Reg::Const(Constant::Bool(true));
        let mut old_offset: Option<Reg> = None;
        //println!("Self {self:?}");
        for inner in self.0.iter().rev() {
            //println!("\n{inner:?}");
            // a = offset / ost % dim
            let mut ost = 1;
            let mut offset = Reg::Const(Constant::U32(0));
            for (a, dim) in inner.iter().enumerate().rev() {
                let a: Reg = old_offset.map_or_else(
                    || Reg::Var(u16::try_from(a).unwrap()),
                    |old_offset| {
                        let a = c.div(old_offset, Reg::Const(Constant::U32(ost)));
                        ost *= u32::try_from(dim.d).unwrap();
                        c.mod_(a, Reg::Const(Constant::U32(u32::try_from(dim.d).unwrap())))
                    },
                );
                //println!("ost: {ost}, {dim:?}");
                // Offset
                //if dim.st != 0 && dim.d != 1 {
                let t = if dim.lp != 0 {
                    let lp = Reg::Const(Constant::U32(u32::try_from(dim.lp.abs()).unwrap()));
                    if dim.lp > 0 {
                        c.sub(a, lp)
                    } else {
                        c.add(a, lp)
                    }
                } else {
                    a
                };
                let stride = Reg::Const(Constant::U32(u32::try_from(dim.st).unwrap()));
                offset = c.mad(t, stride, offset);
                //}
                // Padding condition
                if dim.lp > 0 {
                    let lp = Reg::Const(Constant::U32(u32::try_from(dim.lp - 1).unwrap()));
                    let t = c.cmpgt(a, lp);
                    pc = c.and(t, pc);
                }
                if dim.rp > 0 {
                    let rp = Reg::Const(Constant::U32(
                        u32::try_from(isize::try_from(dim.d).unwrap() - dim.rp).unwrap(),
                    ));
                    let t = c.cmplt(a, rp);
                    pc = c.and(t, pc);
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
        let pcd = c.cast(pc, dtype);
        // Nullify z if padding condition is false (if there is padding at that index)
        z = c.mul(pcd, z);
        z
    }

    /// Load from address into variable
    pub(crate) fn ir_for_indexed_load(
        &self,
        c: &mut IRCompiler,
        address: u16,
        dtype: DType,
    ) -> Reg {
        // With padding, right padding does not affect offset
        // offset = (a0-lp0)*st0 + a1*st1
        // Padding condition, negative right padding does not affect it
        // pc = a0 > lp0-1 && a0 < d0-rp0
        // pc = pc.cast(dtype)
        // x = pc * value[offset]
        // Last view
        let mut pc = Reg::Const(Constant::Bool(true));
        let mut offset = Reg::Const(Constant::U32(0));
        let mut old_offset: Option<Reg> = None;
        //println!("View");
        //for inner in self.0.iter() { println!("{inner:?}") }
        //println!();
        for inner in self.0.iter().rev() {
            //println!("\n{inner:?}");
            // a = offset / ost % dim
            let mut ost = 1;
            offset = Reg::Const(Constant::U32(0));
            for (a, dim) in inner.iter().enumerate().rev() {
                let a = old_offset.map_or_else(
                    || Reg::Var(u16::try_from(a).unwrap()),
                    |old_offset| {
                        let a = c.div(old_offset, Reg::Const(Constant::U32(ost)));
                        ost *= u32::try_from(dim.d).unwrap();
                        c.mod_(a, Reg::Const(Constant::U32(u32::try_from(dim.d).unwrap())))
                    },
                );
                //println!("ost: {ost}, a: {a:?}, {dim:?}");
                // Offset
                let t = if dim.lp != 0 {
                    let lp = Reg::Const(Constant::U32(u32::try_from(dim.lp.abs()).unwrap()));
                    if dim.lp > 0 {
                        c.sub(a, lp)
                    } else {
                        c.add(a, lp)
                    }
                } else {
                    a
                };
                let stride = Reg::Const(Constant::U32(u32::try_from(dim.st).unwrap()));
                offset = c.mad(t, stride, offset);
                // Padding condition
                if dim.lp > 0 {
                    let lp = Reg::Const(Constant::U32(u32::try_from(dim.lp - 1).unwrap()));
                    let t = c.cmpgt(a, lp);
                    pc = c.and(t, pc);
                }
                if dim.rp > 0 {
                    let rp = Reg::Const(Constant::U32(
                        u32::try_from(isize::try_from(dim.d).unwrap() - dim.rp).unwrap(),
                    ));
                    let t = c.cmplt(a, rp);
                    pc = c.and(t, pc);
                }
            }
            old_offset = Some(offset);
        }
        let pcu32 = c.cast(pc, DType::U32);
        offset = c.mul(pcu32, offset);
        let z = Reg::Var(c.load(address, offset, dtype));
        let pcd = c.cast(pc, dtype);
        // Nullify z if padding condition is false (if there is padding at that index)
        c.mul(pcd, z)
    }

    /// Store from variable into address
    pub(crate) fn ir_for_indexed_store(&self, c: &mut IRCompiler, address: u16, var: Reg) {
        let mut offset = Reg::Const(Constant::U32(0));
        if let Some(inner) = self.0.last() {
            for (a, dim) in inner.iter().enumerate() {
                if dim.st != 0 && dim.d != 1 {
                    let stride = Reg::Const(Constant::U32(u32::try_from(dim.st).unwrap()));
                    offset = c.mad(Reg::Var(u16::try_from(a).unwrap()), stride, offset);
                }
            }
        }
        c.ops.push(IROp::Store {
            address,
            x: var,
            offset,
        });
    }
}

impl Display for View {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for inner in &self.0 {
            f.write_fmt(format_args!(
                "V(sh{:?} st{:?} pd{:?})",
                inner.iter().map(|d| d.d).collect::<Vec<usize>>(),
                inner.iter().map(|d| d.st).collect::<Vec<usize>>(),
                inner
                    .iter()
                    .map(|d| (d.lp, d.rp))
                    .collect::<Vec<(isize, isize)>>()
            ))?;
        }
        Ok(())
        /*if let Some(inner) = self.0.last() {
            f.write_fmt(format_args!(
                "V.{}(sh{:?} st{:?} pd{:?})",
                self.0.len(),
                inner.iter().map(|d| d.d).collect::<Vec<usize>>(),
                inner.iter().map(|d| d.st).collect::<Vec<usize>>(),
                inner
                    .iter()
                    .map(|d| (d.lp, d.rp))
                    .collect::<Vec<(isize, isize)>>()
            ))
        } else {
            f.write_str("none")
        }*/
    }
}

#[test]
fn view_split() {
    let mut view = View::contiguous(&[3, 1, 4, 2]);
    println!("{view}");
    view.reshape(2..3, &[2, 2, 1]);
    assert_eq!(view.shape(), [3, 1, 2, 2, 1, 2]);
    view.reshape(0..1, &[1, 3, 1]);
    assert_eq!(view.shape(), [1, 3, 1, 1, 2, 2, 1, 2]);
}

#[test]
fn view_binded() {
    let view = View::binded(&[4, 2, 3], &[5, 1, 2], 6);
    println!("{view}");
    assert_eq!(view.rank(), 6);
    assert_eq!(view.used_axes(), [1, 2, 5]);
    assert_eq!(view.shape(), [1, 2, 3, 1, 1, 4]);
}

#[test]
fn view_reshape() {
    let mut view = View::contiguous(&[3, 1, 4, 2]);
    view.reshape(1..3, &[2, 2]);
    assert_eq!(view.shape(), [3, 2, 2, 2]);
    assert_eq!(view.0.len(), 1);
    let mut view = View::contiguous(&[3, 3]);
    view.reshape(0..2, &[9]);
    assert_eq!(view.shape(), [9]);
    assert_eq!(view.0.len(), 1);
    let mut view = View::contiguous(&[3, 3]);
    view.reshape(0..1, &[1, 3]);
    view.reshape(2..3, &[1, 3]);
    assert_eq!(view.shape(), [1, 3, 1, 3]);
    assert_eq!(view.0.len(), 1);
}

#[test]
fn view_reshape2() {
    let mut view = View::binded(&[4, 2, 3], &[5, 1, 2], 6);
    view.reshape(0..1, &[1, 1, 1]);
    assert_eq!(view.shape(), [1, 1, 1, 2, 3, 1, 1, 4]);
    assert_eq!(view.0.len(), 1);
    let mut view = View::contiguous(&[3, 1, 5]);
    view.reshape(0..1, &[1, 3]);
    assert_eq!(view.shape(), [1, 3, 1, 5]);
    view.expand(2, 4);
    assert_eq!(view.shape(), [1, 3, 4, 5]);
    assert_eq!(view.0.len(), 1);
}

#[test]
fn view_pad2() {
    // Pad view twice in with opposite sings
    //todo!()
}
