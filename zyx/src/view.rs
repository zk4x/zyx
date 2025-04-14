//! View handles movement operations.

use super::ir::{IRCompiler, IROp, Reg};
use crate::{
    DType,
    dtype::Constant,
    shape::{Axis, Dim},
};
use std::{fmt::Display, ops::Range};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct View(Vec<Vec<RDim>>);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct RDim {
    d: Dim,  // dim
    st: Dim, // stride
    lp: isize,     // left pad
    rp: isize,     // right pad
}

fn to_contiguous_rdims(shape: &[Dim]) -> Vec<RDim> {
    let mut st = 1;
    let mut res = Vec::with_capacity(shape.len());
    unsafe { res.set_len(shape.len()); }
    for (i, &d) in shape.iter().enumerate().rev() {
        res[i] = RDim { st, d, lp: 0, rp: 0 };
        st *= d;
    }
    res
}

impl View {
    /// Create empty view for scalars
    pub(crate) const fn none() -> View {
        View(Vec::new())
    }

    pub(crate) fn contiguous(shape: &[Dim]) -> View {
        View(vec![to_contiguous_rdims(shape)])
    }

    /*pub(crate) fn binded(shape: &[usize], axes: &[usize], rank: usize) -> View {
        let mut stride = 1;
        debug_assert!(axes.iter().all(|&a| a < rank));
        let mut a: usize = rank;
        let mut res = Vec::with_capacity(a);
        while a > 0 {
            a -= 1;
            if let Some(i) = axes.iter().position(|axis| *axis == a) {
                let st = stride;
                let d = shape[i];
                stride *= d;
                res.push(RDim { d, st, lp: 0, rp: 0 });
            } else {
                res.push(RDim { d: 1, st: 0, lp: 0, rp: 0 });
            }
        }
        res.reverse();
        View(vec![res])
    }*/

    pub(crate) fn rank(&self) -> usize {
        self.0.last().map_or(1, Vec::len)
    }

    pub(crate) fn shape(&self) -> Vec<Dim> {
        self.0.last().map_or_else(|| vec![1], |inner| inner.iter().map(|dim| dim.d).collect())
    }

    pub(crate) fn original_numel(&self) -> usize {
        let mut res = 1;
        for dim in &self.0[0] {
            if dim.st != 0 {
                res *= usize::try_from(isize::try_from(dim.d).unwrap() - dim.lp - dim.rp).unwrap();
            }
        }
        res
    }

    /*pub(crate) fn numel(&self) -> usize {
        self.0.last().map_or(1, |inner| inner.iter().map(|dim| dim.d).product())
    }*/

    /*#[cfg(debug_assertions)]
    pub(crate) fn is_contiguous(&self) -> bool {
        self.0.last().map_or(true, |inner| {
            let stride = 1;
            inner.iter().all(|dim| dim.lp == 0 && dim.rp == 0 && dim.st == stride)
        })
    }*/

    /*pub(crate) fn used_axes(&self) -> Vec<usize> {
        self.0.last().map_or_else(Vec::new, |inner| {
            (0..inner.len()).filter(|&a| inner[a].st != 0).collect()
        })
    }*/

    /// Inserts new loop, shifts all axes greater than axis up by one
    pub(crate) fn insert_loop(&mut self, axis: usize) {
        //println!("Inserting loop at axis {axis}");
        if let Some(inner) = self.0.last_mut() {
            debug_assert!(axis < inner.len());
            let st = inner[axis].st;
            inner.insert(axis, RDim { d: 1, st, lp: 0, rp: 0 });
        }
        //println!("After insert loop {self:?}");
    }

    // This is used for reshape, merge and split
    pub(crate) fn reshape(&mut self, axes: Range<Axis>, shape: &[Dim]) {
        //println!("Reshape {self} axes {axes:?} into shape {shape:?}");
        debug_assert!(axes.end <= self.0.last().map_or(1, Vec::len));
        debug_assert_eq!(
            self.0.last().unwrap()[axes.clone()].iter().map(|dim| dim.d).product::<Dim>(),
            shape.iter().product::<Dim>()
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
            // If all reshaped axes are expanded
            let expanded_reshape = if axes.clone().all(|a| inner[a].st == 0) {
                contiguous = true;
                true
            } else {
                false
            };
            if axes.clone().any(|a| inner[a].lp != 0 || inner[a].rp != 0) {
                contiguous = false;
            }

            if contiguous {
                //println!("Reshape contiguous");
                for a in axes.clone().rev() {
                    let dim = inner.remove(a);
                    debug_assert_eq!(dim.lp, 0);
                    debug_assert_eq!(dim.rp, 0);
                }
                for &d in shape.iter().rev() {
                    let st = if expanded_reshape { 0 } else { ost };
                    ost *= d;
                    inner.insert(axes.start, RDim { d, st, lp: 0, rp: 0 });
                }
            } else {
                //println!("Reshape non-contiguous");
                let mut old_shape = self.shape();
                old_shape.splice(axes, shape.iter().copied());
                self.0.push(to_contiguous_rdims(&old_shape));
            }
        }
        //println!("After reshape: {self}\n");
    }

    pub(crate) fn permute(&mut self, axes: &[usize]) {
        // Move around strides, dim, rp and lp
        let inner = self.0.last_mut().unwrap();
        debug_assert_eq!(inner.len(), axes.len());
        let mut new = Vec::with_capacity(axes.len());
        for &a in axes {
            let dim = inner[a].clone();
            new.push(dim);
        }
        *inner = new;
    }

    pub(crate) fn expand(&mut self, axis: Axis, ndim: Dim) {
        //println!("View expand {self} axis = {axis} to ndim {ndim}");
        if let Some(inner) = self.0.last_mut() {
            if let Some(dim) = inner.get_mut(axis) {
                debug_assert!(dim.d == ndim || dim.d == 1);
                debug_assert_eq!(dim.lp, 0);
                debug_assert_eq!(dim.rp, 0);
                dim.d = ndim;
                dim.st = 0;
            } else {
                unreachable!("Expand on nonexistent axis.");
            }
        }
    }

    pub(crate) fn pad(&mut self, axis: Axis, left_pad: isize, right_pad: isize) {
        let mut old_shape = self.shape();
        if let Some(inner) = self.0.last_mut() {
            let mut dim = &mut inner[axis];
            dim.d = Dim::try_from(isize::try_from(dim.d).unwrap() + left_pad + right_pad)
                .unwrap();
            if dim.lp < 0 && left_pad > 0 {
                dim.d = Dim::try_from(isize::try_from(dim.d).unwrap() - left_pad).unwrap();
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
                                Dim::try_from(isize::try_from(d).unwrap() + left_pad).unwrap()
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
                dim.d = Dim::try_from(isize::try_from(dim.d).unwrap() - right_pad).unwrap();
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
                                Dim::try_from(isize::try_from(d).unwrap() + right_pad)
                                    .unwrap()
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
        old_shape[axis] =
            Dim::try_from(isize::try_from(old_shape[axis]).unwrap() + left_pad + right_pad)
                .unwrap();
        debug_assert_eq!(self.shape(), old_shape);
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
            let mut offset = Reg::Const(Constant::U64(0));
            for (a, dim) in inner.iter().enumerate().rev() {
                let a: Reg = old_offset.map_or_else(
                    || Reg::Var(u16::try_from(a).unwrap()),
                    |old_offset| {
                        let a = c.div(old_offset, Reg::Const(Constant::U64(ost)));
                        ost *= u64::try_from(dim.d).unwrap();
                        c.mod_(a, Reg::Const(Constant::U64(u64::try_from(dim.d).unwrap())))
                    },
                );
                //println!("ost: {ost}, {dim:?}");
                // Offset
                //if dim.st != 0 && dim.d != 1 {
                let t = if dim.lp != 0 {
                    let lp = Reg::Const(Constant::U64(u64::try_from(dim.lp.abs()).unwrap()));
                    if dim.lp > 0 {
                        c.sub(a, lp)
                    } else {
                        c.add(a, lp)
                    }
                } else {
                    a
                };
                let stride = Reg::Const(Constant::U64(u64::try_from(dim.st).unwrap()));
                offset = c.mad(t, stride, offset);
                //}
                // Padding condition
                if dim.lp > 0 {
                    let lp = Reg::Const(Constant::U64(u64::try_from(dim.lp - 1).unwrap()));
                    let t = c.cmpgt(a, lp);
                    pc = c.and(t, pc);
                }
                if dim.rp > 0 {
                    let rp = Reg::Const(Constant::U64(
                        u64::try_from(isize::try_from(dim.d).unwrap() - dim.rp).unwrap(),
                    ));
                    let t = c.cmplt(a, rp);
                    pc = c.and(t, pc);
                }
            }
            old_offset = Some(offset);
        }
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
        let mut offset = Reg::Const(Constant::U64(0));
        let mut old_offset: Option<Reg> = None;
        //println!("View");
        //for inner in self.0.iter() { println!("{inner:?}") }
        //println!();
        for inner in self.0.iter().rev() {
            //println!("\n{inner:?}");
            // a = offset / ost % dim
            let mut ost = 1;
            offset = Reg::Const(Constant::U64(0));
            for (a, dim) in inner.iter().enumerate().rev() {
                let a = old_offset.map_or_else(
                    || Reg::Var(u16::try_from(a).unwrap()),
                    |old_offset| {
                        let a = c.div(old_offset, Reg::Const(Constant::U64(ost)));
                        ost *= u64::try_from(dim.d).unwrap();
                        c.mod_(a, Reg::Const(Constant::U64(u64::try_from(dim.d).unwrap())))
                    },
                );
                //println!("ost: {ost}, a: {a:?}, {dim:?}");
                // Offset
                let t = if dim.lp != 0 {
                    let lp = Reg::Const(Constant::U64(u64::try_from(dim.lp.abs()).unwrap()));
                    if dim.lp > 0 {
                        c.sub(a, lp)
                    } else {
                        c.add(a, lp)
                    }
                } else {
                    a
                };
                let stride = Reg::Const(Constant::U64(u64::try_from(dim.st).unwrap()));
                offset = c.mad(t, stride, offset);
                // Padding condition
                if dim.lp > 0 {
                    let lp = Reg::Const(Constant::U64(u64::try_from(dim.lp - 1).unwrap()));
                    let t = c.cmpgt(a, lp);
                    pc = c.and(t, pc);
                }
                if dim.rp > 0 {
                    let rp = Reg::Const(Constant::U64(
                        u64::try_from(isize::try_from(dim.d).unwrap() - dim.rp).unwrap(),
                    ));
                    let t = c.cmplt(a, rp);
                    pc = c.and(t, pc);
                }
            }
            old_offset = Some(offset);
        }
        let pcu64 = c.cast(pc, DType::U64);
        offset = c.mul(pcu64, offset);
        let z = Reg::Var(c.load(address, offset, dtype));
        let pcd = c.cast(pc, dtype);
        // Nullify z if padding condition is false (if there is padding at that index)
        c.mul(pcd, z)
    }

    /// Store from variable into address
    pub(crate) fn ir_for_indexed_store(&self, c: &mut IRCompiler, address: u16, var: Reg) {
        let mut offset = Reg::Const(Constant::U64(0));
        if let Some(inner) = self.0.last() {
            for (a, dim) in inner.iter().enumerate() {
                if dim.st != 0 && dim.d != 1 {
                    let stride = Reg::Const(Constant::U64(u64::try_from(dim.st).unwrap()));
                    offset = c.mad(Reg::Var(u16::try_from(a).unwrap()), stride, offset);
                }
            }
        }
        c.ops.push(IROp::Store { address, x: var, offset });
    }
}

impl Display for View {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for inner in &self.0 {
            f.write_fmt(format_args!(
                "V(sh{:?} st{:?} pd{:?})",
                inner.iter().map(|d| d.d).collect::<Vec<Dim>>(),
                inner.iter().map(|d| d.st).collect::<Vec<Dim>>(),
                inner.iter().map(|d| (d.lp, d.rp)).collect::<Vec<(isize, isize)>>()
            ))?;
        }
        Ok(())
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

/*#[test]
fn view_binded() {
    let view = View::binded(&[4, 2, 3], &[5, 1, 2], 6);
    println!("{view}");
    assert_eq!(view.rank(), 6);
    assert_eq!(view.used_axes(), [1, 2, 5]);
    assert_eq!(view.shape(), [1, 2, 3, 1, 1, 4]);
}*/

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
    /*let mut view = View::binded(&[4, 2, 3], &[5, 1, 2], 6);
    view.reshape(0..1, &[1, 1, 1]);
    assert_eq!(view.shape(), [1, 1, 1, 2, 3, 1, 1, 4]);
    assert_eq!(view.0.len(), 1);*/
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
    let mut view = View::contiguous(&[1, 1, 2, 6]);
    view.pad(3, -3, 0);
    view.pad(3, 2, 0);
    assert_eq!(
        view,
        View(vec![
            vec![
                RDim { d: 1, st: 12, lp: 0, rp: 0 },
                RDim { d: 1, st: 12, lp: 0, rp: 0 },
                RDim { d: 2, st: 6, lp: 0, rp: 0 },
                RDim { d: 3, st: 1, lp: -3, rp: 0 }
            ],
            vec![
                RDim { d: 1, st: 6, lp: 0, rp: 0 },
                RDim { d: 1, st: 6, lp: 0, rp: 0 },
                RDim { d: 2, st: 3, lp: 0, rp: 0 },
                RDim { d: 5, st: 1, lp: 2, rp: 0 }
            ]
        ])
    );
}
