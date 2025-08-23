//! View handles movement operations.

use crate::shape::{Axis, Dim};
use std::{fmt::Display, ops::Range};

/// .0[0] is original shape, further shapes are additional reshapes
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct View(pub Vec<Vec<RDim>>); // TODO switch to Box<[]> instead of Vec?

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RDim {
    pub d: Dim,    // dim
    pub st: Dim,   // stride
    pub lp: isize, // left pad
    pub rp: isize, // right pad
}

fn to_contiguous_rdims(shape: &[Dim]) -> Vec<RDim> {
    let mut st = 1;
    let mut res = vec![RDim { st: 0, d: 0, lp: 0, rp: 0 }; shape.len()];
    for (i, &d) in shape.iter().enumerate().rev() {
        res[i] = RDim { st, d, lp: 0, rp: 0 };
        st *= d;
    }
    res
}

impl View {
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
        self.0.last().map_or(0, Vec::len)
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

    pub(crate) fn numel(&self) -> usize {
        self.0.last().map_or(1, |inner| inner.iter().map(|dim| dim.d).product())
    }

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

    // Inserts new loop, shifts all axes greater than axis up by one
    /*pub(crate) fn insert_loop(&mut self, axis: usize) {
        //println!("Inserting loop at axis {axis}");
        let inner = self.0.last_mut().unwrap();
        debug_assert!(axis < inner.len());
        let st = inner[axis].st;
        inner.insert(axis, RDim { d: 1, st, lp: 0, rp: 0 });
        //println!("After insert loop {self:?}");
    }*/

    // This is used for reshape, merge and split
    pub(crate) fn reshape(&mut self, axes: Range<Axis>, shape: &[Dim]) {
        //println!("Reshape {self} axes {axes:?} into shape {shape:?}");
        debug_assert!(
            axes.end <= self.0.last().map_or(1, Vec::len) as Dim,
            "Reshape axes range {axes:?} is greater than view's rank {}",
            self.0.last().map_or(1, Vec::len)
        );
        debug_assert_eq!(
            self.0.last().unwrap()[axes.start as usize..axes.end as usize].iter().map(|dim| dim.d).product::<Dim>(),
            shape.iter().product::<Dim>(),
            "Reshape failed, products are different: {:?} -> {:?}", self.shape(), shape
        );
        let inner = self.0.last_mut().unwrap();
        let mut contiguous = true;
        let mut a = inner.len() as Dim;
        let mut stride = 1;
        let mut ost = 1;
        while a > axes.start {
            a -= 1;
            let dim = &inner[a as usize];
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
        if axes.clone().any(|a| inner[a as usize].st == 0) {
            contiguous = false;
        }
        // If all reshaped axes are expanded
        let expanded_reshape = if axes.clone().all(|a| inner[a as usize].st == 0) {
            contiguous = true;
            true
        } else {
            false
        };
        if axes.clone().any(|a| inner[a as usize].lp != 0 || inner[a as usize].rp != 0) {
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
        //println!("After reshape: {self}\n");
    }

    // If axes are shorter than inner, we just permute the first dimensions
    pub(crate) fn permute(&mut self, axes: &[usize]) {
        // Move around strides, dim, rp and lp
        // Version without allocation
        let inner = self.0.last_mut().unwrap();
        debug_assert!(inner.len() >= axes.len(), "Failed to permute {:?} by axes={axes:?}", self.shape());
        debug_assert_eq!(*axes.iter().max().unwrap(), axes.len() - 1, "Failed to permute {:?} by axes={axes:?}", self.shape());

        let mut temp = inner[axes[0]].clone();
        let mut a = 0;
        std::mem::swap(&mut inner[a], &mut temp);

        for _ in 1..axes.len() {
            let mut i = 0;
            while a != axes[i] {
                i += 1;
            }
            std::mem::swap(&mut inner[i], &mut temp);
            a = i;
        }
    }

    pub(crate) fn expand(&mut self, shape: &[Dim]) {
        // Expands first shape.len() dims
        debug_assert!(self.rank() >= shape.len());
        let inner = self.0.last_mut().unwrap();
        for (dim, &d) in inner.iter_mut().zip(shape) {
            if d != dim.d {
                debug_assert_eq!(dim.d, 1);
                debug_assert_eq!(dim.lp, 0);
                debug_assert_eq!(dim.rp, 0);
                dim.d = d;
                dim.st = 0;
            }
        }
    }

    /*pub fn expand_axis(&mut self, axis: Axis, ndim: Dim) {
        //println!("View expand {self} axis = {axis} to ndim {ndim}");
        let inner = self.0.last_mut().unwrap();
        let dim = &mut inner[axis];
        debug_assert!(dim.d == ndim || dim.d == 1);
        debug_assert_eq!(dim.lp, 0);
        debug_assert_eq!(dim.rp, 0);
        dim.d = ndim;
        dim.st = 0;
    }*/

    pub fn pad(&mut self, padding: &[(isize, isize)]) {
        //println!("view: {:?} padding: {padding:?}", self.shape());
        for (axis, &(lp, rp)) in (0..self.rank()).rev().zip(padding) {
            if lp != 0 || rp != 0 {
                self.pad_axis(axis, lp, rp);
            }
        }
    }

    pub fn pad_axis(&mut self, axis: Axis, left_pad: isize, right_pad: isize) {
        let mut old_shape = self.shape();
        let inner = self.0.last_mut().unwrap();
        let mut dim = &mut inner[axis];
        dim.d = Dim::try_from(isize::try_from(dim.d).unwrap() + left_pad + right_pad).unwrap();
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
                            Dim::try_from(isize::try_from(d).unwrap() + right_pad).unwrap()
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
        old_shape[axis] = Dim::try_from(isize::try_from(old_shape[axis]).unwrap() + left_pad + right_pad).unwrap();
        debug_assert_eq!(self.shape(), old_shape);
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
fn view_permute_1() {
    let mut view = View::contiguous(&[3, 1, 4, 2]);
    view.permute(&[3, 1, 0, 2]);
    assert_eq!(view.shape(), [2, 1, 3, 4]);
    view.permute(&[2, 0, 3, 1]);
    assert_eq!(view.shape(), [3, 2, 4, 1]);
}

#[test]
fn view_permute_2() {
    let mut view = View::contiguous(&[3, 1, 4, 2]);
    view.permute(&[1, 2, 0]);
    assert_eq!(view.shape(), [1, 4, 3, 2]);
    view.permute(&[2, 1, 0]);
    assert_eq!(view.shape(), [3, 4, 1, 2]);
}

// Permute test, no alloc is ~25% faster
/*#[test]
fn view_permute2() {
    let mut view = View::contiguous(&[3, 1, 4, 2]);
    let begin = std::time::Instant::now();
    for _ in 0..10000000 {
        view.permute(&[3, 1, 0, 2]);
    }
    let micros = begin.elapsed().as_micros();
    println!("Took {micros}us");
}*/

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

/*#[test]
fn view_reshape2() {
    /*let mut view = View::binded(&[4, 2, 3], &[5, 1, 2], 6);
    view.reshape(0..1, &[1, 1, 1]);
    assert_eq!(view.shape(), [1, 1, 1, 2, 3, 1, 1, 4]);
    assert_eq!(view.0.len(), 1);*/
    let mut view = View::contiguous(&[3, 1, 5]);
    view.reshape(0..1, &[1, 3]);
    assert_eq!(view.shape(), [1, 3, 1, 5]);
    view.expand_axis(2, 4);
    assert_eq!(view.shape(), [1, 3, 4, 5]);
    assert_eq!(view.0.len(), 1);
}*/

#[test]
fn view_pad2() {
    // Pad view twice in with opposite sings
    //todo!()
    let mut view = View::contiguous(&[1, 1, 2, 6]);
    view.pad_axis(3, -3, 0);
    view.pad_axis(3, 2, 0);
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
