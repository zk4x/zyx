//! View handles movement operations.

use nanoserde::{DeBin, SerBin};

use crate::shape::{Dim, UAxis};
use std::{cmp::Ordering, fmt::Display, ops::Range};

/// .0[0] is original shape, further shapes are additional reshapes
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub struct View(pub Vec<Vec<RDim>>);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RDim {
    pub d: Dim,  // dim
    pub st: Dim, // stride
    pub lp: i32, // left pad
    pub rp: i32, // right pad
}

impl SerBin for RDim {
    fn ser_bin(&self, output: &mut Vec<u8>) {
        self.d.ser_bin(output);
        self.st.ser_bin(output);
        output.extend(self.lp.to_le_bytes());
        output.extend(self.rp.to_le_bytes());
    }
}

impl DeBin for RDim {
    fn de_bin(offset: &mut usize, bytes: &[u8]) -> Result<Self, nanoserde::DeBinErr> {
        let d = Dim::de_bin(offset, bytes)?;
        let st = Dim::de_bin(offset, bytes)?;
        let i32_bytes = std::mem::size_of::<i32>();

        // Read lp: isize (convert from bytes)
        if *offset + i32_bytes > bytes.len() {
            return Err(nanoserde::DeBinErr::new(*offset, i32_bytes, bytes.len()));
        }
        let lp_bytes = &bytes[*offset..*offset + i32_bytes];
        let lp = i32::from_le_bytes(lp_bytes.try_into().unwrap());
        *offset += i32_bytes;

        // Read rp: isize
        if *offset + i32_bytes > bytes.len() {
            return Err(nanoserde::DeBinErr::new(*offset, i32_bytes, bytes.len()));
        }
        let rp_bytes = &bytes[*offset..*offset + i32_bytes];
        let rp = i32::from_le_bytes(rp_bytes.try_into().unwrap());
        *offset += i32_bytes;
        Ok(Self { d, st, lp, rp })
    }
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

    #[cfg(test)]
    fn strides(&self) -> Vec<Dim> {
        self.0.last().map_or_else(|| vec![1], |inner| inner.iter().map(|dim| dim.st).collect())
    }

    pub(crate) fn original_numel(&self) -> usize {
        let mut res = 1;
        for dim in &self.0[0] {
            if dim.st != 0 {
                res *= usize::try_from(i32::try_from(dim.d).unwrap() - dim.lp - dim.rp).unwrap();
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

    // Inserts new loop, shifts all axes greater than axis up by one
    /*pub(crate) fn insert_loop(&mut self, axis: usize) {
        //println!("Inserting loop at axis {axis}");
        let inner = self.0.last_mut().unwrap();
        debug_assert!(axis < inner.len());
        let st = inner[axis].st;
        inner.insert(axis, RDim { d: 1, st, lp: 0, rp: 0 });
        //println!("After insert loop {self:?}");
    }*/

    pub fn is_reshape_contiguous(&self, axes: Range<UAxis>, new_shape: &[Dim]) -> bool {
        //println!("{:?} reshape to {:?}", self, new_shape);
        if let Some(last_block) = self.0.last() {
            // Try to reshape last block in place
            let new_block = try_reshape(&last_block[axes.clone()], new_shape);
            if new_block.is_empty() {
                return false;
            }
        }
        true
    }

    pub fn reshape(&mut self, mut axes: Range<UAxis>, new_shape: &[Dim]) {
        /*println!(
            "Reshape {:?}, axes {:?} into shape {new_shape:?}, {self}",
            self.shape(),
            axes.clone()
        );*/
        debug_assert!(
            axes.end <= self.0.last().map_or(1, Vec::len) as Dim,
            "Reshape axes range {axes:?} is greater than view's rank {}",
            self.0.last().map_or(1, Vec::len)
        );
        debug_assert_eq!(
            self.0.last().unwrap()[axes.start as usize..axes.end as usize].iter().map(|dim| dim.d).product::<Dim>(),
            new_shape.iter().product::<Dim>(),
            "Reshape failed, products are different: {:?} axes {axes:?} -> {:?}",
            self.shape(),
            new_shape
        );

        let mut new_shape: Vec<Dim> = new_shape.into();
        // Means we are inserting new dims in front
        if axes.end == 0 {
            axes.end = 1;
            new_shape.push(self.0.last().unwrap()[0].d);
        }

        if let Some(last_block) = self.0.last_mut() {
            // Try to reshape last block in place
            let new_block = try_reshape(&last_block[axes.clone()], &new_shape);
            if !new_block.is_empty() {
                // Reshape succeeded in place, done
                _ = last_block.splice(axes, new_block);
                return;
            }
        }

        //println!("Reshape non-contiguous");
        // Reshape failed, so append a new block with contiguous strides and zero padding
        let mut shape = self.shape();
        shape.splice(axes, new_shape.iter().copied());
        self.0.push(to_contiguous_rdims(&shape));
    }

    // This is used for reshape, merge and split
    /*pub(crate) fn reshape_direct(&mut self, axes: Range<Axis>, shape: &[Dim]) {
        //println!("Reshape {self} axes {axes:?} into shape {shape:?}");
        debug_assert!(
            axes.end <= self.0.last().map_or(1, Vec::len) as Dim,
            "Reshape axes range {axes:?} is greater than view's rank {}",
            self.0.last().map_or(1, Vec::len)
        );
        debug_assert_eq!(
            self.0.last().unwrap()[axes.start as usize..axes.end as usize].iter().map(|dim| dim.d).product::<Dim>(),
            shape.iter().product::<Dim>(),
            "Reshape failed, products are different: {:?} -> {:?}",
            self.shape(),
            shape
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
        //println!("contiguous={contiguous}");
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
    }*/

    /// Permute by reversing the order of axes
    pub(crate) fn reverse(&mut self) {
        self.0.last_mut().unwrap().reverse();
    }

    /// If axes are shorter than inner, we just permute the first dimensions
    pub(crate) fn permute(&mut self, axes: &[usize]) {
        // Move around strides, dim, rp and lp
        let inner = self.0.last_mut().unwrap();
        debug_assert!(
            inner.len() >= axes.len(),
            "Failed to permute {:?} by axes={axes:?}",
            self.shape()
        );
        debug_assert_eq!(
            *axes.iter().max().unwrap(),
            axes.len() - 1,
            "Failed to permute {:?} by axes={axes:?}",
            self.shape()
        );

        let mut temp_data = inner.clone();
        for i in 0..axes.len() {
            temp_data[i] = inner[axes[i]].clone();
        }
        *inner = temp_data;
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

    pub fn pad(&mut self, padding: &[(i32, i32)]) {
        //println!("view: {:?} padding: {padding:?}", self.shape());
        let rank = self.rank();
        for (axis, &(lp, rp)) in (rank - padding.len()..rank).rev().zip(padding) {
            if lp != 0 || rp != 0 {
                self.pad_axis(axis, lp, rp);
            }
        }
        //println!("Shape after padding: {:?}", self.shape());
    }

    pub fn pad_axis(&mut self, axis: UAxis, left_pad: i32, right_pad: i32) {
        let mut old_shape = self.shape();
        let inner = self.0.last_mut().unwrap();
        let mut dim = &mut inner[axis];
        dim.d = Dim::try_from(i32::try_from(dim.d).unwrap() + left_pad + right_pad).unwrap();
        if dim.lp < 0 && left_pad > 0 {
            dim.d = Dim::try_from(i32::try_from(dim.d).unwrap() - left_pad).unwrap();
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
                            Dim::try_from(i32::try_from(d).unwrap() + left_pad).unwrap()
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
            dim.d = Dim::try_from(i32::try_from(dim.d).unwrap() - right_pad).unwrap();
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
                            Dim::try_from(i32::try_from(d).unwrap() + right_pad).unwrap()
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
        old_shape[axis] = Dim::try_from(i32::try_from(old_shape[axis]).unwrap() + left_pad + right_pad).unwrap();
        debug_assert_eq!(self.shape(), old_shape);
    }
}

fn try_reshape(block: &[RDim], new_shape: &[usize]) -> Vec<RDim> {
    fn is_contiguous_block(dims: &[RDim]) -> bool {
        //println!("is contiguous: {dims:?}");
        let mut expected_stride = dims.last().map_or(1, |rd| rd.st);
        for rd in dims.iter().rev() {
            if rd.lp != 0 || rd.rp != 0 {
                return false;
            }
            if rd.d == 1 {
                continue; // Stride doesn't matter if dim == 1
            }
            if rd.st != expected_stride {
                //println!("non-contiguous");
                return false;
            }
            expected_stride *= rd.d;
        }
        //println!("contiguous");
        true
    }

    /*let old_total: usize = block.iter().map(|rd| rd.d).product();
    let new_total: usize = new_shape.iter().product();

    if old_total != new_total {
        return Vec::new();
    }*/

    if block.iter().map(|rd| rd.d).eq(new_shape.iter().copied()) {
        return block.into(); // Same shape, nothing to do
    }

    let mut new_dims = vec![RDim { d: 0, st: 0, lp: 0, rp: 0 }; new_shape.len()];
    let (mut orig_start, mut new_start) = (0, 0);
    let old_len = block.len();
    let new_len = new_shape.len();

    while orig_start < old_len && new_start < new_len {
        let (mut orig_prod, mut new_prod) = (block[orig_start].d, new_shape[new_start]);
        let (mut i, mut j) = (orig_start + 1, new_start + 1);

        // Expand until products match
        loop {
            match orig_prod.cmp(&new_prod) {
                Ordering::Less => {
                    orig_prod *= block[i].d;
                    i += 1;
                    debug_assert!(i <= old_len);
                }
                Ordering::Greater => {
                    new_prod *= new_shape[j];
                    j += 1;
                    debug_assert!(j <= new_len);
                }
                Ordering::Equal => {
                    if i < old_len {
                        if block[i].d == 1 {
                            i += 1;
                            continue;
                        }
                    }
                    if j < new_len {
                        if new_shape[j] == 1 {
                            j += 1;
                            continue;
                        }
                    }
                    break;
                }
            }
        }

        let orig_slice = &block[orig_start..i];
        let new_slice_shape = &new_shape[new_start..j];

        if orig_slice.iter().map(|rd| rd.d).eq(new_slice_shape.iter().copied()) {
            // Shape unchanged: copy original RDims as-is, skip contiguous check
            for (k, rd) in (new_start..j).zip(orig_slice.iter()) {
                new_dims[k] = rd.clone();
            }
        } else {
            // Shape changed: check contiguity and no padding allowed
            if !is_contiguous_block(orig_slice) {
                return Vec::new();
            }

            // Recompute strides, zero padding
            let mut stride = orig_slice.last().map_or(1, |rd| rd.st);
            for k in (new_start..j).rev() {
                let dim = new_shape[k];
                new_dims[k] = RDim { d: dim, st: if dim == 1 { 0 } else { stride }, lp: 0, rp: 0 };
                stride *= dim;
            }
        }

        orig_start = i;
        new_start = j;
    }

    if orig_start != old_len || new_start != new_len {
        //println!("{new_dims:?}");
        //println!("{orig_start}, {new_start}");
        //return Vec::new();
        unreachable!();
    }

    new_dims
}

impl Display for View {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for inner in &self.0 {
            f.write_fmt(format_args!(
                "V(sh{:?} st{:?} pd{:?})",
                inner.iter().map(|d| d.d).collect::<Vec<Dim>>(),
                inner.iter().map(|d| d.st).collect::<Vec<Dim>>(),
                inner.iter().map(|d| (d.lp, d.rp)).collect::<Vec<(i32, i32)>>()
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
    //println!("{view}");
    view.reshape(2..3, &[2, 2, 1]);
    assert_eq!(view.shape(), [3, 1, 2, 2, 1, 2]);
    view.reshape(0..1, &[1, 3, 1]);
    assert_eq!(view.shape(), [1, 3, 1, 1, 2, 2, 1, 2]);
    assert_eq!(view.0.len(), 1);
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
fn view_reshape_1() {
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
fn view_reshape_2() {
    let mut view = View::contiguous(&[512, 368]);
    view.permute(&[1, 0]);
    view.reshape(0..2, &[1, 368, 512]);
    println!("{view}");
    debug_assert_eq!(view.0.len(), 1);
    debug_assert_eq!(view.shape(), [1, 368, 512]);
    debug_assert_eq!(view.strides(), [0, 1, 368]);
}

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

#[test]
fn view_serialization() {
    let view = View::contiguous(&[3, 4, 1, 7, 1]);
    let x = view.serialize_bin();
    let view2: View = nanoserde::DeBin::deserialize_bin(&x).ok().unwrap();
    assert_eq!(view, view2);
}
