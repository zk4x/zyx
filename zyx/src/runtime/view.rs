//! View handles movement operations on nodes.
//! It is midlayer between graph and IR representation of movement ops.

use std::{collections::BTreeMap, fmt::Display};

use crate::shape::{Axis, Dimension};

pub(super) type Stride = usize;

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct StridedDim {
    pub(super) axis: Axis,
    pub(super) dim: Dimension,
    pub(super) stride: Stride,
}

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct ReshapedDim {
    pub(super) axis: Axis,
    pub(super) dim: Dimension,
    pub(super) stride: Stride,
    pub(super) lp: isize,
    pub(super) rp: isize,
}

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, PartialEq, Eq, Clone, PartialOrd, Ord)]
pub(super) enum View {
    None,
    //Contiguous(Vec<StridedDim>), // TODO perhaps later, mainly for cpu and perhaps wide loads on gpu
    Strided(Vec<StridedDim>),
    // First is typical strided, second is group of axes and their padding. Very ugly, but works.
    // If you can make it nicer, please do.
    Padded(Vec<StridedDim>, Vec<(Vec<Axis>, (isize, isize))>),
    //Reshaped(), // TODO perhaps for some weird optimizations, but it may actually reduce performace
    // since then loads are very unpredictable
}

impl View {
    pub(super) fn new(shape: &[usize]) -> Self {
        let mut stride = 1;
        let mut view: Vec<StridedDim> = shape
            .iter()
            .enumerate()
            .rev()
            .map(|(axis, dim)| {
                let temp = stride;
                stride *= dim;
                StridedDim {
                    axis,
                    stride: temp,
                    dim: *dim,
                }
            })
            .collect();
        view.reverse();
        return View::Strided(view);
    }

    /// Creates view binded to specific axes
    pub(super) fn binded(shape: &[usize], axes: &[usize]) -> Self {
        assert_eq!(shape.len(), axes.len());
        let mut stride = 1;
        let mut view: Vec<StridedDim> = shape
            .iter()
            .zip(axes)
            .rev()
            .map(|(&dim, &axis)| {
                let temp = stride;
                stride *= dim;
                StridedDim {
                    axis,
                    stride: temp,
                    dim,
                }
            })
            .collect();
        view.reverse();
        return View::Strided(view);
    }

    pub(super) fn shape(&self) -> Vec<usize> {
        match self {
            View::None => vec![1],
            View::Strided(dims) => dims.iter().map(|dim| dim.dim).collect(),
            View::Padded(dims, _) => dims.iter().map(|dim| dim.dim).collect(),
        }
    }

    pub(super) fn rank(&self) -> usize {
        match self {
            View::None => 1,
            View::Strided(dims) => dims.len(),
            View::Padded(dims, _) => dims.len(),
        }
    }

    /// Returns sorted used axes
    pub(super) fn used_axes(&self) -> Vec<Axis> {
        match self {
            View::None => Vec::new(),
            View::Strided(dims) | View::Padded(dims, _) => {
                let mut res: Vec<Axis> = dims
                    .iter()
                    .flat_map(|x| if x.stride != 0 { Some(x.axis) } else { None })
                    .collect();
                res.sort();
                res
            }
        }
    }

    pub(super) fn requires_conditional_padding(&self) -> bool {
        // View requires conditional padding if any padding is more than zero
        match self {
            View::None | View::Strided(_) => false,
            View::Padded(_, padding) => {
                return padding.iter().any(|(_, (lp, rp))| *lp > 0 || *rp > 0);
            }
        }
    }

    pub(super) fn original_numel(&self) -> usize {
        //println!("Original numel {self}");
        match self {
            View::None => 1,
            View::Strided(dims) => dims
                .iter()
                .map(|dim| if dim.stride != 0 { dim.dim } else { 1 })
                .product(),
            View::Padded(dims, axes) => axes
                .iter()
                .map(|(axes, (lp, rp))| {
                    let numel: usize = dims
                        .iter()
                        .filter_map(|StridedDim { axis, dim, .. }| {
                            if axes.contains(axis) {
                                Some(*dim)
                            } else {
                                None
                            }
                        })
                        .product();
                    //println!("{numel}, {lp}, {rp}");
                    (numel as isize - lp - rp) as usize
                })
                .product(),
        }
    }

    pub(super) fn permute(&mut self, axes: &[usize]) {
        //println!("Permuting {self} by {axes:?}");
        assert_eq!(self.rank(), axes.len());
        match self {
            View::None => {}
            View::Strided(dims) => {
                *dims = axes.iter().map(|axis| dims[*axis]).collect();
                for (a, dim) in dims.iter_mut().enumerate() {
                    dim.axis = a;
                }
            }
            View::Padded(dims, padding) => {
                *dims = axes.iter().map(|axis| dims[*axis]).collect();
                for (a, dim) in dims.iter_mut().enumerate() {
                    dim.axis = a;
                }
                // TODO is this correct?
                let axes_map: BTreeMap<usize, usize> =
                    (0..axes.len()).zip(axes.iter().copied()).collect();
                for (axes, _) in padding {
                    for d in axes {
                        *d = axes_map[d];
                    }
                }
            }
        }
    }

    //pub(super) fn arbitrary_permute(&mut self, axes: &[usize]) { todo!() }

    pub(super) fn pad_axis(&mut self, axis: Axis, left_pad: isize, right_pad: isize) {
        //println!("Padding {axis} with {left_pad}, {right_pad}");
        let paxis = axis;
        match self {
            View::None => {}
            View::Strided(dims) => {
                if dims.iter().any(|&StridedDim { axis, .. }| axis == paxis) {
                    *self = View::Padded(
                        dims.iter()
                            .map(|&StridedDim { axis, dim, stride }| {
                                if axis == paxis {
                                    StridedDim {
                                        axis,
                                        dim: (dim as isize + left_pad + right_pad) as usize,
                                        stride,
                                    }
                                } else {
                                    StridedDim { axis, dim, stride }
                                }
                            })
                            .collect(),
                        vec![(vec![axis], (left_pad, right_pad))],
                    );
                }
            }
            View::Padded(dims, padding) => {
                if let Some(StridedDim { dim, .. }) = dims
                    .iter_mut()
                    .find(|StridedDim { axis, .. }| *axis == paxis)
                {
                    //println!("Padding axis {axis}, dim {dim} with {left_pad}, {right_pad}");
                    *dim = (*dim as isize + left_pad + right_pad) as usize;
                    if let Some((_, (lp, rp))) =
                        padding.iter_mut().find(|(axes, _)| axes.contains(&axis))
                    {
                        *lp += left_pad;
                        *rp += right_pad;
                    } else {
                        padding.push((vec![axis], (left_pad, right_pad)));
                        padding.sort();
                    }
                }
            }
        }
        //println!("Result {self}");
    }

    pub(super) fn expand(&mut self, axis: Axis, dimension: Dimension) {
        // TODO probably instead of changing stride to 0, we can simply
        // remove the dimension alltogether
        /*let _ = dimension;
        match self {
            View::None => {}
            View::Strided(dims) => {
                dims.retain(|x| x.axis != axis);
            }
            View::Padded(dims, pa) => {
                dims.retain(|x| x.axis != axis);
                pa.axes.iter_mut().for_each(|(v, _)| v.retain(|a| *a != axis));
                pa.axes.retain(|(axes, _)| !axes.is_empty());
            }
        }*/
        match self {
            View::None => {}
            View::Strided(dims) => {
                for StridedDim {
                    axis: paxis,
                    dim,
                    stride,
                    ..
                } in dims.iter_mut()
                {
                    if axis == *paxis {
                        assert_eq!(*dim, 1);
                        *stride = 0;
                        *dim = dimension;
                    }
                }
            }
            View::Padded(dims, padding) => {
                for StridedDim {
                    axis: paxis,
                    dim,
                    stride,
                    ..
                } in dims.iter_mut()
                {
                    if axis == *paxis {
                        assert_eq!(*dim, 1);
                        *stride = 0;
                        *dim = dimension;
                        // Remove expanded axes from padding
                        for (axes, _) in padding.iter_mut() {
                            if let Some(id) = axes.iter().position(|&a| a == axis) {
                                axes.remove(id);
                            }
                        }
                    }
                }
            }
        }
    }

    pub(super) fn numel(&self) -> usize {
        match self {
            View::None => 0,
            View::Strided(dims) => dims.iter().map(|dim| dim.dim).product(),
            View::Padded(dims, _) => dims.iter().map(|dim| dim.dim).product(),
        }
    }

    pub(super) fn is_contiguous(&self) -> bool {
        &View::new(&self.shape()) == self
    }

    pub(super) fn split_axis(&mut self, axis: Axis, dimensions: &[usize]) {
        //println!("{axis}, {dimensions:?}");
        match self {
            View::None => {}
            View::Strided(dims) => {
                // Rename all following axes
                for st_dim in dims.iter_mut() {
                    if axis < st_dim.axis {
                        st_dim.axis += dimensions.len() - 1;
                    }
                }
                if let Some((id, st_dim)) = dims
                    .iter_mut()
                    .enumerate()
                    .find(|(_, dim)| dim.axis == axis)
                {
                    let mut stride = st_dim.stride;
                    dims.remove(id);
                    let mut temp_axis = axis + dimensions.len();
                    for dim in dimensions.iter().copied().rev() {
                        temp_axis -= 1;
                        dims.insert(
                            id,
                            StridedDim {
                                axis: temp_axis,
                                dim,
                                stride,
                            },
                        );
                        stride *= dim;
                    }
                }
            }
            View::Padded(dims, padding) => {
                let dim_len = dimensions.len();
                for st_dim in dims.iter_mut() {
                    if axis < st_dim.axis {
                        st_dim.axis += dim_len - 1;
                    }
                }
                if let Some((id, st_dim)) = dims
                    .iter_mut()
                    .enumerate()
                    .find(|(_, dim)| dim.axis == axis)
                {
                    let mut stride = st_dim.stride;
                    dims.remove(id);
                    let mut temp_axis = axis + dimensions.len();
                    for dim in dimensions.iter().copied().rev() {
                        temp_axis -= 1;
                        dims.insert(
                            id,
                            StridedDim {
                                axis: temp_axis,
                                dim,
                                stride,
                            },
                        );
                        stride *= dim;
                    }
                }
                // If key in padding axes is greater than axis, then add dim_len - 1 to it
                for (axes, _) in padding.iter_mut() {
                    for a in axes {
                        if *a > axis {
                            *a += dim_len - 1;
                        }
                    }
                }
                // Split padding
                if let Some((axes, _)) = padding.iter_mut().find(|(k, _)| k.contains(&axis)) {
                    //std::println!("Original: {axes:?} splitting into: {axis}..{}", axis+dim_len);
                    for a in axis + 1..axis + dim_len {
                        if dims.iter().find(|dim| dim.axis == a).unwrap().dim != 1 {
                            axes.push(a);
                        }
                    }
                    // Would not be needed on btreeset
                    axes.sort();
                }
            }
        }
    }
}

impl Display for View {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            View::None => f.write_str("View::None"),
            View::Strided(dims) => f.write_fmt(format_args!(
                "V:S ax{:?} sh{:?} st{:?}",
                dims.iter().map(|d| d.axis).collect::<Vec<Dimension>>(),
                dims.iter().map(|d| d.dim).collect::<Vec<Dimension>>(),
                dims.iter().map(|d| d.stride).collect::<Vec<Stride>>()
            )),
            View::Padded(dims, padding) => f.write_fmt(format_args!(
                "V:P ax{:?} sh{:?} st{:?} pd{:?}",
                dims.iter().map(|d| d.axis).collect::<Vec<Dimension>>(),
                dims.iter().map(|d| d.dim).collect::<Vec<Dimension>>(),
                dims.iter().map(|d| d.stride).collect::<Vec<Stride>>(),
                padding,
            )),
        }
    }
}
