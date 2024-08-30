use std::{collections::BTreeMap, fmt::Display};

use crate::shape::{Axis, Dimension};

pub(crate) type Stride = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct StridedDim {
    pub(crate) axis: Axis,
    pub(crate) dim: Dimension,
    pub(crate) stride: Stride,
}

#[derive(Debug, PartialEq, Eq, Clone, PartialOrd, Ord)]
pub(crate) enum View {
    None,
    //Contiguous(Vec<StridedDim>), // TODO perhaps later, mainly for cpu and perhaps wide loads on gpu
    Strided(Vec<StridedDim>),
    // First is typical strided, second is group of axes and their padding. Very ugly, but works.
    // If you can make it nicer, please do.
    Padded(Vec<StridedDim>, PaddedAxes),
    //Reshaped(), // TODO perhaps for some weird optimizations, but it may actually reduce performace
    // since then loads are very unpredictable
}

#[derive(Debug, PartialEq, Eq, Clone, PartialOrd, Ord)]
pub(crate) struct PaddedAxes {
    pub(crate) axes: Vec<(Vec<Axis>, (isize, isize))>,
}

impl PaddedAxes {
    fn new(axis: Axis, left_pad: isize, right_pad: isize) -> Self {
        Self {
            axes: vec![(vec![axis], (left_pad, right_pad))],
        }
    }

    fn pad(&mut self, axis: Axis, left_pad: isize, right_pad: isize) {
        if let Some((_, (lp, rp))) = self.axes.iter_mut().find(|(k, _)| k.contains(&axis)) {
            *lp += left_pad;
            *rp += right_pad;
        } else {
            self.axes.push((vec![axis], (left_pad, right_pad)));
        }
    }
}

impl View {
    pub(crate) fn new(shape: &[usize]) -> Self {
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

    pub(crate) fn shape(&self) -> Vec<usize> {
        match self {
            View::None => vec![1],
            View::Strided(dims) => dims.iter().map(|dim| dim.dim).collect(),
            View::Padded(dims, _) => dims.iter().map(|dim| dim.dim).collect(),
        }
    }

    pub(crate) fn rank(&self) -> usize {
        match self {
            View::None => 1,
            View::Strided(dims) => dims.len(),
            View::Padded(dims, _) => dims.len(),
        }
    }

    pub(crate) fn requires_conditional_padding(&self) -> bool {
        // View requires conditional padding if any padding is more than zero
        if let View::Padded(_, padded_axes) = self {
            return padded_axes.axes.iter().any(|(_, (lp, rp))| *lp > 0 || *rp > 0);
        }
        false
    }

    /*fn numel(&self) -> usize {
        self.0.iter().map(|dim| dim.dim).product()
    }*/

    pub(crate) fn permute(&mut self, axes: &[usize]) {
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
                let axes_map: BTreeMap<usize, usize> = (0..axes.len()).zip(axes.iter().copied()).collect();
                for (axes, _) in &mut padding.axes {
                    for d in axes {
                        *d = axes_map[d];
                    }
                }
            }
        }
    }

    pub(crate) fn pad(&mut self, axis: Axis, left_pad: isize, right_pad: isize) {
        //println!("Padding view with {left_pad}, {right_pad}");
        match self {
            View::None => {}
            View::Strided(dims) => {
                let mut dims = dims.clone();
                dims[axis].dim = (dims[axis].dim as isize + left_pad + right_pad) as usize;
                *self = View::Padded(dims, PaddedAxes::new(axis, left_pad, right_pad));
            }
            View::Padded(dims, padding) => {
                dims[axis].dim = (dims[axis].dim as isize + left_pad + right_pad) as usize;
                padding.pad(axis, left_pad, right_pad);
            }
        }
    }

    pub(crate) fn expand(&mut self, axis: Axis, dimension: Dimension) {
        let _ = dimension;
        // TODO probably instead of changing stride to 0, we can simply
        // remove the dimension alltogether
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
        }
        /*match self {
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
            View::Padded(dims, _) => {
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
        }*/
    }

    pub(crate) fn numel(&self) -> usize {
        match self {
            View::None => 0,
            View::Strided(dims) => dims.iter().map(|dim| dim.dim).product(),
            View::Padded(dims, _) => dims.iter().map(|dim| dim.dim).product(),
        }
    }

    pub(crate) fn is_contiguous(&self) -> bool {
        &View::new(&self.shape()) == self
    }

    pub(crate) fn split_axis(&mut self, axis: Axis, dimensions: &[usize]) {
        match self {
            View::None => {}
            View::Strided(dims) => {
                // Rename all following axes
                for st_dim in dims.iter_mut() {
                    if axis < st_dim.axis {
                        st_dim.axis += dimensions.len() - 1;
                    }
                }
                if let Some((id, st_dim)) = dims.iter_mut().enumerate().find(|(_, dim)| dim.axis == axis) {
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
                if let Some((id, st_dim)) = dims.iter_mut().enumerate().find(|(_, dim)| dim.axis == axis) {
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
                for (axes, _) in padding.axes.iter_mut() {
                    for a in axes {
                        if *a > axis {
                            *a += dim_len - 1;
                        }
                    }
                }
                // Split padding
                if let Some((axes, _)) = padding.axes.iter_mut().find(|(k, _)| k.contains(&axis)) {
                    //std::println!("Original: {axes:?} splitting into: {axis}..{}", axis+dim_len);
                    for a in axis + 1..axis + dim_len {
                        axes.push(a);
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
                "View::Strided axes: {:?}, shape: {:?}, strides: {:?}",
                dims.iter().map(|d| d.axis).collect::<Vec<Dimension>>(),
                dims.iter().map(|d| d.dim).collect::<Vec<Dimension>>(),
                dims.iter().map(|d| d.stride).collect::<Vec<Stride>>()
            )),
            View::Padded(dims, padding) => f.write_fmt(format_args!(
                "View::Padded axes: {:?}, shape: {:?}, strides: {:?}, padding: {:?}",
                dims.iter().map(|d| d.axis).collect::<Vec<Dimension>>(),
                dims.iter().map(|d| d.dim).collect::<Vec<Dimension>>(),
                dims.iter().map(|d| d.stride).collect::<Vec<Stride>>(),
                padding.axes,
            )),
        }
    }
}
