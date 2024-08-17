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
    axes: Vec<(Vec<Axis>, (isize, isize))>,
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

    /*fn numel(&self) -> usize {
        self.0.iter().map(|dim| dim.dim).product()
    }*/

    pub(crate) fn permute(&mut self, axes: &[usize]) {
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
                    // If axes within single padding group are permuted, there is no change
                    // If axes within different groups are flipped, then what?
                    todo!()
                }
            }
        }
    }

    pub(crate) fn pad(&mut self, axis: Axis, left_pad: isize, right_pad: isize) {
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

    pub(crate) fn split_axis(&mut self, axis: Axis, dimensions: &[usize]) {
        match self {
            View::None => {}
            View::Strided(dims) => {
                let mut stride = dims[axis].stride;
                dims.remove(axis);
                let mut temp_axis = axis + dimensions.len();
                for dim in dimensions.iter().copied().rev() {
                    temp_axis -= 1;
                    dims.insert(
                        axis,
                        StridedDim {
                            axis: temp_axis,
                            dim,
                            stride,
                        },
                    );
                    stride *= dim;
                }
                // Rename all following axes
                for a in axis + dimensions.len()..dims.len() {
                    dims[a].axis += dimensions.len() - 1;
                }
            }
            View::Padded(dims, padding) => {
                let dim_len = dimensions.len();
                let mut stride = dims[axis].stride;
                dims.remove(axis);
                let mut temp_axis = axis + dim_len;
                for dim in dimensions.iter().copied().rev() {
                    temp_axis -= 1;
                    dims.insert(
                        axis,
                        StridedDim {
                            axis: temp_axis,
                            dim,
                            stride,
                        },
                    );
                    stride *= dim;
                }
                // Rename all following axes
                for a in (axis + dim_len..dims.len()).rev() {
                    dims[a].axis += dim_len - 1;
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
                        //if !axes.contains(&a) {
                        //}
                        axes.push(a);
                    }
                    // Would not be needed on btreeset
                    axes.sort();
                }
            }
        }
    }

    pub(crate) fn expand(&mut self, axis: Axis, dimension: Dimension) {
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
        }
    }

    /*pub(super) fn index(&self) -> Index {
        // TODO add index for padded views
        if self.is_contiguous() {
            Index::Contiguous {
                dims: self.0.iter().map(|dim| (dim.axis, dim.stride)).collect(),
            }
        } else if self.0.iter().all(|dim| dim.lp == 0 && dim.rp == 0) {
            Index::Strided {
                dims: self.0.iter().map(|dim| (dim.axis, dim.stride)).collect(),
            }
        } else {
            Index::Padded {
                dims: self
                    .0
                    .iter()
                    .map(|dim| (dim.axis, (dim.dim, dim.stride, dim.lp, dim.rp)))
                    .collect(),
            }
        }
    }*/

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
}

/*use std::{fmt::Display, format as f};

pub(crate) enum Scope {
    Global,
    Local,
    Register,
}

impl Display for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Scope::Global => "g",
            Scope::Local => "l",
            Scope::Register => "r",
        })
    }
}*/

/*impl View {
    pub(crate) fn to_str(&self, id: u64, scope: Scope, _temp_id: u8) -> (Vec<String>, String) {
        match self {
            View::None => return (Vec::new(), f!("{}{}", scope, id)),
            View::Strided(dims) => {
                //std::println!("Using contiguous or strided index");
                let mut res = String::new();
                for StridedDim { axis, stride, .. } in dims {
                    res += &f!("i{axis}*{stride}+");
                }
                res.pop();
                return (Vec::new(), f!("{}{}[{res}]", scope, id));
            }
            View::Padded(dims, padding) => {
                //std::println!("Using padded index");
                let mut res = String::new();
                // When the padding does not apply
                let mut padding_condition = String::new();
                for StridedDim { axis, stride, .. } in dims {
                    if let Some((axes, (lp, rp))) = padding
                        .axes
                        .iter()
                        .find(|(axes, _)| axes.iter().max().unwrap() == axis)
                    {
                        //std::println!("Padding {id} with {lp}, {rp}");
                        let mut idx = String::new();
                        let mut st = 1;
                        let mut dim = 1;
                        for axis in axes.iter().rev() {
                            idx = f!("i{axis}*{st}+{idx}");
                            st *= dims[*axis].dim;
                            dim *= dims[*axis].dim;
                        }
                        idx.pop();
                        if *lp > 0 {
                            padding_condition += &f!("{idx} < {lp} || ");
                            res += &f!("(i{axis}-{lp})*{stride}+");
                        } else if *lp < 0 {
                            let lp = -lp;
                            res += &f!("(i{axis}+{lp})*{stride}+");
                        } else {
                            res += &f!("i{axis}*{stride}+");
                        }
                        // rp negative does essentially nothing, we only care if it's positive
                        //std::println!("dim: {dim}, paddding {lp}, {rp}");
                        if *rp > 0 {
                            padding_condition += &f!("{idx} > {} || ", dim as isize - rp - 1);
                        }
                    } else {
                        res += &f!("i{axis}*{stride}+");
                    }
                }
                res.pop();
                return (
                    Vec::new(),
                    if padding_condition.is_empty() {
                        f!("{}{}[{res}]", scope, id)
                    } else {
                        f!(
                            "{} ? 0 : {}{}[{res}]",
                            &padding_condition[..padding_condition.len() - 4],
                            scope,
                            id
                        )
                    },
                );
            } /*View::Reshaped { dims, reshapes, .. } => {
              let mut res = String::new();
              for (id, mul) in dims {
                  res += &f!("i{id}*{mul}+");
              }
              res.pop();
              let mut res = vec![res];
              for reshape in reshapes[..reshapes.len() - 1].iter() {
                  let mut idx = String::new();
                  for (div, m, mul) in reshape.iter() {
                      idx += &f!("t{temp_id}/{div}%{m}*{mul}+");
                  }
                  idx.pop();
                  res.push(idx);
              }
              let mut idx = String::new();
              for (div, m, mul) in reshapes.last().unwrap().iter() {
                  idx += &f!("t{temp_id}/{div}%{m}*{mul}+");
              }
              idx.pop();
              return (res, f!("{}{}[{idx}]", scope, id));
              }*/
        }
    }
}*/
