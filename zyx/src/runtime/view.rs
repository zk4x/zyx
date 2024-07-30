use alloc::vec::Vec;

pub(crate) type Axis = usize;
pub(crate) type Dimension = usize;
pub(crate) type Stride = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StridedDim {
    axis: Axis,
    dim: Dimension,
    stride: Stride,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PaddedDim {
    axis: Axis,
    dim: Dimension,
    stride: Stride,
    lp: isize,
    rp: isize,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) enum View {
    None,
    //Contiguous(Vec<StridedDim>), // TODO perhaps later, mainly for cpu and perhaps wide loads on gpu
    Strided(Vec<StridedDim>),
    Padded(Vec<PaddedDim>),
    //Reshaped(), // TODO perhaps for some weird optimizations, but it may actually reduce performace
    // since then loads are very unpredictable
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
        self.0.iter().map(|dim| dim.dim).collect()
    }

    pub(crate) fn rank(&self) -> usize {
        self.0.len()
    }

    /*fn numel(&self) -> usize {
        self.0.iter().map(|dim| dim.dim).product()
    }*/

    pub(crate) fn permute(&mut self, axes: &[usize]) {
        assert_eq!(self.0.len(), axes.len());
        self.0 = axes.iter().map(|axis| self.0[*axis]).collect();
        for (a, dim) in self.0.iter_mut().enumerate() {
            dim.axis = a;
        }
    }

    pub(crate) fn pad(&mut self, padding: &[(isize, isize)]) {
        for (a, (lp, rp)) in all_axes.iter().copied().zip(padding.iter().skip(shape.len() - n)) {
            self.0[a].dim = (view.0[a].dim as isize + lp + rp) as usize;
            self.0[a].lp = *lp;
            self.0[a].rp = *rp;
        }
    }

    pub(crate) fn split_axis(&mut self, axis: Axis, dimensions: &[usize]) {
        let mut stride = view.0[axis].stride;
        let PaddedDim { lp, rp, .. } = view.0.remove(axis);
        let mut temp_axis = axis + dimensions.len();
        for dim in dimensions.iter().copied().rev() {
            temp_axis -= 1;
            view.0.insert(
                axis,
                PaddedDim {
                    axis: temp_axis,
                    dim,
                    stride,
                    // TODO
                    lp: 0,
                    rp: 0,
                },
            );
            stride *= dim;
        }
        // TODO do proper padding on splitted dimension
        view.0[axis].lp = lp;
        view.0[axis].rp = rp;
        // Rename all following axes
        for a in axis + dimensions.len()..view.0.len() {
            view.0[a].axis += dimensions.len() - 1;
        }
    }

    pub(crate) fn expand(&mut self, shape: &[usize]) {
        view.0[*a].dim = shape[*a];
        view.0[*a].stride = 0;
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

    pub(crate) fn is_contiguous(&self) -> bool {
        &View::new(&self.shape()) == self
    }
}

#[cfg(any(
    feature = "cuda",
    feature = "opencl",
    feature = "wgsl",
    feature = "hsa"
))]
use super::compiler::Scope;
#[cfg(any(
    feature = "cuda",
    feature = "opencl",
    feature = "wgsl",
    feature = "hsa"
))]
use alloc::{string::String, format as f};

#[cfg(any(
    feature = "cuda",
    feature = "opencl",
    feature = "wgsl",
    feature = "hsa"
))]
impl View {
    pub(crate) fn to_str(&self, id: u64, scope: Scope, temp_id: u8) -> (Vec<String>, String) {
        match self {
            View::None => return (Vec::new(), f!("{}{}", scope, id)),
            View::Strided { dims } => {
                //std::println!("Using contiguous or strided index");
                let mut res = String::new();
                for (id, mul) in dims {
                    res += &f!("i{id}*{mul}+");
                }
                res.pop();
                return (Vec::new(), f!("{}{}[{res}]", scope, id));
            }
            View::Padded { dims } => {
                //std::println!("Using padded index");
                let mut res = String::new();
                // When the padding does not apply
                let mut padding_condition = String::new();
                for (id, (dim, mul, lp, rp)) in dims {
                    //std::println!("Padding {id} with {lp}, {rp}");
                    if *lp > 0 {
                        padding_condition += &f!("i{id} < {lp} || ");
                        res += &f!("(i{id}-{lp})*{mul}+");
                    } else {
                        let lp = -lp;
                        res += &f!("(i{id}+{lp})*{mul}+");
                    }
                    // rp negative does essentially nothing, we only care if it's positive
                    if *rp > 0 {
                        padding_condition += &f!("i{id} > {} || ", *dim as isize - lp - rp);
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
            }
            /*View::Reshaped { dims, reshapes, .. } => {
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
        },

    }
}
