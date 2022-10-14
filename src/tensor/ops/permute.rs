use crate::{shape::Dims, ops::Permute, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone)]
pub struct PermuteBackwardV<'g, S> {
    grad: &'g RefCell<S>,
    dims: Vec<i32>,
}

impl<'g, S> Backward<S> for PermuteBackwardV<'g, S>
where
    S: Default + Permute<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad + res_grad.permute(&self.dims));
    }
}

impl<'g, S> Permute for &'g Variable<S>
where
    S: 'g + Clone + Permute<Output = S>,
{
    type Output = Tensor<S, PermuteBackwardV<'g, S>>;
    fn permute(self, dims: &[i32]) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().permute(dims),
            func: PermuteBackwardV {
                grad: &self.grad,
                dims: dims.argsort(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PermuteBackwardT<F> {
    func: F,
    dims: Vec<i32>,
}

impl<S, F> Backward<S> for PermuteBackwardT<F>
where
    S: Permute<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(res_grad.permute(&self.dims));
    }
}

impl<S, F> Permute for Tensor<S, F>
where
    S: Permute<Output = S>,
{
    type Output = Tensor<S, PermuteBackwardT<F>>;
    fn permute(self, dims: &[i32]) -> Self::Output {
        Tensor {
            data: self.data.permute(dims),
            func: PermuteBackwardT {
                func: self.func,
                dims: dims.argsort(),
            }
        }
    }
}