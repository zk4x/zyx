use crate::{ops::{Permute}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}, shape::{IntoDims, Dims}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone)]
pub struct PermuteBackwardV<'g, S> {
    grad: &'g RefCell<S>,
    dims: Dims,
}

impl<'g, S> Backward<S> for PermuteBackwardV<'g, S>
where
    S: Default + Permute<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad + res_grad.permute(self.dims));
    }
}

impl<'g, S> Permute for &'g Variable<S>
where
    S: 'g + Clone + Permute<Output = S>,
{
    type Output = Tensor<S, PermuteBackwardV<'g, S>>;
    fn permute(self, dims: impl IntoDims) -> Self::Output {
        let dims = dims.dims();
        Tensor {
            data: self.data().clone().permute(dims.clone()),
            grad_fn: PermuteBackwardV {
                grad: &self.grad,
                dims: dims.argsort(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PermuteBackwardT<F> {
    grad_fn: F,
    dims: Dims,
}

impl<S, F> Backward<S> for PermuteBackwardT<F>
where
    S: Permute<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.permute(self.dims));
    }
}

impl<S, F> Permute for Tensor<S, F>
where
    S: Permute<Output = S>,
{
    type Output = Tensor<S, PermuteBackwardT<F>>;
    fn permute(self, dims: impl IntoDims) -> Self::Output {
        let dims = dims.dims();
        Tensor {
            data: self.data.permute(dims.clone()),
            grad_fn: PermuteBackwardT {
                grad_fn: self.grad_fn,
                dims: dims.argsort(),
            }
        }
    }
}