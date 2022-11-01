use crate::{ops::Permute, tensor::{Variable, Tensor, Backward, Gradient}, shape::{IntoDims, Dims}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone)]
pub struct PermuteBackwardV<'g, G> {
    grad: &'g Gradient<G>,
    dims: Dims,
}

impl<S, S2> Backward<S> for PermuteBackwardV<'_, S2>
where
    S2: Add<S2, Output = S2>,
    S: Permute<Output = S2>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad.permute(self.dims));
    }
}

impl<'g, S, G> Permute for &'g Variable<S, G>
where
    S: Clone + Permute,
{
    type Output = Tensor<<S as Permute>::Output, PermuteBackwardV<'g, S>>;
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
    S: Permute,
    F: Backward<<S as Permute>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.permute(self.dims));
    }
}

impl<S, F> Permute for Tensor<S, F>
where
    S: Permute,
{
    type Output = Tensor<<S as Permute>::Output, PermuteBackwardT<F>>;
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