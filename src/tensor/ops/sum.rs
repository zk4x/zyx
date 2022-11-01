use crate::{ops::{Sum, Expand, GetShape}, tensor::{Variable, Tensor, Backward, Gradient}, shape::{Shape, IntoDims}};
use std::ops::Add;

#[derive(Debug, Clone)]
pub struct SumBackwardV<'g, G> {
    grad: &'g Gradient<G>,
    shape: Shape,
}

impl<S> Backward<S> for SumBackwardV<'_, S>
where
    S: Default + Add<Output = S> + Expand<Output = S> + GetShape,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad.expand(self.shape));
    }
}

impl<'g, S, G> Sum for &'g Variable<S, G>
where
    S: Clone + Sum + GetShape,
{
    type Output = Tensor<<S as Sum>::Output, SumBackwardV<'g, G>>;
    fn sum(self, dims: impl IntoDims) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().sum(dims),
            grad_fn: SumBackwardV {
                grad: &self.grad,
                shape: self.data().shape(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SumBackwardT<F> {
    grad_fn: F,
    shape: Shape,
}

impl<S, F> Backward<S> for SumBackwardT<F>
where
    S: Expand,
    F: Backward<<S as Expand>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.expand(self.shape));
    }
}

impl<S, F> Sum for Tensor<S, F>
where
    S: Clone + Sum + GetShape,
{
    type Output = Tensor<<S as Sum>::Output, SumBackwardT<F>>;
    fn sum(self, dims: impl IntoDims) -> Self::Output {
        let shape = self.data.shape();
        Tensor {
            data: self.data.sum(dims),
            grad_fn: SumBackwardT {
                grad_fn: self.grad_fn,
                shape,
            }
        }
    }
}
