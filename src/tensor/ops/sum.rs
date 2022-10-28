use crate::{ops::{Sum, Expand, GetShape}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}, shape::{Shape, IntoDims}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone)]
pub struct SumBackwardV<'g, S> {
    grad: &'g RefCell<S>,
    shape: Shape,
}

impl<'g, S> Backward<S> for SumBackwardV<'g, S>
where
    S: Default + Add<Output = S> + Expand<Output = S> + GetShape,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad + res_grad.expand(self.shape));
    }
}

impl<'g, S> Sum for &'g Variable<S>
where
    S: 'g + Clone + Sum<Output = S> + GetShape,
{
    type Output = Tensor<S, SumBackwardV<'g, S>>;
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
    S: Clone + Sum<Output = S> + GetShape,
{
    type Output = Tensor<S, SumBackwardT<F>>;
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
