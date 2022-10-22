use crate::{ops::{Max, Expand, GetShape}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}, shape::{IntoDims, Shape}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone)]
pub struct MaxBackwardV<'g, S> {
    grad: &'g RefCell<S>,
    shape: Shape,
}

impl<'g, S> Backward<S> for MaxBackwardV<'g, S>
where
    S: Default + Add<Output = S> + Expand<Output = S> + GetShape,
{
    fn backward(self, res_grad: S) {
        // TODO: This is not correct. Max does not simply expand.
        // Max sets values at max indices to 1 and other values to 0.
        // So res_grad values must be added to indices where there were maximums previously.
        // So Instead of shape, we need to store indices of those values.
        self.grad.replace_take(|grad| grad + res_grad.expand(self.shape));
    }
}

impl<'g, S> Max for &'g Variable<S>
where
    S: 'g + Clone + Max<Output = S> + GetShape,
{
    type Output = Tensor<S, MaxBackwardV<'g, S>>;
    fn max(self, dims: impl IntoDims) -> Self::Output {
        Tensor {
            data: (*self.data.borrow()).clone().max(dims),
            grad_fn: MaxBackwardV {
                grad: &self.grad,
                shape: self.data.borrow().shape(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaxBackwardT<F> {
    grad_fn: F,
    shape: Shape,
}

impl<S, F> Backward<S> for MaxBackwardT<F>
where
    S: Expand<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.expand(self.shape));
    }
}

impl<S, F> Max for Tensor<S, F>
where
    S: Max<Output = S> + GetShape,
{
    type Output = Tensor<S, MaxBackwardT<F>>;
    fn max(self, dims: impl IntoDims) -> Self::Output {
        let shape = self.data.shape();
        Tensor {
            data: self.data.max(dims),
            grad_fn: MaxBackwardT {
                grad_fn: self.grad_fn,
                shape,
            },
        }
    }
}
