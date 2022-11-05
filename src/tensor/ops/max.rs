use crate::{ops::{Max, Expand, GetShape}, tensor::{Variable, Tensor, Backward, GradientRef}, shape::{IntoDims, Shape}};
use std::ops::Add;

#[derive(Debug, Clone)]
pub struct MaxBackwardV<'g, G> {
    grad: GradientRef<'g, G>,
    shape: Shape,
}

impl<S, G> Backward<S> for MaxBackwardV<'_, G>
where
    S: Expand<Output = G>,
    G: Add<G, Output = G>,
{
    fn backward(self, res_grad: S) {
        // TODO: This is not correct. Max does not simply expand.
        // Max sets values at max indices to 1 and other values to 0.
        // So res_grad values must be added to indices where there were maximums previously.
        // So Instead of shape, we need to store indices of those values.
        self.grad.accumulate(res_grad.expand(self.shape));
    }
}

impl<'g, S, G> Max for &'g Variable<S, G>
where
    S: Clone + Max<Output = S> + GetShape,
{
    type Output = Tensor<S, MaxBackwardV<'g, G>>;
    fn max(self, dims: impl IntoDims) -> Self::Output {
        Tensor {
            data: (*self.data.borrow()).clone().max(dims),
            grad_fn: MaxBackwardV {
                grad: GradientRef::new(&self.grad),
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
    S: Expand,
    F: Backward<<S as Expand>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.expand(self.shape));
    }
}

impl<S, F> Max for Tensor<S, F>
where
    S: Max + GetShape,
{
    type Output = Tensor<<S as Max>::Output, MaxBackwardT<F>>;
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
