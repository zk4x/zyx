use crate::{ops::{Reshape, GetShape}, tensor::{Variable, Tensor, Backward, Gradient}, shape::{IntoShape, Shape}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone)]
pub struct ReshapeBackwardV<'g, G> {
    grad: &'g Gradient<G>,
    shape: Shape,
}

impl<S, G> Backward<S> for ReshapeBackwardV<'_, G>
where
    S: Reshape<Output = G>,
    G: Add<G, Output = G>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad.reshape(self.shape));
    }
}

impl<'g, S, G> Reshape for &'g Variable<S, G>
where
    S: Clone + Reshape + GetShape,
{
    type Output = Tensor<<S as Reshape>::Output, ReshapeBackwardV<'g, G>>;
    fn reshape(self, shape: impl IntoShape) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().reshape(shape),
            grad_fn: ReshapeBackwardV {
                grad: &self.grad,
                shape: self.data().shape(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReshapeBackwardT<F> {
    grad_fn: F,
    shape: Shape,
}

impl<S, F> Backward<S> for ReshapeBackwardT<F>
where
    S: Reshape,
    F: Backward<<S as Reshape>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.reshape(self.shape));
    }
}

impl<S, F> Reshape for Tensor<S, F>
where
    S: Reshape + GetShape,
{
    type Output = Tensor<<S as Reshape>::Output, ReshapeBackwardT<F>>;
    fn reshape(self, res_shape: impl IntoShape) -> Self::Output {
        let shape = self.data.shape();
        Tensor {
            data: self.data.reshape(res_shape),
            grad_fn: ReshapeBackwardT {
                grad_fn: self.grad_fn,
                shape,
            }
        }
    }
}