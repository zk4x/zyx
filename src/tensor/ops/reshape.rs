use crate::{ops::{Reshape, GetShape}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}, shape::{IntoShape, Shape}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone)]
pub struct ReshapeBackwardV<'g, S> {
    grad: &'g RefCell<S>,
    shape: Shape,
}

impl<'g, S> Backward<S> for ReshapeBackwardV<'g, S>
where
    S: Default + Reshape<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad + res_grad.reshape(self.shape));
    }
}

impl<'g, S> Reshape for &'g Variable<S>
where
    S: 'g + Clone + Reshape<Output = S> + GetShape,
{
    type Output = Tensor<S, ReshapeBackwardV<'g, S>>;
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
    S: Reshape<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.reshape(self.shape));
    }
}

impl<S, F> Reshape for Tensor<S, F>
where
    S: Reshape<Output = S> + GetShape,
{
    type Output = Tensor<S, ReshapeBackwardT<F>>;
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