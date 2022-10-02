use crate::{ops::{Reshape, GetShape}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone)]
pub struct ReshapeBackwardG<'g, S> {
    grad: &'g RefCell<S>,
    shape: Vec<usize>,
}

impl<'g, S> Backward<S> for ReshapeBackwardG<'g, S>
where
    S: Default + Reshape<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad + res_grad.reshape(&self.shape));
    }
}

impl<'g, S> Reshape for &'g Variable<S>
where
    S: 'g + Clone + Reshape<Output = S>,
    S: GetShape,
{
    type Output = Tensor<S, ReshapeBackwardG<'g, S>>;
    fn reshape(self, shape: &[usize]) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().reshape(shape),
            func: ReshapeBackwardG {
                grad: &self.grad,
                shape: self.data().shape(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReshapeBackwardF<F> {
    func: F,
    shape: Vec<usize>,
}

impl<S, F> Backward<S> for ReshapeBackwardF<F>
where
    S: Reshape<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(res_grad.reshape(&self.shape));
    }
}

impl<S, F> Reshape for Tensor<S, F>
where
    S: Reshape<Output = S> + GetShape,
{
    type Output = Tensor<S, ReshapeBackwardF<F>>;
    fn reshape(self, res_shape: &[usize]) -> Self::Output {
        let shape = self.data.shape();
        Tensor {
            data: self.data.reshape(res_shape),
            func: ReshapeBackwardF {
                func: self.func,
                shape,
            }
        }
    }
}