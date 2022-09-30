use crate::{ops::{Reshape, GetShape}, tensor::{Tensor, TensorGrad, TensorFunc, Backward}};
use std::{rc::Rc, ops::Add, cell::RefCell};

impl<S> Reshape for Tensor<S>
where
    for<'a> &'a S: Reshape<Output = S>,
{
    type Output = Tensor<S>;
    fn reshape(self, shape: &[usize]) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.reshape(shape)),
        }
    }
}

#[derive(Debug)]
pub struct ReshapeBackwardG<'g, S> {
    grad: &'g RefCell<S>,
    shape: Vec<usize>,
}

impl<'g, S> Backward<S> for ReshapeBackwardG<'g, S>
where
    for<'a> &'a S: Reshape<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_with(|grad| &*grad + &res_grad.reshape(&self.shape));
    }
}

impl<'g, S> Reshape for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Reshape<Output = S> + GetShape,
{
    type Output = TensorFunc<S, ReshapeBackwardG<'g, S>>;
    fn reshape(self, shape: &[usize]) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data().reshape(shape)),
            func: ReshapeBackwardG {
                grad: &self.grad,
                shape: self.data.borrow().shape(),
            }
        }
    }
}

#[derive(Debug)]
pub struct ReshapeBackwardF<F> {
    func: F,
    shape: Vec<usize>,
}

impl<S, F> Backward<S> for ReshapeBackwardF<F>
where
    for<'a> &'a S: Reshape<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(res_grad.reshape(&self.shape));
    }
}

impl<S, F> Reshape for TensorFunc<S, F>
where
    for<'a> &'a S: Reshape<Output = S> + GetShape,
{
    type Output = TensorFunc<S, ReshapeBackwardF<F>>;
    fn reshape(self, shape: &[usize]) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.reshape(shape)),
            func: ReshapeBackwardF {
                func: self.func,
                shape: self.data.shape(),
            }
        }
    }
}