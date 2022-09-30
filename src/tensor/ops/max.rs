use crate::{ops::{Max, Expand, GetShape}, tensor::{Tensor, TensorGrad, TensorFunc, Backward}};
use std::{rc::Rc, ops::Add, cell::RefCell};

impl<S> Max for Tensor<S>
where
    for<'a> &'a S: Max<Output = S>,
{
    type Output = Tensor<S>;
    fn max(self, dims: &[i32]) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.max(dims)),
        }
    }
}

#[derive(Debug)]
pub struct MaxBackwardG<'g, S> {
    grad: &'g RefCell<S>,
    shape: Vec<usize>,
}

impl<'g, S> Backward<S> for MaxBackwardG<'g, S>
where
    for<'a> &'a S: Add<Output = S> + Expand<Output = S> + GetShape,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_with(|grad| &*grad + &res_grad.expand(&self.shape));
    }
}

impl<'g, S> Max for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Max<Output = S> + GetShape,
{
    type Output = TensorFunc<S, MaxBackwardG<'g, S>>;
    fn max(self, dims: &[i32]) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.borrow().max(dims)),
            func: MaxBackwardG {
                grad: &self.grad,
                shape: self.data.borrow().shape(),
            }
        }
    }
}

#[derive(Debug)]
pub struct MaxBackwardF<F> {
    func: F,
    shape: Vec<usize>,
}

impl<S, F> Backward<S> for MaxBackwardF<F>
where
    for<'a> &'a S: Add<Output = S> + Expand<Output = S> + GetShape,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(res_grad.expand(&self.shape));
    }
}

impl<S, F> Max for TensorFunc<S, F>
where
    for<'a> &'a S: Max<Output = S> + GetShape,
{
    type Output = TensorFunc<S, MaxBackwardF<F>>;
    fn max(self, dims: &[i32]) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.max(dims)),
            func: MaxBackwardF {
                func: self.func,
                shape: self.data.shape(),
            },
        }
    }
}
