use crate::{ops::{Min, Expand, GetShape}, tensor::{Tensor, TensorGrad, TensorFunc, Backward}};
use std::{rc::Rc, ops::Add, cell::RefCell};

impl<S> Min for Tensor<S>
where
    for<'a> &'a S: Min<Output = S>,
{
    type Output = Tensor<S>;
    fn min(self, dims: &[i32]) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.min(dims)),
        }
    }
}

#[derive(Debug)]
pub struct MinBackwardG<'g, S> {
    grad: &'g RefCell<S>,
    shape: Vec<usize>,
}

impl<'g, S> Backward<S> for MinBackwardG<'g, S>
where
    for<'a> &'a S: Add<Output = S> + Expand<Output = S> + GetShape,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_with(|grad| &*grad + &res_grad.expand(&self.shape));
    }
}

impl<'g, S> Min for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Min<Output = S> + GetShape,
{
    type Output = TensorFunc<S, MinBackwardG<'g, S>>;
    fn min(self, dims: &[i32]) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.borrow().min(dims)),
            func: MinBackwardG {
                grad: &self.grad,
                shape: self.data.borrow().shape(),
            }
        }
    }
}

#[derive(Debug)]
pub struct MinBackwardF<F> {
    func: F,
    shape: Vec<usize>,
}

impl<S, F> Backward<S> for MinBackwardF<F>
where
    for<'a> &'a S: Add<Output = S> + Expand<Output = S> + GetShape,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(res_grad.expand(&self.shape));
    }
}

impl<S, F> Min for TensorFunc<S, F>
where
    for<'a> &'a S: Min<Output = S> + GetShape,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, MinBackwardF<F>>;
    fn min(self, dims: &[i32]) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.min(dims)),
            func: MinBackwardF {
                func: self.func,
                shape: self.data.shape(),
            }
        }
    }
}
