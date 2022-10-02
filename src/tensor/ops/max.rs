use crate::{ops::{Max, Expand, GetShape}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone)]
pub struct MaxBackwardG<'g, S> {
    grad: &'g RefCell<S>,
    shape: Vec<usize>,
}

impl<'g, S> Backward<S> for MaxBackwardG<'g, S>
where
    S: Default + Add<Output = S> + Expand<Output = S> + GetShape,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad + res_grad.expand(&self.shape));
    }
}

impl<'g, S> Max for &'g Variable<S>
where
    S: 'g + Clone + Max<Output = S>,
    S: GetShape,
{
    type Output = Tensor<S, MaxBackwardG<'g, S>>;
    fn max(self, dims: &[i32]) -> Self::Output {
        Tensor {
            data: (*self.data.borrow()).clone().max(dims),
            func: MaxBackwardG {
                grad: &self.grad,
                shape: self.data.borrow().shape(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaxBackwardF<F> {
    func: F,
    shape: Vec<usize>,
}

impl<S, F> Backward<S> for MaxBackwardF<F>
where
    S: Expand<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(res_grad.expand(&self.shape));
    }
}

impl<S, F> Max for Tensor<S, F>
where
    S: Max<Output = S> + GetShape,
{
    type Output = Tensor<S, MaxBackwardF<F>>;
    fn max(self, dims: &[i32]) -> Self::Output {
        let shape = self.data.shape();
        Tensor {
            data: self.data.max(dims),
            func: MaxBackwardF {
                func: self.func,
                shape,
            },
        }
    }
}
