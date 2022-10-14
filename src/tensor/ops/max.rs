use crate::{ops::{Max, Expand, GetShape}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone)]
pub struct MaxBackwardV<'g, S> {
    grad: &'g RefCell<S>,
    shape: Vec<usize>,
}

impl<'g, S> Backward<S> for MaxBackwardV<'g, S>
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
    type Output = Tensor<S, MaxBackwardV<'g, S>>;
    fn max(self, dims: &[i32]) -> Self::Output {
        Tensor {
            data: (*self.data.borrow()).clone().max(dims),
            func: MaxBackwardV {
                grad: &self.grad,
                shape: self.data.borrow().shape(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaxBackwardT<F> {
    func: F,
    shape: Vec<usize>,
}

impl<S, F> Backward<S> for MaxBackwardT<F>
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
    type Output = Tensor<S, MaxBackwardT<F>>;
    fn max(self, dims: &[i32]) -> Self::Output {
        let shape = self.data.shape();
        Tensor {
            data: self.data.max(dims),
            func: MaxBackwardT {
                func: self.func,
                shape,
            },
        }
    }
}
