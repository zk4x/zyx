use crate::{ops::{Min, Expand, IntoShape}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone)]
pub struct MinBackwardV<'g, S> {
    grad: &'g RefCell<S>,
    shape: Vec<usize>,
}

impl<'g, S> Backward<S> for MinBackwardV<'g, S>
where
    S: Default + Add<Output = S> + Expand<Output = S> + IntoShape,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad + res_grad.expand(&self.shape));
    }
}

impl<'g, S> Min for &'g Variable<S>
where
    S: 'g + Clone + Min<Output = S>,
    S: IntoShape,
{
    type Output = Tensor<S, MinBackwardV<'g, S>>;
    fn min(self, dims: &[i32]) -> Self::Output {
        Tensor {
            data: (*self.data.borrow()).clone().min(dims),
            func: MinBackwardV {
                grad: &self.grad,
                shape: self.data.borrow().shape(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MinBackwardT<F> {
    func: F,
    shape: Vec<usize>,
}

impl<S, F> Backward<S> for MinBackwardT<F>
where
    S: Expand<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(res_grad.expand(&self.shape));
    }
}

impl<S, F> Min for Tensor<S, F>
where
    S: Min<Output = S> + IntoShape,
    F: FnOnce(S),
{
    type Output = Tensor<S, MinBackwardT<F>>;
    fn min(self, dims: &[i32]) -> Self::Output {
        let shape = self.data.shape();
        Tensor {
            data: self.data.min(dims),
            func: MinBackwardT {
                func: self.func,
                shape,
            }
        }
    }
}
