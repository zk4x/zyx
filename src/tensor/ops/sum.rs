use crate::{ops::{Sum, Expand, IntoShape}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone)]
pub struct SumBackwardV<'g, S> {
    grad: &'g RefCell<S>,
    shape: Vec<usize>,
}

impl<'g, S> Backward<S> for SumBackwardV<'g, S>
where
    S: Default + Add<Output = S> + Expand<Output = S> + IntoShape,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad + res_grad.expand(&self.shape));
    }
}

impl<'g, S> Sum for &'g Variable<S>
where
    S: 'g + Clone + Sum<Output = S> + IntoShape,
{
    type Output = Tensor<S, SumBackwardV<'g, S>>;
    fn sum(self, dims: &[i32]) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().sum(dims),
            func: SumBackwardV {
                grad: &self.grad,
                shape: self.data().shape(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SumBackwardT<F> {
    func: F,
    shape: Vec<usize>,
}

impl<S, F> Backward<S> for SumBackwardT<F>
where
    S: Expand<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(res_grad.expand(&self.shape));
    }
}

impl<S, F> Sum for Tensor<S, F>
where
    S: Clone + Sum<Output = S> + IntoShape,
{
    type Output = Tensor<S, SumBackwardT<F>>;
    fn sum(self, dims: &[i32]) -> Self::Output {
        let shape = self.data.shape();
        Tensor {
            data: self.data.sum(dims),
            func: SumBackwardT {
                func: self.func,
                shape,
            }
        }
    }
}
