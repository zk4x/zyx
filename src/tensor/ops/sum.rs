use crate::{ops::{Sum, Expand, GetShape}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone)]
pub struct SumBackwardG<'g, S> {
    grad: &'g RefCell<S>,
    shape: Vec<usize>,
}

impl<'g, S> Backward<S> for SumBackwardG<'g, S>
where
    S: Default + Add<Output = S> + Expand<Output = S> + GetShape,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad + res_grad.expand(&self.shape));
    }
}

impl<'g, S> Sum for &'g Variable<S>
where
    S: 'g + Clone + Sum<Output = S> + GetShape,
{
    type Output = Tensor<S, SumBackwardG<'g, S>>;
    fn sum(self, dims: &[i32]) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().sum(dims),
            func: SumBackwardG {
                grad: &self.grad,
                shape: self.data().shape(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SumBackwardF<F> {
    func: F,
    shape: Vec<usize>,
}

impl<S, F> Backward<S> for SumBackwardF<F>
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
    S: Clone + Sum<Output = S> + GetShape,
{
    type Output = Tensor<S, SumBackwardF<F>>;
    fn sum(self, dims: &[i32]) -> Self::Output {
        let shape = self.data.shape();
        Tensor {
            data: self.data.sum(dims),
            func: SumBackwardF {
                func: self.func,
                shape,
            }
        }
    }
}
