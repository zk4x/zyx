use crate::{ops::{ReLU, DReLU}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::{Add, Mul}, cell::RefCell};

#[derive(Debug, Clone, Copy)]
pub struct ReLUBackwardLeaf<'g, S> {
    grad: &'g RefCell<S>,
    data: S,
}

impl<'g, S> Backward<S> for ReLUBackwardLeaf<'g, S>
where
    S: Default + DReLU<Output = S> + Mul<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad + res_grad * self.data.drelu());
    }
}

impl<'g, S> ReLU for &'g Variable<S>
where
    S: 'g + Clone + ReLU<Output = S>,
{
    type Output = Tensor<S, ReLUBackwardLeaf<'g, S>>;
    fn relu(self) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().relu(),
            func: ReLUBackwardLeaf {
                grad: &self.grad,
                data: (*self.data()).clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReLUBackward<S, F> {
    func: F,
    data: S,
}

impl<S, F> Backward<S> for ReLUBackward<S, F>
where
    S: DReLU<Output = S> + Mul<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(res_grad * self.data.drelu());
    }
}

impl<S, F> ReLU for Tensor<S, F>
where
    S: Clone + ReLU<Output = S>,
{
    type Output = Tensor<S, ReLUBackward<S, F>>;
    fn relu(self) -> Self::Output {
        Tensor {
            data: self.data.clone().relu(),
            func: ReLUBackward {
                func: self.func,
                data: self.data,
            },
        }
    }
}
