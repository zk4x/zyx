use crate::{ops::{ReLU, DReLU}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::{Add, Mul}, cell::RefCell};

#[derive(Debug, Clone, Copy)]
pub struct ReLUBackwardV<'g, S> {
    grad: &'g RefCell<S>,
    data: S,
}

impl<'g, S> Backward<S> for ReLUBackwardV<'g, S>
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
    type Output = Tensor<S, ReLUBackwardV<'g, S>>;
    fn relu(self) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().relu(),
            func: ReLUBackwardV {
                grad: &self.grad,
                data: (*self.data()).clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReLUBackwardT<S, F> {
    func: F,
    data: S,
}

impl<S, F> Backward<S> for ReLUBackwardT<S, F>
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
    type Output = Tensor<S, ReLUBackwardT<S, F>>;
    fn relu(self) -> Self::Output {
        Tensor {
            data: self.data.clone().relu(),
            func: ReLUBackwardT {
                func: self.func,
                data: self.data,
            },
        }
    }
}
