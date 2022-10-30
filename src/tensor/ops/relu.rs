use crate::{ops::{ReLU, DReLU}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::{Add, Mul}, cell::RefCell};

#[derive(Debug, Clone, Copy)]
pub struct ReLUBackwardV<'g, S> {
    grad: &'g RefCell<S>,
    data: S,
}

impl<S, S2> Backward<S> for ReLUBackwardV<'_, S2>
where
    S2: Default + DReLU + Add<<S as Mul<<S2 as DReLU>::Output>>::Output, Output = S2>,
    S: Mul<<S2 as DReLU>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad + res_grad * self.data.drelu());
    }
}

impl<'g, S> ReLU for &'g Variable<S>
where
    S: Clone + ReLU,
{
    type Output = Tensor<<S as ReLU>::Output, ReLUBackwardV<'g, S>>;
    fn relu(self) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().relu(),
            grad_fn: ReLUBackwardV {
                grad: &self.grad,
                data: (*self.data()).clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReLUBackwardT<S, F> {
    grad_fn: F,
    data: S,
}

impl<S, S2, F> Backward<S> for ReLUBackwardT<S2, F>
where
    S2: DReLU,
    S: Mul<<S2 as DReLU>::Output>,
    F: Backward<<S as Mul<<S2 as DReLU>::Output>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad * self.data.drelu());
    }
}

impl<S, F> ReLU for Tensor<S, F>
where
    S: Clone + ReLU,
{
    type Output = Tensor<<S as ReLU>::Output, ReLUBackwardT<S, F>>;
    fn relu(self) -> Self::Output {
        Tensor {
            data: self.data.clone().relu(),
            grad_fn: ReLUBackwardT {
                grad_fn: self.grad_fn,
                data: self.data,
            },
        }
    }
}
