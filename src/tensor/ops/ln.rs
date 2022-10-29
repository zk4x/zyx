use crate::{ops::{Ln, Pow}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::{Add, Mul}, cell::RefCell};

#[derive(Debug, Clone, Copy)]
pub struct LnBackwardV<'g, S> {
    grad: &'g RefCell<S>,
    data: S,
}

impl<S, S2> Backward<S> for LnBackwardV<'_, S2>
where
    S2: Default + Pow<i32> + Add<<S as Mul<<S2 as Pow<i32>>::Output>>::Output, Output = S2>,
    S: Mul<<S2 as Pow<i32>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad + res_grad * self.data.pow(-1));
    }
}

impl<'g, S> Ln for &'g Variable<S>
where
    S: Clone + Ln,
{
    type Output = Tensor<<S as Ln>::Output, LnBackwardV<'g, S>>;
    fn ln(self) -> Self::Output {
        Tensor {
            data: self.data().clone().ln(),
            grad_fn: LnBackwardV {
                grad: &self.grad,
                data: self.data().clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LnBackwardT<S, F> {
    grad_fn: F,
    data: S,
}

impl<S, S2, F> Backward<S> for LnBackwardT<S2, F>
where
    S2: Pow<i32>,
    S: Mul<<S2 as Pow<i32>>::Output>,
    F: Backward<<S as Mul<<S2 as Pow<i32>>::Output>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad * self.data.pow(-1));
    }
}

impl<S, F> Ln for Tensor<S, F>
where
    S: Clone + Ln,
{
    type Output = Tensor<<S as Ln>::Output, LnBackwardT<S, F>>;
    fn ln(self) -> Self::Output {
        Tensor {
            data: self.data.clone().ln(),
            grad_fn: LnBackwardT {
                grad_fn: self.grad_fn,
                data: self.data,
            }
        }
    }
}
