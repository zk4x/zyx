use crate::{ops::{Ln, Ones, Pow}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::{Add, Mul, Neg}, cell::RefCell};

#[derive(Debug, Clone, Copy)]
pub struct LnBackwardV<'g, S> {
    grad: &'g RefCell<S>,
    data: S,
}

impl<S> Backward<S> for LnBackwardV<'_, S>
where
    S: Default + Ln<Output = S> + Mul<Output = S> + Add<Output = S> + Pow<Output = S> + Neg<Output = S> + Ones,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad + res_grad * self.data.pow(-S::ones([1])));
    }
}

impl<'g, S> Ln for &'g Variable<S>
where
    S: 'g + Clone + Ln<Output = S>,
{
    type Output = Tensor<S, LnBackwardV<'g, S>>;
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

impl<S, F> Backward<S> for LnBackwardT<S, F>
where
    S: Ln<Output = S> + Mul<Output = S> + Pow<Output = S> + Neg<Output = S> + Ones,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad * self.data.pow(-S::ones([1])));
    }
}

impl<S, F> Ln for Tensor<S, F>
where
    S: Clone + Ln<Output = S>,
{
    type Output = Tensor<S, LnBackwardT<S, F>>;
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
