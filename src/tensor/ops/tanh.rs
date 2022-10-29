use crate::{ops::{Tanh, Ones, Pow}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::{Add, Mul, Neg}, cell::RefCell};

#[derive(Debug, Clone, Copy)]
pub struct TanhBackwardV<'g, S> {
    grad: &'g RefCell<S>,
    data: S,
}

impl<S> Backward<S> for TanhBackwardV<'_, S>
where
    S: Default + Tanh<Output = S> + Mul<Output = S> + Add<Output = S> + Pow<Output = S> + Neg<Output = S> + Ones,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad + res_grad * (-self.data.pow(S::ones([1]) + S::ones([1])) + S::ones([1])));
    }
}

impl<'g, S> Tanh for &'g Variable<S>
where
    S: 'g + Clone + Tanh<Output = S>,
{
    type Output = Tensor<S, TanhBackwardV<'g, S>>;
    fn tanh(self) -> Self::Output {
        let data = (*self.data()).clone().tanh();
        Tensor {
            data: data.clone(),
            grad_fn: TanhBackwardV {
                grad: &self.grad,
                data,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TanhBackwardT<S, F> {
    grad_fn: F,
    data: S,
}

impl<S, F> Backward<S> for TanhBackwardT<S, F>
where
    S: Tanh<Output = S> + Mul<Output = S> + Add<Output = S> + Pow<Output = S> + Neg<Output = S> + Ones,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad * (-self.data.pow(S::ones([1]) + S::ones([1])) + S::ones([1])));
    }
}

impl<S, F> Tanh for Tensor<S, F>
where
    S: Clone + Tanh<Output = S>,
{
    type Output = Tensor<S, TanhBackwardT<S, F>>;
    fn tanh(self) -> Self::Output {
        let data = self.data.tanh();
        Tensor {
            data: data.clone(),
            grad_fn: TanhBackwardT {
                grad_fn: self.grad_fn,
                data,
            },
        }
    }
}