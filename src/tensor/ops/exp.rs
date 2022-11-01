use crate::{ops::Exp, tensor::{Variable, Tensor, Backward, Gradient}};
use std::{ops::{Add, Mul}, cell::RefCell};

#[derive(Debug, Clone, Copy)]
pub struct ExpBackwardV<'g, S, S2> {
    grad: &'g Gradient<S>,
    data: S2,
}

impl<S, S2> Backward<S> for ExpBackwardV<'_, S, S2>
where
    S: Mul<S2, Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad * self.data);
    }
}

impl<'g, S, S2> Exp for &'g Variable<S>
where
    S: Clone + Exp<Output = S2>,
    S2: Clone,
{
    type Output = Tensor<S2, ExpBackwardV<'g, S, S2>>;
    fn exp(self) -> Self::Output {
        let data = (*self.data()).clone().exp();
        Tensor {
            data: data.clone(),
            grad_fn: ExpBackwardV {
                grad: &self.grad,
                data,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ExpBackwardT<S2, F> {
    grad_fn: F,
    data: S2,
}

impl<S, S2, F> Backward<S> for ExpBackwardT<S2, F>
where
    S: Mul<S2>,
    F: Backward<<S as Mul<S2>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad * self.data);
    }
}

impl<S, S2, F> Exp for Tensor<S, F>
where
    S: Exp<Output = S2>,
    S2: Clone,
{
    type Output = Tensor<S2, ExpBackwardT<S2, F>>;
    fn exp(self) -> Self::Output {
        let data = self.data.exp();
        Tensor {
            data: data.clone(),
            grad_fn: ExpBackwardT {
                grad_fn: self.grad_fn,
                data,
            },
        }
    }
}