use crate::tensor::{Variable, Tensor, Backward, Gradient};
use std::{ops::{Sub, Add, Neg}, cell::RefCell};

#[derive(Debug, Clone, Copy)]
pub struct NegBackwardV<'g, G> {
    grad: &'g Gradient<G>,
}

impl<S, S2> Backward<S> for NegBackwardV<'_, S2>
where
    S: Neg<Output = S2>,
    S2: Add<S2, Output = S2>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(-res_grad);
    }
}

impl<'g, S, G> Neg for &'g Variable<S, G>
where
    S: Clone + Neg,
{
    type Output = Tensor<<S as Neg>::Output, NegBackwardV<'g, S>>;
    fn neg(self) -> Self::Output {
        Tensor {
            data: self.data().clone().neg(),
            grad_fn: NegBackwardV {
                grad: &self.grad,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NegBackwardT<F> {
    grad_fn: F,
}

impl<S, F> Backward<S> for NegBackwardT<F>
where
    S: Neg,
    F: Backward<<S as Neg>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(-res_grad);
    }
}

impl<S, F> Neg for Tensor<S, F>
where
    S: Neg,
{
    type Output = Tensor<<S as Neg>::Output, NegBackwardT<F>>;
    fn neg(self) -> Self::Output {
        Tensor {
            data: self.data.neg(),
            grad_fn: NegBackwardT {
                grad_fn: self.grad_fn,
            },
        }
    }
}
