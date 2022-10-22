use crate::tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake};
use std::{ops::{Sub, Neg}, cell::RefCell};

#[derive(Debug, Clone, Copy)]
pub struct NegBackwardV<'g, S> {
    grad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for NegBackwardV<'g, S>
where
    S: Default + Neg<Output = S> + Sub<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad - res_grad);
    }
}

impl<'g, S> Neg for &'g Variable<S>
where
    S: 'g + Clone + Neg<Output = S>,
{
    type Output = Tensor<S, NegBackwardV<'g, S>>;
    fn neg(self) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().neg(),
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
    S: Neg<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(-res_grad);
    }
}

impl<S, F> Neg for Tensor<S, F>
where
    S: Neg<Output = S>,
{
    type Output = Tensor<S, NegBackwardT<F>>;
    fn neg(self) -> Self::Output {
        Tensor {
            data: self.data.neg(),
            grad_fn: NegBackwardT {
                grad_fn: self.grad_fn,
            },
        }
    }
}
