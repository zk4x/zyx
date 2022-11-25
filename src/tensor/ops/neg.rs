use crate::tensor::{Variable, Tensor, Backward, GradientRef, GradAcc};
use core::ops::Neg;

#[derive(Debug, Clone, Copy)]
pub struct NegBackwardV<'g, G> {
    grad: GradientRef<'g, G>,
}

impl<S, G> Backward<S> for NegBackwardV<'_, G>
where
    S: Neg,
    G: GradAcc<<S as Neg>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(-res_grad);
    }
}

impl<'g, S> Neg for &'g Variable<S>
where
    S: Clone + Neg,
{
    type Output = Tensor<<S as Neg>::Output, NegBackwardV<'g, S>>;
    fn neg(self) -> Self::Output {
        Tensor {
            data: self.data.clone().neg(),
            grad_fn: NegBackwardV {
                grad: GradientRef::new(&self.grad),
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
