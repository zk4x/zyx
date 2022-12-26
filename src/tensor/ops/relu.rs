use crate::{ops::{ReLU, DReLU}, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}};
use core::ops::Mul;

#[derive(Debug, Clone, Copy)]
pub struct ReLUBackwardV<'g, S, G> {
    grad: GradientRef<'g, G>,
    data: S,
}

impl<S, S2, G> Backward<S> for ReLUBackwardV<'_, S2, G>
where
    S2: DReLU,
    <S2 as DReLU>::Output: Mul<S>,
    G: GradAcc<<<S2 as DReLU>::Output as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(self.data.drelu() * res_grad);
    }
}

impl<'g, S> ReLU for &'g Variable<S>
where
    S: Clone + ReLU,
{
    type Output = Tensor<<S as ReLU>::Output, ReLUBackwardV<'g, S, S>>;
    fn relu(self) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().relu(),
            grad_fn: ReLUBackwardV {
                grad: GradientRef::new(&self.grad),
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
    <S2 as DReLU>::Output: Mul<S>,
    F: Backward<<<S2 as DReLU>::Output as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(self.data.drelu() * res_grad);
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
