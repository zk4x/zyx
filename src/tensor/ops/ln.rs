use crate::{
    ops::{Ln, Pow},
    tensor::{Backward, GradAcc, GradientRef, Tensor, Variable},
};
use core::ops::Mul;

#[derive(Debug, Clone, Copy)]
pub struct LnBackwardV<'g, S, G> {
    grad: GradientRef<'g, G>,
    data: S,
}

impl<S, S2, G> Backward<S> for LnBackwardV<'_, S2, G>
where
    S2: Pow<i32>,
    <S2 as Pow<i32>>::Output: Mul<S>,
    G: GradAcc<<<S2 as Pow<i32>>::Output as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(self.data.pow(-1) * res_grad);
    }
}

impl<'g, S> Ln for &'g Variable<S>
where
    S: Clone + Ln,
{
    type Output = Tensor<<S as Ln>::Output, LnBackwardV<'g, S, S>>;
    fn ln(self) -> Self::Output {
        Tensor {
            data: self.data.clone().ln(),
            grad_fn: LnBackwardV {
                grad: GradientRef::new(&self.grad),
                data: self.data.clone(),
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
    <S2 as Pow<i32>>::Output: Mul<S>,
    F: Backward<<<S2 as Pow<i32>>::Output as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(self.data.pow(-1) * res_grad);
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
            },
        }
    }
}
