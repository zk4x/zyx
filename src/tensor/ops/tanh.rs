use crate::{ops::{Tanh, Pow}, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}};
use core::ops::{Sub, Mul};

#[derive(Debug, Clone, Copy)]
pub struct TanhBackwardV<'g, S2, G> {
    grad: GradientRef<'g, G>,
    res: S2,
}

impl<S, S2, G> Backward<S> for TanhBackwardV<'_, S2, G>
where
    S2: Pow<i32>,
    i32: Sub<<S2 as Pow<i32>>::Output>,
    <i32 as Sub<<S2 as Pow<i32>>::Output>>::Output: Mul<S>,
    G: GradAcc<<<i32 as Sub<<S2 as Pow<i32>>::Output>>::Output as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate((1 - self.res.pow(2)) * res_grad);
    }
}

impl<'g, S> Tanh for &'g Variable<S>
where
    S: Clone + Tanh,
    <S as Tanh>::Output: Clone,
{
    type Output = Tensor<<S as Tanh>::Output, TanhBackwardV<'g, <S as Tanh>::Output, S>>;
    fn tanh(self) -> Self::Output {
        let res = self.data.clone().tanh();
        Tensor {
            data: res.clone(),
            grad_fn: TanhBackwardV {
                grad: GradientRef::new(&self.grad),
                res,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TanhBackwardT<S2, F> {
    grad_fn: F,
    res: S2,
}

impl<S, S2, F> Backward<S> for TanhBackwardT<S2, F>
where
    S2: Pow<i32>,
    i32: Sub<<S2 as Pow<i32>>::Output>,
    <i32 as Sub<<S2 as Pow<i32>>::Output>>::Output: Mul<S>,
    F: Backward<<<i32 as Sub<<S2 as Pow<i32>>::Output>>::Output as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward((1 - self.res.pow(2)) * res_grad);
    }
}

impl<S, F> Tanh for Tensor<S, F>
where
    S: Tanh,
    <S as Tanh>::Output: Clone,
{
    type Output = Tensor<<S as Tanh>::Output, TanhBackwardT<<S as Tanh>::Output, F>>;
    fn tanh(self) -> Self::Output {
        let res = self.data.tanh();
        Tensor {
            data: res.clone(),
            grad_fn: TanhBackwardT {
                grad_fn: self.grad_fn,
                res,
            },
        }
    }
}