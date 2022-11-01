use crate::{ops::{Tanh, Pow}, tensor::{Variable, Tensor, Backward, Gradient}};
use std::{ops::{Add, Mul, Neg}, cell::RefCell};

#[derive(Debug, Clone, Copy)]
pub struct TanhBackwardV<'g, S, S2> {
    grad: &'g Gradient<S>,
    res: S2,
}

impl<S, S2, S3> Backward<S> for TanhBackwardV<'_, S2, S3>
where
    S2: Default + Add<<S as Mul<<<<S3 as Pow<i32>>::Output as Neg>::Output as Add<i32>>::Output>>::Output, Output = S2>,
    S3: Pow<i32>,
    <S3 as Pow<i32>>::Output: Neg,
    <<S3 as Pow<i32>>::Output as Neg>::Output: Add<i32>,
    S: Mul<<<<S3 as Pow<i32>>::Output as Neg>::Output as Add<i32>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad * (-self.res.pow(2) + 1));
    }
}

impl<'g, S> Tanh for &'g Variable<S>
where
    S: Clone + Tanh,
    <S as Tanh>::Output: Clone,
{
    type Output = Tensor<<S as Tanh>::Output, TanhBackwardV<'g, S, <S as Tanh>::Output>>;
    fn tanh(self) -> Self::Output {
        let res = self.data().clone().tanh();
        Tensor {
            data: res.clone(),
            grad_fn: TanhBackwardV {
                grad: &self.grad,
                res,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TanhBackwardT<S, F> {
    grad_fn: F,
    res: S,
}

impl<S, S2, F> Backward<S> for TanhBackwardT<S2, F>
where
    S2: Pow<i32> + Neg,
    <S2 as Pow<i32>>::Output: Neg,
    <<S2 as Pow<i32>>::Output as Neg>::Output: Add<i32>,
    S: Mul<<<<S2 as Pow<i32>>::Output as Neg>::Output as Add<i32>>::Output>,
    F: Backward<<S as Mul<<<<S2 as Pow<i32>>::Output as Neg>::Output as Add<i32>>::Output>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad * (-self.res.pow(2) + 1));
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