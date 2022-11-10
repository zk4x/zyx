use crate::{ops::Exp, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}};
use core::ops::Mul;

#[derive(Debug, Clone, Copy)]
pub struct ExpBackwardV<'g, S2, G> {
    res: S2,
    grad: GradientRef<'g, G>,
}

impl<S, S2, G> Backward<S> for ExpBackwardV<'_, S2, G>
where
    S: Mul<S2, Output = G>,
    G: GradAcc<<S as Mul<S2>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad * self.res);
    }
}

impl<'g, S> Exp for &'g Variable<S>
where
    S: Clone + Exp,
    <S as Exp>::Output: Clone,
{
    type Output = Tensor<<S as Exp>::Output, ExpBackwardV<'g, <S as Exp>::Output, S>>;
    fn exp(self) -> Self::Output {
        let res = (*self.data()).clone().exp();
        Tensor {
            data: res.clone(),
            grad_fn: ExpBackwardV {
                grad: GradientRef::new(&self.grad),
                res,
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