use core::marker::PhantomData;

use crate::{
    ops::{HasShape, Reshapable},
    shape::Shape,
    tensor::{Backward, GradAcc, GradientRef, Tensor, Variable},
};

#[derive(Debug, Clone)]
pub struct ReshapeBackwardV<'g, G> {
    grad: GradientRef<'g, G>,
}

impl<B, G> Backward<B> for ReshapeBackwardV<'_, G>
where
    B: Reshapable<G::S>,
    G: HasShape + GradAcc<<B as Reshapable<G::S>>::Output>,
{
    fn backward(self, res_grad: B) {
        self.grad.accumulate(res_grad._reshape());
    }
}

impl<'g, S, Sh> Reshapable<Sh> for &'g Variable<S>
where
    Sh: Shape,
    S: Clone + Reshapable<Sh> + HasShape,
{
    type Output = Tensor<<S as Reshapable<Sh>>::Output, ReshapeBackwardV<'g, S>>;
    fn _reshape(self) -> Self::Output {
        Tensor {
            data: (*self.data()).clone()._reshape(),
            grad_fn: ReshapeBackwardV {
                grad: GradientRef::new(&self.grad),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReshapeBackwardT<F, Sh> {
    grad_fn: F,
    shape: PhantomData<Sh>,
}

impl<S, F, Sh> Backward<S> for ReshapeBackwardT<F, Sh>
where
    Sh: Shape,
    S: Reshapable<Sh>,
    F: Backward<<S as Reshapable<Sh>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad._reshape());
    }
}

impl<B, F, S> Reshapable<S> for Tensor<B, F>
where
    S: Shape,
    B: Reshapable<S> + HasShape,
{
    type Output = Tensor<<B as Reshapable<S>>::Output, ReshapeBackwardT<F, <B as HasShape>::S>>;
    fn _reshape(self) -> Self::Output {
        Tensor {
            data: self.data._reshape(),
            grad_fn: ReshapeBackwardT {
                grad_fn: self.grad_fn,
                shape: PhantomData,
            },
        }
    }
}
