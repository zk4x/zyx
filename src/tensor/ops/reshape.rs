use core::marker::PhantomData;

use crate::{ops::{Reshapable, HasShape}, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, shape::Shape};

#[derive(Debug, Clone)]
pub struct ReshapableBackwardV<'g, G> {
    grad: GradientRef<'g, G>,
}

impl<S, G> Backward<S> for ReshapableBackwardV<'_, G>
where
    S: Reshapable<G::Sh>,
    G: HasShape + GradAcc<<S as Reshapable<G::Sh>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad.reshape());
    }
}

impl<'g, S, Sh> Reshapable<Sh> for &'g Variable<S>
where
    Sh: Shape,
    S: Clone + Reshapable<Sh> + HasShape,
{
    type Output = Tensor<<S as Reshapable<Sh>>::Output, ReshapableBackwardV<'g, S>>;
    fn reshape(self) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().reshape(),
            grad_fn: ReshapableBackwardV {
                grad: GradientRef::new(&self.grad),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReshapableBackwardT<F, Sh> {
    grad_fn: F,
    shape: PhantomData<Sh>,
}

impl<S, F, Sh> Backward<S> for ReshapableBackwardT<F, Sh>
where
    Sh: Shape,
    S: Reshapable<Sh>,
    F: Backward<<S as Reshapable<Sh>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.reshape());
    }
}

impl<S, F, Sh> Reshapable<Sh> for Tensor<S, F>
where
    Sh: Shape,
    S: Reshapable<Sh> + HasShape,
{
    type Output = Tensor<<S as Reshapable<Sh>>::Output, ReshapableBackwardT<F, <S as HasShape>::Sh>>;
    fn reshape(self) -> Self::Output {
        Tensor {
            data: self.data.reshape(),
            grad_fn: ReshapableBackwardT {
                grad_fn: self.grad_fn,
                shape: PhantomData,
            }
        }
    }
}
