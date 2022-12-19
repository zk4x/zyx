use core::marker::PhantomData;

use crate::{ops::{Sum, Expand, HasShape}, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, shape::{Shape, Axes}};

#[derive(Debug, Clone)]
pub struct SumBackwardV<'g, G> {
    grad: GradientRef<'g, G>,
}

impl<S, G> Backward<S> for SumBackwardV<'_, G>
where
    S: Expand<G::Sh>,
    G: HasShape,
    //G: GradAcc<<S as Expand<G::Sh>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad.expand());
    }
}

impl<'g, S, Dims> Sum<Dims> for &'g Variable<S>
where
    S: Clone + Sum<Dims> + HasShape,
    Dims: Axes,
{
    type Output = Tensor<<S as Sum<Dims>>::Output, SumBackwardV<'g, S>>;
    fn sum(self) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().sum(),
            grad_fn: SumBackwardV {
                grad: GradientRef::new(&self.grad),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SumBackwardT<F, Sh> {
    grad_fn: F,
    shape: PhantomData<Sh>,
}

impl<S, F, Sh> Backward<S> for SumBackwardT<F, Sh>
where
    Sh: Shape,
    S: Expand<Sh>,
    F: Backward<<S as Expand<Sh>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.expand());
    }
}

impl<S, F, Dims> Sum<Dims> for Tensor<S, F>
where
    S: Clone + Sum<Dims> + HasShape,
    Dims: Axes,
{
    type Output = Tensor<<S as Sum<Dims>>::Output, SumBackwardT<F, <S as HasShape>::Sh>>;
    fn sum(self, dims: Dims) -> Self::Output {
        let shape = self.data.shape();
        Tensor {
            data: self.data.sum(),
            grad_fn: SumBackwardT {
                grad_fn: self.grad_fn,
                shape: PhantomData,
            }
        }
    }
}
