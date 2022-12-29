use core::marker::PhantomData;

use crate::{ops::Permutable, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, shape::{Axes, PermutableBy}};

#[derive(Debug, Clone)]
pub struct PermuteBackwardV<'g, G, Dims> {
    grad: GradientRef<'g, G>,
    dims: PhantomData<Dims>,
}

impl<S, G, Dims> Backward<S> for PermuteBackwardV<'_, G, Dims>
where
    Dims: Axes + PermutableBy<Dims>,
    <Dims as PermutableBy<Dims>>::Output: Axes,
    S: Permutable<<Dims as PermutableBy<Dims>>::Output>,
    G: GradAcc<<S as Permutable<<Dims as PermutableBy<Dims>>::Output>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad._permute());
    }
}

impl<'g, S, Dims> Permutable<Dims> for &'g Variable<S>
where
    Dims: Axes,
    S: Clone + Permutable<Dims>,
{
    type Output = Tensor<<S as Permutable<Dims>>::Output, PermuteBackwardV<'g, S, Dims>>;
    fn _permute(self) -> Self::Output {
        Tensor {
            data: self.data.clone()._permute(),
            grad_fn: PermuteBackwardV {
                grad: GradientRef::new(&self.grad),
                dims: PhantomData,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PermuteBackwardT<F, Dims> {
    grad_fn: F,
    dims: PhantomData<Dims>,
}

impl<S, F, Dims> Backward<S> for PermuteBackwardT<F, Dims>
where
    Dims: Axes + PermutableBy<Dims>,
    <Dims as PermutableBy<Dims>>::Output: Axes,
    S: Permutable<<Dims as PermutableBy<Dims>>::Output>,
    F: Backward<<S as Permutable<<Dims as PermutableBy<Dims>>::Output>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad._permute());
    }
}

impl<S, F, Dims> Permutable<Dims> for Tensor<S, F>
where
    Dims: Axes,
    S: Permutable<Dims>,
{
    type Output = Tensor<<S as Permutable<Dims>>::Output, PermuteBackwardT<F, Dims>>;
    fn _permute(self) -> Self::Output {
        Tensor {
            data: self.data._permute(),
            grad_fn: PermuteBackwardT {
                grad_fn: self.grad_fn,
                dims: PhantomData,
            }
        }
    }
}
