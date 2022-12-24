use core::marker::PhantomData;

use crate::{ops::Permutable, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, shape::{Axes, Argsortable}};

#[derive(Debug, Clone)]
pub struct PermutableBackwardV<'g, G, Dims> {
    grad: GradientRef<'g, G>,
    dims: PhantomData<Dims>,
}

impl<S, G, Dims> Backward<S> for PermutableBackwardV<'_, G, Dims>
where
    Dims: Axes + Argsortable,
    S: Permutable<Dims::Argsort>,
    G: GradAcc<<S as Permutable<Dims::Argsort>>::Output>,
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
    type Output = Tensor<<S as Permutable<Dims>>::Output, PermutableBackwardV<'g, S, Dims>>;
    fn _permute(self) -> Self::Output {
        Tensor {
            data: self.data.clone()._permute(),
            grad_fn: PermutableBackwardV {
                grad: GradientRef::new(&self.grad),
                dims: PhantomData,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PermutableBackwardT<F, Dims> {
    grad_fn: F,
    dims: PhantomData<Dims>,
}

impl<S, F, Dims> Backward<S> for PermutableBackwardT<F, Dims>
where
    Dims: Axes + Argsortable,
    S: Permutable<Dims::Argsort>,
    F: Backward<<S as Permutable<Dims::Argsort>>::Output>,
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
    type Output = Tensor<<S as Permutable<Dims>>::Output, PermutableBackwardT<F, Dims>>;
    fn _permute(self) -> Self::Output {
        Tensor {
            data: self.data._permute(),
            grad_fn: PermutableBackwardT {
                grad_fn: self.grad_fn,
                dims: PhantomData,
            }
        }
    }
}
