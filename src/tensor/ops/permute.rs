use core::marker::PhantomData;

use crate::{ops::Permute, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, shape::Axes};

#[derive(Debug, Clone)]
pub struct PermuteBackwardV<'g, G, Dims> {
    grad: GradientRef<'g, G>,
    dims: PhantomData<Dims>,
}

impl<S, G, Dims> Backward<S> for PermuteBackwardV<'_, G, Dims>
where
    Dims: Axes,
    S: Permute<Dims::Argsort>,
    G: GradAcc<<S as Permute<Dims::Argsort>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad.permute());
    }
}

impl<'g, S, Dims> Permute<Dims> for &'g Variable<S>
where
    Dims: Axes,
    S: Clone + Permute<Dims>,
{
    type Output = Tensor<<S as Permute<Dims>>::Output, PermuteBackwardV<'g, S, Dims>>;
    fn permute(self, dims: Dims) -> Self::Output {
        Tensor {
            data: self.data.clone().permute(),
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
    Dims: Axes,
    S: Permute<Dims::Argsort>,
    F: Backward<<S as Permute<Dims::Argsort>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.permute());
    }
}

impl<S, F, Dims> Permute<Dims> for Tensor<S, F>
where
    Dims: Axes,
    S: Permute<Dims>,
{
    type Output = Tensor<<S as Permute<Dims>>::Output, PermuteBackwardT<F, Dims>>;
    fn permute(self) -> Self::Output {
        Tensor {
            data: self.data.permute(),
            grad_fn: PermuteBackwardT {
                grad_fn: self.grad_fn,
                dims: PhantomData,
            }
        }
    }
}
