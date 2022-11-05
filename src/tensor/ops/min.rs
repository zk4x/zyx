use crate::{ops::{Min, Expand, GetShape}, tensor::{Variable, Tensor, Backward, GradientRef}, shape::{IntoDims, Shape}};
use std::ops::Add;

#[derive(Debug, Clone)]
pub struct MinBackwardV<'g, G> {
    grad: GradientRef<'g, G>,
    shape: Shape,
}

impl<S> Backward<S> for MinBackwardV<'_, S>
where
    S: Default + Add<Output = S> + Expand<Output = S> + GetShape,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad.expand(self.shape));
    }
}

impl<'g, S, G> Min for &'g Variable<S, G>
where
    S: 'g + Clone + Min<Output = S> + GetShape,
{
    type Output = Tensor<S, MinBackwardV<'g, G>>;
    fn min(self, dims: impl IntoDims) -> Self::Output {
        Tensor {
            data: self.data.clone().min(dims),
            grad_fn: MinBackwardV {
                grad: GradientRef::new(&self.grad),
                shape: self.data.shape(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MinBackwardT<F> {
    grad_fn: F,
    shape: Shape,
}

impl<S, F> Backward<S> for MinBackwardT<F>
where
    S: Expand<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.expand(self.shape));
    }
}

impl<S, F> Min for Tensor<S, F>
where
    S: Min<Output = S> + GetShape,
    F: Backward<S>,
{
    type Output = Tensor<S, MinBackwardT<F>>;
    fn min(self, dims: impl IntoDims) -> Self::Output {
        let shape = self.data.shape();
        Tensor {
            data: self.data.min(dims),
            grad_fn: MinBackwardT {
                grad_fn: self.grad_fn,
                shape,
            }
        }
    }
}
