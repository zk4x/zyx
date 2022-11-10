use crate::{ops::{Reshape, GetShape}, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, shape::{IntoShape, Shape}};

#[derive(Debug, Clone)]
pub struct ReshapeBackwardV<'g, G> {
    grad: GradientRef<'g, G>,
    shape: Shape,
}

impl<S, G> Backward<S> for ReshapeBackwardV<'_, G>
where
    S: Reshape,
    G: GradAcc<<S as Reshape>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad.reshape(self.shape));
    }
}

impl<'g, S> Reshape for &'g Variable<S>
where
    S: Clone + Reshape + GetShape,
{
    type Output = Tensor<<S as Reshape>::Output, ReshapeBackwardV<'g, S>>;
    fn reshape(self, shape: impl IntoShape) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().reshape(shape),
            grad_fn: ReshapeBackwardV {
                grad: GradientRef::new(&self.grad),
                shape: self.data().shape(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReshapeBackwardT<F> {
    grad_fn: F,
    shape: Shape,
}

impl<S, F> Backward<S> for ReshapeBackwardT<F>
where
    S: Reshape,
    F: Backward<<S as Reshape>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.reshape(self.shape));
    }
}

impl<S, F> Reshape for Tensor<S, F>
where
    S: Reshape + GetShape,
{
    type Output = Tensor<<S as Reshape>::Output, ReshapeBackwardT<F>>;
    fn reshape(self, res_shape: impl IntoShape) -> Self::Output {
        let shape = self.data.shape();
        Tensor {
            data: self.data.reshape(res_shape),
            grad_fn: ReshapeBackwardT {
                grad_fn: self.grad_fn,
                shape,
            }
        }
    }
}