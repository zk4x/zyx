use crate::{ops::{Reshape, GetShape}, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, shape::Shape};

#[derive(Debug, Clone)]
pub struct ReshapeBackwardV<'g, G, Sh> {
    grad: GradientRef<'g, G>,
    shape: Sh,
}

impl<S, G, Sh> Backward<S> for ReshapeBackwardV<'_, G, Sh>
where
    Sh: Shape<D = usize>,
    S: Reshape<Sh>,
    G: GradAcc<<S as Reshape<Sh>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad.reshape(self.shape));
    }
}

impl<'g, S, Sh> Reshape<Sh> for &'g Variable<S>
where
    Sh: Shape<D = usize>,
    S: Clone + Reshape<Sh> + GetShape,
{
    type Output = Tensor<<S as Reshape<Sh>>::Output, ReshapeBackwardV<'g, S, <S as GetShape>::Output>>;
    fn reshape(self, shape: Sh) -> Self::Output {
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
pub struct ReshapeBackwardT<F, Sh> {
    grad_fn: F,
    shape: Sh,
}

impl<S, F, Sh> Backward<S> for ReshapeBackwardT<F, Sh>
where
    Sh: Shape<D = usize>,
    S: Reshape<Sh>,
    F: Backward<<S as Reshape<Sh>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.reshape(self.shape));
    }
}

impl<S, F, Sh> Reshape<Sh> for Tensor<S, F>
where
    Sh: Shape<D = usize>,
    S: Reshape<Sh> + GetShape,
{
    type Output = Tensor<<S as Reshape<Sh>>::Output, ReshapeBackwardT<F, <S as GetShape>::Output>>;
    fn reshape(self, res_shape: Sh) -> Self::Output {
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
