use crate::{ops::{Reshape, HasShape}, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, shape::Shape};

#[derive(Debug, Clone)]
pub struct ReshapeBackwardV<'g, G, Sh> {
    grad: GradientRef<'g, G>,
    shape: Sh,
}

impl<S, G, Sh> Backward<S> for ReshapeBackwardV<'_, G, Sh>
where
    Sh: Shape,
    S: Reshape<Sh>,
    G: GradAcc<<S as Reshape<Sh>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad.reshape());
    }
}

impl<'g, S, Sh> Reshape<Sh> for &'g Variable<S>
where
    Sh: Shape,
    S: Clone + Reshape<Sh> + HasShape,
{
    type Output = Tensor<<S as Reshape<Sh>>::Output, ReshapeBackwardV<'g, S, <S as HasShape>::Sh>>;
    fn reshape(self, shape: Sh) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().reshape(),
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
    Sh: Shape,
    S: Reshape<Sh>,
    F: Backward<<S as Reshape<Sh>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.reshape());
    }
}

impl<S, F, Sh> Reshape<Sh> for Tensor<S, F>
where
    Sh: Shape,
    S: Reshape<Sh> + HasShape,
{
    type Output = Tensor<<S as Reshape<Sh>>::Output, ReshapeBackwardT<F, <S as HasShape>::Sh>>;
    fn reshape(self, res_shape: Sh) -> Self::Output {
        let shape = self.data.shape();
        Tensor {
            data: self.data.reshape(),
            grad_fn: ReshapeBackwardT {
                grad_fn: self.grad_fn,
                shape,
            }
        }
    }
}
