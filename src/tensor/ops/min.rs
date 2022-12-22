// TODO: This is not correct. Minimizable does not simply expand. It takes

/*use crate::{ops::{Minimizable, Expand, HasShape}, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, shape::Shape};

#[derive(Debug, Clone)]
pub struct MinimizableBackwardV<'g, G> {
    grad: GradientRef<'g, G>,
    shape: Shape,
}

impl<S, G> Backward<S> for MinimizableBackwardV<'_, G>
where
    S: Expand,
    G: GradAcc<<S as Expand>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad.expand(self.shape));
    }
}

impl<'g, S> Minimizable for &'g Variable<S>
where
    S: Clone + Minimizable + HasShape,
{
    type Output = Tensor<<S as Minimizable>::Output, MinimizableBackwardV<'g, S>>;
    fn min(self, dims: impl Shape<i32>) -> Self::Output {
        Tensor {
            data: self.data.clone().min(dims),
            grad_fn: MinimizableBackwardV {
                grad: GradientRef::new(&self.grad),
                shape: self.data.shape(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MinimizableBackwardT<F> {
    grad_fn: F,
    shape: Shape,
}

impl<S, F> Backward<S> for MinimizableBackwardT<F>
where
    S: Expand,
    F: Backward<<S as Expand>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.expand(self.shape));
    }
}

impl<S, F> Minimizable for Tensor<S, F>
where
    S: Minimizable + HasShape,
    F: Backward<S>,
{
    type Output = Tensor<<S as Minimizable>::Output, MinimizableBackwardT<F>>;
    fn min(self, dims: impl Shape<i32>) -> Self::Output {
        let shape = self.data.shape();
        Tensor {
            data: self.data.min(dims),
            grad_fn: MinimizableBackwardT {
                grad_fn: self.grad_fn,
                shape,
            }
        }
    }
}*/
