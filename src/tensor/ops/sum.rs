use crate::{ops::{Sum, Expand, GetShape}, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, shape::{Shape, IntoDims}};

#[derive(Debug, Clone)]
pub struct SumBackwardV<'g, G> {
    grad: GradientRef<'g, G>,
    shape: Shape,
}

impl<S, G> Backward<S> for SumBackwardV<'_, G>
where
    S: Expand,
    G: GradAcc<<S as Expand>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad.expand(self.shape));
    }
}

impl<'g, S> Sum for &'g Variable<S>
where
    S: Clone + Sum + GetShape,
{
    type Output = Tensor<<S as Sum>::Output, SumBackwardV<'g, S>>;
    fn sum(self, dims: impl IntoDims) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().sum(dims),
            grad_fn: SumBackwardV {
                grad: GradientRef::new(&self.grad),
                shape: self.data().shape(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SumBackwardT<F> {
    grad_fn: F,
    shape: Shape,
}

impl<S, F> Backward<S> for SumBackwardT<F>
where
    S: Expand,
    F: Backward<<S as Expand>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.expand(self.shape));
    }
}

impl<S, F> Sum for Tensor<S, F>
where
    S: Clone + Sum + GetShape,
{
    type Output = Tensor<<S as Sum>::Output, SumBackwardT<F>>;
    fn sum(self, dims: impl IntoDims) -> Self::Output {
        let shape = self.data.shape();
        Tensor {
            data: self.data.sum(dims),
            grad_fn: SumBackwardT {
                grad_fn: self.grad_fn,
                shape,
            }
        }
    }
}
