use crate::{ops::{Sum, Expand, GetShape}, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, shape::Shape};

#[derive(Debug, Clone)]
pub struct SumBackwardV<'g, G, Sh> {
    grad: GradientRef<'g, G>,
    shape: Sh,
}

impl<S, G, Sh> Backward<S> for SumBackwardV<'_, G, Sh>
where
    S: Expand<Sh>,
    G: GradAcc<<S as Expand<Sh>>::Output>,
    Sh: Shape<D = usize>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad.expand(self.shape));
    }
}

impl<'g, S, Dims> Sum<Dims> for &'g Variable<S>
where
    S: Clone + Sum<Dims> + GetShape,
    Dims: Shape<D = i32>,
{
    type Output = Tensor<<S as Sum<Dims>>::Output, SumBackwardV<'g, S, <S as GetShape>::Output>>;
    fn sum(self, dims: Dims) -> Self::Output {
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
pub struct SumBackwardT<F, Sh> {
    grad_fn: F,
    shape: Sh,
}

impl<S, F, Sh> Backward<S> for SumBackwardT<F, Sh>
where
    Sh: Shape<D = usize>,
    S: Expand<Sh>,
    F: Backward<<S as Expand<Sh>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.expand(self.shape));
    }
}

impl<S, F, Dims> Sum<Dims> for Tensor<S, F>
where
    S: Clone + Sum<Dims> + GetShape,
    Dims: Shape<D = i32>,
{
    type Output = Tensor<<S as Sum<Dims>>::Output, SumBackwardT<F, <S as GetShape>::Output>>;
    fn sum(self, dims: Dims) -> Self::Output {
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
