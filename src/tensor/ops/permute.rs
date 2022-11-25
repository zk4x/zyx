use crate::{ops::Permute, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, shape::Shape};

#[derive(Debug, Clone)]
pub struct PermuteBackwardV<'g, G, Dims> {
    grad: GradientRef<'g, G>,
    dims: Dims,
}

impl<S, G, Dims> Backward<S> for PermuteBackwardV<'_, G, Dims>
where
    Dims: Shape<D = i32>,
    S: Permute<Dims>,
    G: GradAcc<<S as Permute<Dims>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad.permute(self.dims));
    }
}

impl<'g, S, Dims> Permute<Dims> for &'g Variable<S>
where
    Dims: Shape<D = i32>,
    S: Clone + Permute<Dims>,
{
    type Output = Tensor<<S as Permute<Dims>>::Output, PermuteBackwardV<'g, S, Dims>>;
    fn permute(self, dims: Dims) -> Self::Output {
        Tensor {
            data: self.data.clone().permute(dims),
            grad_fn: PermuteBackwardV {
                grad: GradientRef::new(&self.grad),
                dims: dims.argsort(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PermuteBackwardT<F, Dims> {
    grad_fn: F,
    dims: Dims,
}

impl<S, F, Dims> Backward<S> for PermuteBackwardT<F, Dims>
where
    Dims: Shape<D = i32>,
    S: Permute<Dims>,
    F: Backward<<S as Permute<Dims>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.permute(self.dims));
    }
}

impl<S, F, Dims> Permute<Dims> for Tensor<S, F>
where
    Dims: Shape<D = i32>,
    S: Permute<Dims>,
{
    type Output = Tensor<<S as Permute<Dims>>::Output, PermuteBackwardT<F, Dims>>;
    fn permute(self, dims: Dims) -> Self::Output {
        Tensor {
            data: self.data.permute(dims),
            grad_fn: PermuteBackwardT {
                grad_fn: self.grad_fn,
                dims: dims.argsort(),
            }
        }
    }
}
