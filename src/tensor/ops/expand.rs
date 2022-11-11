use crate::{ops::{Expand, Max, GetShape}, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, shape::{IntoShape, Dims}};

#[derive(Debug, Clone)]
pub struct ExpandBackwardV<'g, G> {
    grad: GradientRef<'g, G>,
    dims: Dims,
}

impl<S, G> Backward<S> for ExpandBackwardV<'_, G>
where
    S: Max,
    G: GradAcc<<S as Max>::Output>,
{
    fn backward(self, res_grad: S) {
        // TODO: is max correct reduce for expand backward?
        self.grad.accumulate(res_grad.max(self.dims));
    }
}

impl<'g, S> Expand for &'g Variable<S>
where
    S: Clone + Expand + GetShape,
{
    type Output = Tensor<<S as Expand>::Output, ExpandBackwardV<'g, S>>;
    fn expand(self, shape: impl IntoShape) -> Self::Output {
        let shape = shape.shape();
        let dims = Dims(self.data().shape().into_iter().zip(shape.clone().into_iter()).enumerate().filter_map(|(i, (a, b))| if a != b { Some(i as i32) } else { None }).collect());
        Tensor {
            data: (*self.data()).clone().expand(shape),
            grad_fn: ExpandBackwardV {
                grad: GradientRef::new(&self.grad),
                dims,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExpandBackwardT<F> {
    grad_fn: F,
    dims: Dims,
}

impl<S, F> Backward<S> for ExpandBackwardT<F>
where
    S: Max,
    F: Backward<<S as Max>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.max(self.dims));
    }
}

impl<S, F> Expand for Tensor<S, F>
where
    S: Expand + GetShape,
    F: Backward<S>,
{
    type Output = Tensor<<S as Expand>::Output, ExpandBackwardT<F>>;
    fn expand(self, shape: impl IntoShape) -> Self::Output {
        let shape = shape.shape();
        let dims = Dims(self.data.shape().into_iter().zip(shape.clone().into_iter()).enumerate().filter_map(|(i, (a, b))| if a != b { Some(i as i32) } else { None }).collect());
        Tensor {
            data: self.data.expand(shape),
            grad_fn: ExpandBackwardT {
                grad_fn: self.grad_fn,
                dims,
            },
        }
    }
}
