/*use crate::{ops::{Maximizable, Expand, HasShape}, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, shape::Shape};

#[derive(Debug, Clone)]
pub struct MaxBackwardV<'g, G, I> {
    grad: GradientRef<'g, G>,
    indices: I,
}

impl<S, G, I> Backward<S> for MaxBackwardV<'_, G, I>
where
    //S: ,
    //G: GradAcc<<S as >::Output>,
{
    fn backward(self, res_grad: S) {
        // TODO: This is not correct. Maximizable does not simply expand.
        // Maximizable sets values at max indices to 1 and other values to 0.
        // So res_grad values must be added to indices where there were maximums previously.
        // So Instead of shape, we need to store indices of those values.
        self.grad.accumulate(res_grad.set_values_at_indices(self.indices));
    }
}

impl<'g, S, Dims> Maximizable<Dims> for &'g Variable<S>
where
    S: Clone + Maximizable<Dims>,
{
    type Values = Tensor<S::Values, MaxBackwardV<'g, S, Self::Indices>>;
    type Indices = S::Indices;

    fn max(self) -> (Self::Values, Self::Indices) {
        let (data, indices) = self.data.clone().max();
        (Tensor {
            data,
            grad_fn: MaxBackwardV {
                grad: GradientRef::new(&self.grad),
                indices: indices.clone(),
            }
        }, indices)
    }
}

#[derive(Debug, Clone)]
pub struct MaxBackwardT<F, I> {
    grad_fn: F,
    indices: I,
}

impl<S, F> Backward<S> for MaxBackwardT<F, I>
where
    //S: Expand,
    //F: Backward<<S as Expand>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.expand(self.shape));
    }
}

impl<S, F> Maximizable for Tensor<S, F>
where
    S: Maximizable + HasShape,
{
    type Output = Tensor<<S as Maximizable>::Output, MaxBackwardT<F>>;
    fn max(self, dims: impl Shape<i32>) -> Self::Output {
        let shape = self.data.shape();
        Tensor {
            data: self.data.max(dims),
            grad_fn: MaxBackwardT {
                grad_fn: self.grad_fn,
                shape,
            },
        }
    }
}*/
