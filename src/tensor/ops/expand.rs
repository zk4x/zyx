// TODO make this work

/*use crate::{ops::{Expand, Maximizable, HasShape}, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, shape::Shape};

#[derive(Debug, Clone)]
pub struct ExpandBackwardV<'g, G, Dims> {
    grad: GradientRef<'g, G>,
    dims: Dims,
}

impl<S, G, Dims> Backward<S> for ExpandBackwardV<'_, G, Dims>
where
    S: Maximizable<Dims>,
    G: GradAcc<<S as Maximizable<Dims>>::Output>,
{
    fn backward(self, res_grad: S) {
        // TODO: is max correct reduce for expand backward?
        self.grad.accumulate(res_grad.max(self.dims));
    }
}

impl<'g, S, Sh> Expand<Sh> for &'g Variable<S>
where
    S: Clone + Expand<Sh> + HasShape,
{
    type Output = Tensor<<S as Expand<Sh>>::Output, ExpandBackwardV<'g, S>>;
    fn expand(self, shape: Sh) -> Self::Output {
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
pub struct ExpandBackwardT<F, Dims> {
    grad_fn: F,
    dims: Dims,
}

impl<S, F> Backward<S> for ExpandBackwardT<F>
where
    S: Maximizable,
    F: Backward<<S as Maximizable>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad.max(self.dims));
    }
}

impl<S, F> Expand for Tensor<S, F>
where
    S: Expand + HasShape,
    F: Backward<S>,
{
    type Output = Tensor<<S as Expand>::Output, ExpandBackwardT<F>>;
    fn expand(self, shape: impl Shape<usize>) -> Self::Output {
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
}*/
