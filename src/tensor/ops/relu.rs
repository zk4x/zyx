use crate::{ops::{ReLU, DReLU}, tensor::{Variable, Tensor, Backward, GradientRef}};
use std::ops::{Add, Mul};

#[derive(Debug, Clone, Copy)]
pub struct ReLUBackwardV<'g, S, G> {
    grad: GradientRef<'g, G>,
    data: S,
}

impl<S, S2, G> Backward<S> for ReLUBackwardV<'_, S2, G>
where
    S2: DReLU,
    S: Mul<<S2 as DReLU>::Output, Output = G>,
    G: Add<G, Output = G>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad * self.data.drelu());
    }
}

impl<'g, S, G> ReLU for &'g Variable<S, G>
where
    S: Clone + ReLU,
{
    type Output = Tensor<<S as ReLU>::Output, ReLUBackwardV<'g, S, G>>;
    fn relu(self) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().relu(),
            grad_fn: ReLUBackwardV {
                grad: GradientRef::new(&self.grad),
                data: (*self.data()).clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReLUBackwardT<S, F> {
    grad_fn: F,
    data: S,
}

impl<S, S2, F> Backward<S> for ReLUBackwardT<S2, F>
where
    S2: DReLU,
    S: Mul<<S2 as DReLU>::Output>,
    F: Backward<<S as Mul<<S2 as DReLU>::Output>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad * self.data.drelu());
    }
}

impl<S, F> ReLU for Tensor<S, F>
where
    S: Clone + ReLU,
{
    type Output = Tensor<<S as ReLU>::Output, ReLUBackwardT<S, F>>;
    fn relu(self) -> Self::Output {
        Tensor {
            data: self.data.clone().relu(),
            grad_fn: ReLUBackwardT {
                grad_fn: self.grad_fn,
                data: self.data,
            },
        }
    }
}
