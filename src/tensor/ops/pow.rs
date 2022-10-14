use crate::{ops::Pow, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{cell::RefCell, ops::Add};

#[derive(Debug, Clone, Copy)]
pub struct PowBackwardSV<'g, S> {
    ygrad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for PowBackwardSV<'g, S>
where
    S: Default + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad + res_grad);
    }
}

impl<'g, S> Pow<&'g Variable<S>> for S
where
    S: 'g + Clone + Pow<Output = S>,
{
    type Output = Tensor<S, PowBackwardSV<'g, S>>;
    fn pow(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.pow(rhs.data().clone()),
            func: PowBackwardSV {
                ygrad: &rhs.grad,
            }
        }
    }
}
