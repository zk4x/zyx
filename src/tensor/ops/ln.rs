use crate::{ops::{Ln, Ones, Pow}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::{Add, Mul, Neg}, cell::RefCell};

#[derive(Debug, Clone, Copy)]
pub struct LnBackwardLeaf<'g, S> {
    grad: &'g RefCell<S>,
    data: S,
}

impl<'g, S> Backward<S> for LnBackwardLeaf<'g, S>
where
    S: Default + Ln<Output = S> + Mul<Output = S> + Add<Output = S> + Pow<Output = S> + Neg<Output = S> + Ones,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad + res_grad * self.data.pow(-S::ones(&[1])));
    }
}

impl<'g, S> Ln for &'g Variable<S>
where
    S: 'g + Clone + Ln<Output = S>,
{
    type Output = Tensor<S, LnBackwardLeaf<'g, S>>;
    fn ln(self) -> Self::Output {
        Tensor {
            data: self.data().clone().ln(),
            func: LnBackwardLeaf {
                grad: &self.grad,
                data: self.data().clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LnBackward<S, F> {
    func: F,
    data: S,
}

impl<S, F> Backward<S> for LnBackward<S, F>
where
    S: Ln<Output = S> + Mul<Output = S> + Pow<Output = S> + Neg<Output = S> + Ones,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(res_grad * self.data.pow(-S::ones(&[1])));
    }
}

impl<S, F> Ln for Tensor<S, F>
where
    S: Clone + Ln<Output = S>,
{
    type Output = Tensor<S, LnBackward<S, F>>;
    fn ln(self) -> Self::Output {
        Tensor {
            data: self.data.clone().ln(),
            func: LnBackward {
                func: self.func,
                data: self.data,
            }
        }
    }
}
