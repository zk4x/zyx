use crate::{ops::Exp, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::{Add, Mul}, cell::RefCell};

#[derive(Debug, Clone, Copy)]
pub struct ExpBackwardV<'g, S> {
    grad: &'g RefCell<S>,
    data: S,
}

impl<'g, S> Backward<S> for ExpBackwardV<'g, S>
where
    S: Default + Exp<Output = S> + Mul<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad + res_grad * self.data);
    }
}

impl<'g, S> Exp for &'g Variable<S>
where
    S: 'g + Clone + Exp<Output = S>,
{
    type Output = Tensor<S, ExpBackwardV<'g, S>>;
    fn exp(self) -> Self::Output {
        let data = (*self.data()).clone().exp();
        Tensor {
            data: data.clone(),
            func: ExpBackwardV {
                grad: &self.grad,
                data,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ExpBackwardT<S, F> {
    func: F,
    data: S,
}

impl<S, F> Backward<S> for ExpBackwardT<S, F>
where
    S: Exp<Output = S> + Mul<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(res_grad * self.data);
    }
}

impl<S, F> Exp for Tensor<S, F>
where
    S: Clone + Exp<Output = S>,
{
    type Output = Tensor<S, ExpBackwardT<S, F>>;
    fn exp(self) -> Self::Output {
        let data = self.data.exp();
        Tensor {
            data: data.clone(),
            func: ExpBackwardT {
                func: self.func,
                data,
            },
        }
    }
}