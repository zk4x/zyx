use crate::{ops::{Tanh, Ones, Pow}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::{Add, Mul, Neg}, cell::RefCell};

#[derive(Debug, Clone, Copy)]
pub struct TanhBackwardG<'g, S> {
    grad: &'g RefCell<S>,
    data: S,
}

impl<'g, S> Backward<S> for TanhBackwardG<'g, S>
where
    S: Default + Tanh<Output = S> + Mul<Output = S> + Add<Output = S> + Pow<Output = S> + Neg<Output = S> + Ones,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad + res_grad * (-self.data.pow(S::ones(&[1]) + S::ones(&[1])) + S::ones(&[1])));
    }
}

impl<'g, S> Tanh for &'g Variable<S>
where
    S: 'g + Clone + Tanh<Output = S>,
{
    type Output = Tensor<S, TanhBackwardG<'g, S>>;
    fn tanh(self) -> Self::Output {
        let data = (*self.data()).clone().tanh();
        Tensor {
            data: data.clone(),
            func: TanhBackwardG {
                grad: &self.grad,
                data,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TanhBackwardF<S, F> {
    func: F,
    data: S,
}

impl<S, F> Backward<S> for TanhBackwardF<S, F>
where
    S: Tanh<Output = S> + Mul<Output = S> + Add<Output = S> + Pow<Output = S> + Neg<Output = S> + Ones,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(res_grad * (-self.data.pow(S::ones(&[1]) + S::ones(&[1])) + S::ones(&[1])));
    }
}

impl<S, F> Tanh for Tensor<S, F>
where
    S: Clone + Tanh<Output = S>,
{
    type Output = Tensor<S, TanhBackwardF<S, F>>;
    fn tanh(self) -> Self::Output {
        let data = self.data.tanh();
        Tensor {
            data: data.clone(),
            func: TanhBackwardF {
                func: self.func,
                data,
            },
        }
    }
}