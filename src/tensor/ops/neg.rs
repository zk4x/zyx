use crate::tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake};
use std::{ops::{Sub, Neg}, cell::RefCell};

#[derive(Debug, Clone, Copy)]
pub struct NegBackwardG<'g, S> {
    grad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for NegBackwardG<'g, S>
where
    S: Default + Neg<Output = S> + Sub<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_take(|grad| grad - res_grad);
    }
}

impl<'g, S> Neg for &'g Variable<S>
where
    S: 'g + Clone + Neg<Output = S>,
{
    type Output = Tensor<S, NegBackwardG<'g, S>>;
    fn neg(self) -> Self::Output {
        Tensor {
            data: (*self.data()).clone().neg(),
            func: NegBackwardG {
                grad: &self.grad,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NegBackwardF<F> {
    func: F,
}

impl<S, F> Backward<S> for NegBackwardF<F>
where
    S: Neg<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(-res_grad);
    }
}

impl<S, F> Neg for Tensor<S, F>
where
    F: FnOnce(S),
    S: Neg<Output = S>,
{
    type Output = Tensor<S, NegBackwardF<F>>;
    fn neg(self) -> Self::Output {
        Tensor {
            data: self.data.neg(),
            func: NegBackwardF {
                func: self.func,
            },
        }
    }
}
