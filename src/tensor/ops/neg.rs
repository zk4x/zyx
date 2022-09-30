use crate::{ops::Pow, tensor::{Tensor, TensorGrad, TensorFunc, Backward}};
use std::{rc::Rc, ops::{Add, Mul, Neg}, cell::RefCell};

impl<S> Neg for Tensor<S>
where
    for<'a> &'a S: Neg<Output = S>,
{
    type Output = Tensor<S>;
    fn neg(self) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.neg()),
        }
    }
}

#[derive(Debug)]
pub struct NegBackwardG<'g, S> {
    grad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for NegBackwardG<'g, S>
where
    for<'a> &'a S: Neg<Output = S> + Mul<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_with(|grad| &*grad + &(-&res_grad));
    }
}

impl<'g, S> Neg for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Neg<Output = S>,
{
    type Output = TensorFunc<S, NegBackwardG<'g, S>>;
    fn neg(self) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data().neg()),
            func: NegBackwardG {
                grad: &self.grad,
            }
        }
    }
}

#[derive(Debug)]
pub struct NegBackwardF<F> {
    func: F,
}

impl<S, F> Backward<S> for NegBackwardF<F>
where
    for<'a> &'a S: Neg<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(-&res_grad);
    }
}

impl<S, F> Neg for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: Neg<Output = S> + Mul<Output = S> + Pow<Output = S>,
{
    type Output = TensorFunc<S, NegBackwardF<F>>;
    fn neg(self) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.neg()),
            func: NegBackwardF {
                func: self.func,
            },
        }
    }
}
