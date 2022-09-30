use crate::{ops::{Tanh, Ones, Pow}, tensor::{Tensor, TensorGrad, TensorFunc}};
use std::{rc::Rc, ops::{Add, Mul, Neg}};

impl<S> Tanh for Tensor<S>
where
    for<'a> &'a S: Tanh<Output = S>,
{
    type Output = Tensor<S>;
    fn tanh(self) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.tanh()),
        }
    }
}

impl<'g, S> Tanh for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Tanh<Output = S>
        + Mul<Output = S>
        + Add<Output = S>
        + Pow<Output = S>
        + Neg<Output = S>,
    S: Ones,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn tanh(self) -> Self::Output {
        let self_grad = &self.grad;
        let data = Rc::new(self.data.borrow().tanh());
        TensorFunc {
            data: Rc::clone(&data),
            func: move |res_grad| {
                self_grad.replace_with(|grad| &*grad + &(&res_grad * &(&-&data.pow(&(&S::ones(&[1]) + &S::ones(&[1]))) + &S::ones(&[1]))));
            },
        }
    }
}

impl<S, F> Tanh for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: Tanh<Output = S>
        + Mul<Output = S>
        + Add<Output = S>
        + Pow<Output = S>
        + Neg<Output = S>,
    S: Ones,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn tanh(self) -> Self::Output {
        let self_func = self.func;
        let data = Rc::new(self.data.tanh());
        TensorFunc {
            data: Rc::clone(&data),
            func: move |res_grad| self_func(&res_grad * &(&-&data.pow(&(&S::ones(&[1]) + &S::ones(&[1]))) + &S::ones(&[1]))),
        }
    }
}