use crate::{ops::{Ln, Ones, Pow}, tensor::{Tensor, TensorGrad, TensorFunc}};
use std::{rc::Rc, ops::{Add, Mul, Neg}};

impl<S> Ln for Tensor<S>
where
    for<'a> &'a S: Ln<Output = S>,
{
    type Output = Tensor<S>;
    fn ln(self) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.ln()),
        }
    }
}

impl<'g, S> Ln for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Ln<Output = S>
        + Mul<Output = S>
        + Add<Output = S>
        + Pow<Output = S>
        + std::ops::Neg<Output = S>,
    S: Ones,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn ln(self) -> Self::Output {
        let self_grad = &self.grad;
        let self_data = self.data();
        TensorFunc {
            data: Rc::new(self_data.ln()),
            func: move |res_grad| { self_grad.replace_with(|grad| &*grad + &(&res_grad * &self_data.pow(&-&S::ones(&[1])))); },
        }
    }
}

impl<S, F> Ln for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: Ln<Output = S> + std::ops::Mul<Output = S> + Pow<Output = S> + Neg<Output = S>,
    S: Ones,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn ln(self) -> Self::Output {
        let self_func = self.func;
        let self_data = self.data;
        TensorFunc {
            data: Rc::new(self_data.ln()),
            func: move |res_grad| self_func(&res_grad * &self_data.pow(&-&S::ones(&[1]))),
        }
    }
}