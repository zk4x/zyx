use crate::{ops::Exp, tensor::{Tensor, TensorGrad, TensorFunc}};
use std::{rc::Rc, ops::{Add, Mul}};

impl<S> Exp for Tensor<S>
where
    for<'a> &'a S: Exp<Output = S>,
{
    type Output = Tensor<S>;
    fn exp(self) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.exp()),
        }
    }
}

impl<'g, S> Exp for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Exp<Output = S> + Mul<Output = S> + Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn exp(self) -> Self::Output {
        let self_grad = &self.grad;
        let data = Rc::new(self.data.borrow().exp());
        TensorFunc {
            data: Rc::clone(&data),
            func: move |res_grad| { self_grad.replace_with(|grad| &*grad + &(&res_grad * &data)); },
        }
    }
}

impl<S, F> Exp for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: Exp<Output = S> + Mul<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn exp(self) -> Self::Output {
        let self_func = self.func;
        let data = Rc::new(self.data.exp());
        TensorFunc {
            data: Rc::clone(&data),
            func: move |res_grad| self_func(&res_grad * &data),
        }
    }
}