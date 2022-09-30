use crate::{ops::Pow, tensor::{Tensor, TensorGrad, TensorFunc}};
use std::{rc::Rc, ops::{Add, Mul, Neg}};

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

impl<'g, S> Neg for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Mul<Output = S> + Add<Output = S>,
    for<'b> &'b S: Neg<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn neg(self) -> Self::Output {
        let self_grad = &self.grad;
        TensorFunc {
            data: Rc::new(self.data().neg()),
            func: move |res_grad: S| { self_grad.replace_with(|grad| &*grad + &(-&res_grad)); },
        }
    }
}

impl<S, F> Neg for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: Neg<Output = S> + Mul<Output = S> + Pow<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn neg(self) -> Self::Output {
        let self_func = self.func;
        TensorFunc {
            data: Rc::new(self.data.neg()),
            func: move |res_grad| self_func(-&res_grad),
        }
    }
}
