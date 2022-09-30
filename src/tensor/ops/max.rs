use crate::{ops::{Max, Expand, GetShape}, tensor::{Tensor, TensorGrad, TensorFunc}};
use std::{rc::Rc, ops::Add};

impl<S> Max for Tensor<S>
where
    for<'a> &'a S: Max<Output = S>,
{
    type Output = Tensor<S>;
    fn max(self, dims: &[i32]) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.max(dims)),
        }
    }
}

impl<'g, S> Max for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Max<Output = S> + Add<Output = S> + Expand<Output = S> + GetShape,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn max(self, dims: &[i32]) -> Self::Output {
        let self_grad = &self.grad;
        let self_shape = self.data.borrow().shape();
        TensorFunc {
            data: Rc::new(self.data.borrow().max(dims)),
            func: move |res_grad: S| { self_grad.replace_with(|grad| &*grad + &res_grad.expand(&self_shape)); },
        }
    }
}

impl<S, F> Max for TensorFunc<S, F>
where
    for<'a> &'a S: Max<Output = S> + Add<Output = S> + Expand<Output = S> + GetShape,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn max(self, dims: &[i32]) -> Self::Output {
        let self_func = self.func;
        let self_shape = self.data.shape();
        TensorFunc {
            data: Rc::new(self.data.max(dims)),
            func: move |res_grad: S| self_func(res_grad.expand(&self_shape)),
        }
    }
}