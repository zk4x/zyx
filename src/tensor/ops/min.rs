use crate::{ops::{Min, Expand, GetShape}, tensor::{Tensor, TensorGrad, TensorFunc}};
use std::{rc::Rc, ops::Add};

impl<S> Min for Tensor<S>
where
    for<'a> &'a S: Min<Output = S>,
{
    type Output = Tensor<S>;
    fn min(self, dims: &[i32]) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.min(dims)),
        }
    }
}

impl<'g, S> Min for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Min<Output = S> + Add<Output = S> + Expand<Output = S> + GetShape,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn min(self, dims: &[i32]) -> Self::Output {
        let self_grad = &self.grad;
        let self_shape = self.data.borrow().shape();
        TensorFunc {
            data: Rc::new(self.data.borrow().min(dims)),
            func: move |res_grad: S| { self_grad.replace_with(|grad| &*grad + &res_grad.expand(&self_shape)); },
        }
    }
}

impl<S, F> Min for TensorFunc<S, F>
where
    for<'a> &'a S: Min<Output = S> + Add<Output = S> + Expand<Output = S> + GetShape,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn min(self, dims: &[i32]) -> Self::Output {
        let self_func = self.func;
        let self_shape = self.data.shape();
        TensorFunc {
            data: Rc::new(self.data.min(dims)),
            func: move |res_grad: S| self_func(res_grad.expand(&self_shape)),
        }
    }
}