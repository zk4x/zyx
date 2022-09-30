use crate::{ops::{Reshape, GetShape}, tensor::{Tensor, TensorGrad, TensorFunc}};
use std::{rc::Rc, ops::Add};

impl<S> Reshape for Tensor<S>
where
    for<'a> &'a S: Reshape<Output = S>,
{
    type Output = Tensor<S>;
    fn reshape(self, shape: &[usize]) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.reshape(shape)),
        }
    }
}

impl<'g, S> Reshape for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Reshape<Output = S> + Add<Output = S> + GetShape,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn reshape(self, shape: &[usize]) -> Self::Output {
        let self_grad = &self.grad;
        let self_shape = self.data.borrow().shape();
        TensorFunc {
            data: Rc::new(self.data().reshape(shape)),
            func: move |res_grad: S| { self_grad.replace_with(|grad| &*grad + &res_grad.reshape(&self_shape)); },
        }
    }
}

impl<S, F> Reshape for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: Reshape<Output = S> + GetShape,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn reshape(self, shape: &[usize]) -> Self::Output {
        let self_func = self.func;
        let self_shape = self.data.shape();
        TensorFunc {
            data: Rc::new(self.data.reshape(shape)),
            func: move |res_grad: S| self_func(res_grad.reshape(&self_shape)),
        }
    }
}