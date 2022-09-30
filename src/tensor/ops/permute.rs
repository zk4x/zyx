use crate::{shape::Dims, ops::Permute, tensor::{Tensor, TensorGrad, TensorFunc}};
use std::{rc::Rc, ops::Add};

impl<S> Permute for Tensor<S>
where
    for<'a> &'a S: Permute<Output = S>,
{
    type Output = Tensor<S>;
    fn permute(self, dims: &[i32]) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.permute(dims)),
        }
    }
}

impl<'g, S> Permute for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Permute<Output = S> + Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn permute(self, dims: &[i32]) -> Self::Output {
        let self_grad = &self.grad;
        let inv_dims = dims.argsort();
        TensorFunc {
            data: Rc::new(self.data().permute(dims)),
            func: move |res_grad: S| { self_grad.replace_with(|grad| &*grad + &res_grad.permute(&inv_dims)); },
        }
    }
}

impl<S, F> Permute for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: Permute<Output = S> + Permute<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn permute(self, dims: &[i32]) -> Self::Output {
        let self_func = self.func;
        let inv_dims = dims.argsort();
        TensorFunc {
            data: Rc::new(self.data.permute(dims)),
            func: move |res_grad: S| {
                self_func(res_grad.permute(&inv_dims));
            },
        }
    }
}