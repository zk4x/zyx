use crate::{shape::Dims, ops::Permute, tensor::{Tensor, TensorGrad, TensorFunc, Backward}};
use std::{rc::Rc, ops::Add, cell::RefCell};

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

#[derive(Debug)]
pub struct PermuteBackwardG<'g, S> {
    grad: &'g RefCell<S>,
    dims: Vec<i32>,
}

impl<'g, S> Backward<S> for PermuteBackwardG<'g, S>
where
    for<'a> &'a S: Permute<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_with(|grad| &*grad + &res_grad.permute(&self.dims));
    }
}

impl<'g, S> Permute for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Permute<Output = S>,
{
    type Output = TensorFunc<S, PermuteBackwardG<'g, S>>;
    fn permute(self, dims: &[i32]) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data().permute(dims)),
            func: PermuteBackwardG {
                grad: &self.grad,
                dims: dims.argsort(),
            }
        }
    }
}

#[derive(Debug)]
pub struct PermuteBackwardF<F> {
    func: F,
    dims: Vec<i32>,
}

impl<S, F> Backward<S> for PermuteBackwardF<F>
where
    for<'a> &'a S: Permute<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(res_grad.permute(&self.dims));
    }
}

impl<S, F> Permute for TensorFunc<S, F>
where
    for<'a> &'a S: Permute<Output = S>,
{
    type Output = TensorFunc<S, PermuteBackwardF<F>>;
    fn permute(self, dims: &[i32]) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.permute(dims)),
            func: PermuteBackwardF {
                func: self.func,
                dims: dims.argsort(),
            }
        }
    }
}