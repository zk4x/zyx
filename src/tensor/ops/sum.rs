use crate::{ops::{Sum, Expand, GetShape}, tensor::{Tensor, TensorGrad, TensorFunc}};
use std::{rc::Rc, ops::Add};

impl<S> Sum for Tensor<S>
where
    for<'a> &'a S: Sum<Output = S>,
{
    type Output = Tensor<S>;
    fn sum(self, dims: &[i32]) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.sum(dims)),
        }
    }
}

// Why is compiler unable to distinguish between this function and function above?
/*impl<S> Tensor<S> {
    pub fn sum<const N: usize>(&self) -> Tensor<S>
    where
        S: GetShape<N>, // this is needed so that compiler can infer N
        for<'a> &'a S: ops::Sum<[i32; N], N, Output = S>,
    {
        use ops::Sum;
        Tensor {
            data: Rc::new(self.data.sum(std::array::from_fn(|i| i as i32))),
        }
    }
}*/

impl<'g, S> Sum for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Sum<Output = S> + Add<Output = S> + Expand<Output = S> + GetShape,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sum(self, dims: &[i32]) -> Self::Output {
        let self_grad = &self.grad;
        let self_shape = self.data.borrow().shape();
        TensorFunc {
            data: Rc::new(self.data.borrow().sum(dims)),
            func: move |res_grad: S| { self_grad.replace_with(|grad| &*grad + &res_grad.expand(&self_shape)); },
        }
    }
}

impl<S, F> Sum for TensorFunc<S, F>
where
    for<'a> &'a S: Sum<Output = S> + Add<Output = S> + Expand<Output = S> + GetShape,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sum(self, dims: &[i32]) -> Self::Output {
        let self_func = self.func;
        let self_shape = self.data.shape();
        TensorFunc {
            data: Rc::new(self.data.sum(dims)),
            func: move |res_grad: S| self_func(res_grad.expand(&self_shape)),
        }
    }
}