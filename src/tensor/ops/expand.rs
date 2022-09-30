use crate::{ops::{Expand, Max, GetShape}, tensor::{Tensor, TensorGrad, TensorFunc}};
use std::{rc::Rc, ops::Add};

impl<S> Expand for Tensor<S>
where
    for<'a> &'a S: Expand<Output = S>,
{
    type Output = Tensor<S>;
    fn expand(self, shape: &[usize]) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.expand(shape)),
        }
    }
}

impl<'g, S> Expand for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Expand<Output = S> + Add<Output = S> + Max<Output = S> + GetShape,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn expand(self, shape: &[usize]) -> Self::Output {
        // TODO: is max correct reduce for expand backward?
        let self_grad = &self.grad;
        let dims: Vec<i32> = self.data.borrow().shape().iter().zip(shape.iter()).enumerate().filter_map(|(i, (a, b))| if a != b { Some(i as i32) } else { None }).collect();
        TensorFunc {
            data: Rc::new(self.data().expand(shape)),
            func: move |res_grad: S| { self_grad.replace_with(|grad| &*grad + &res_grad.max(&dims)); },
        }
    }
}

impl<S, F> Expand for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: Expand<Output = S> + Max<Output = S> + GetShape,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn expand(self, shape: &[usize]) -> Self::Output {
        let self_func = self.func;
        let dims: Vec<i32> = self.data.shape().iter().zip(shape.iter()).enumerate().filter_map(|(i, (a, b))| if a != b { Some(i as i32) } else { None }).collect();
        TensorFunc {
            data: Rc::new(self.data.expand(shape)),
            func: move |res_grad: S| self_func(res_grad.max(&dims)),
        }
    }
}