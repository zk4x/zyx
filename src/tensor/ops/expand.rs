use crate::{ops::{Expand, Max, GetShape}, tensor::{Tensor, TensorGrad, TensorFunc, Backward}};
use std::{rc::Rc, ops::Add, cell::RefCell};

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

#[derive(Debug)]
pub struct ExpandBackwardG<'g, S> {
    grad: &'g RefCell<S>,
    dims: Vec<i32>,
}

impl<'g, S> Clone for ExpandBackwardG<'g, S> {
    fn clone(&self) -> Self {
        Self {
            grad: self.grad,
            dims: self.dims.clone(),
        }
    }
}

impl<'g, S> Backward<S> for ExpandBackwardG<'g, S>
where
    for<'a> &'a S: Expand<Output = S> + Add<Output = S> + Max<Output = S> + GetShape,
{
    fn backward(self, res_grad: S) {
        // TODO: is max correct reduce for expand backward?
        self.grad.replace_with(|grad| &*grad + &res_grad.max(&self.dims));
    }
}

impl<'g, S> Expand for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Expand<Output = S> + GetShape,
{
    type Output = TensorFunc<S, ExpandBackwardG<'g, S>>;
    fn expand(self, shape: &[usize]) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.borrow().expand(shape)),
            func: ExpandBackwardG {
                grad: &self.grad,
                dims: self.data.borrow().shape().iter().zip(shape.iter()).enumerate().filter_map(|(i, (a, b))| if a != b { Some(i as i32) } else { None }).collect(),
            }
        }
    }
}

#[derive(Debug)]
pub struct ExpandBackwardF<F> {
    func: F,
    dims: Vec<i32>,
}

impl<F> Clone for ExpandBackwardF<F>
where
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            func: self.func.clone(),
            dims: self.dims.clone(),
        }
    }
}

impl<S, F> Backward<S> for ExpandBackwardF<F>
where
    for<'a> &'a S: Expand<Output = S> + Max<Output = S> + GetShape,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(res_grad.max(&self.dims));
    }
}

impl<S, F> Expand for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: Expand<Output = S> + GetShape,
{
    type Output = TensorFunc<S, ExpandBackwardF<F>>;
    fn expand(self, shape: &[usize]) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.expand(shape)),
            func: ExpandBackwardF {
                func: self.func,
                dims: self.data.shape().iter().zip(shape.iter()).enumerate().filter_map(|(i, (a, b))| if a != b { Some(i as i32) } else { None }).collect(),
            },
        }
    }
}
