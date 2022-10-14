use crate::{ops::{Expand, Max, IntoShape}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone)]
pub struct ExpandBackwardV<'g, S> {
    grad: &'g RefCell<S>,
    dims: Vec<i32>,
}

impl<'g, S> Backward<S> for ExpandBackwardV<'g, S>
where
    S: Default + Expand<Output = S> + Add<Output = S> + Max<Output = S> + IntoShape,
{
    fn backward(self, res_grad: S) {
        // TODO: is max correct reduce for expand backward?
        self.grad.replace_take(|grad| grad + res_grad.max(&self.dims));
    }
}

impl<'g, S> Expand for &'g Variable<S>
where
    S: 'g + Clone + Expand<Output = S> + IntoShape,
{
    type Output = Tensor<S, ExpandBackwardV<'g, S>>;
    fn expand(self, shape: &[usize]) -> Self::Output {
        let dims = self.data().shape().iter().zip(shape.iter()).enumerate().filter_map(|(i, (a, b))| if a != b { Some(i as i32) } else { None }).collect();
        Tensor {
            data: (*self.data()).clone().expand(shape),
            func: ExpandBackwardV {
                grad: &self.grad,
                dims,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExpandBackwardT<F> {
    func: F,
    dims: Vec<i32>,
}

impl<S, F> Backward<S> for ExpandBackwardT<F>
where
    S: Expand<Output = S> + Max<Output = S> + IntoShape,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(res_grad.max(&self.dims));
    }
}

impl<S, F> Expand for Tensor<S, F>
where
    F: FnOnce(S),
    S: Expand<Output = S> + IntoShape,
{
    type Output = Tensor<S, ExpandBackwardT<F>>;
    fn expand(self, shape: &[usize]) -> Self::Output {
        let dims = self.data.shape().iter().zip(shape.iter()).enumerate().filter_map(|(i, (a, b))| if a != b { Some(i as i32) } else { None }).collect();
        Tensor {
            data: self.data.expand(shape),
            func: ExpandBackwardT {
                func: self.func,
                dims,
            },
        }
    }
}
