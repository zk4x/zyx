use crate::{ops::{Expand, Max, GetShape}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone)]
pub struct ExpandBackwardG<'g, S> {
    grad: &'g RefCell<S>,
    dims: Vec<i32>,
}

impl<'g, S> Backward<S> for ExpandBackwardG<'g, S>
where
    S: Default + Expand<Output = S> + Add<Output = S> + Max<Output = S> + GetShape,
{
    fn backward(self, res_grad: S) {
        // TODO: is max correct reduce for expand backward?
        self.grad.replace_take(|grad| grad + res_grad.max(&self.dims));
    }
}

impl<'g, S> Expand for &'g Variable<S>
where
    S: 'g + Clone + Expand<Output = S> + GetShape,
{
    type Output = Tensor<S, ExpandBackwardG<'g, S>>;
    fn expand(self, shape: &[usize]) -> Self::Output {
        let dims = self.data().shape().iter().zip(shape.iter()).enumerate().filter_map(|(i, (a, b))| if a != b { Some(i as i32) } else { None }).collect();
        Tensor {
            data: (*self.data()).clone().expand(shape),
            func: ExpandBackwardG {
                grad: &self.grad,
                dims,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExpandBackwardF<F> {
    func: F,
    dims: Vec<i32>,
}

impl<S, F> Backward<S> for ExpandBackwardF<F>
where
    S: Expand<Output = S> + Max<Output = S> + GetShape,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(res_grad.max(&self.dims));
    }
}

impl<S, F> Expand for Tensor<S, F>
where
    F: FnOnce(S),
    S: Expand<Output = S> + GetShape,
{
    type Output = Tensor<S, ExpandBackwardF<F>>;
    fn expand(self, shape: &[usize]) -> Self::Output {
        let dims = self.data.shape().iter().zip(shape.iter()).enumerate().filter_map(|(i, (a, b))| if a != b { Some(i as i32) } else { None }).collect();
        Tensor {
            data: self.data.expand(shape),
            func: ExpandBackwardF {
                func: self.func,
                dims,
            },
        }
    }
}
