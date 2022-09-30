use crate::{ops::{ReLU, DReLU}, tensor::{Tensor, TensorGrad, TensorFunc, Backward}};
use std::{rc::Rc, ops::{Add, Mul}, cell::RefCell};

impl<S> ReLU for Tensor<S>
where
    for<'a> &'a S: ReLU<Output = S>,
{
    type Output = Tensor<S>;
    fn relu(self) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.relu()),
        }
    }
}

pub struct ReLUBackwardLeaf<'g, S> {
    grad: &'g RefCell<S>,
    data: Rc<S>,
}

impl<'g, S> Clone for ReLUBackwardLeaf<'g, S> {
    fn clone(&self) -> Self {
        Self {
            grad: self.grad,
            data: self.data.clone(),
        }
    }
}

impl<'g, S> Backward<S> for ReLUBackwardLeaf<'g, S>
where
    for<'a> &'a S: DReLU<Output = S> + Mul<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_with(|grad| &*grad + &(&res_grad * &self.data.drelu()));
    }
}

impl<'g, S> ReLU for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: ReLU<Output = S>,
{
    type Output = TensorFunc<S, ReLUBackwardLeaf<'g, S>>;
    fn relu(self) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.borrow().relu()),
            func: ReLUBackwardLeaf {
                grad: &self.grad,
                data: Rc::clone(&self.data.borrow()),
            },
        }
    }
}

#[derive(Debug)]
pub struct ReLUBackward<S, F> {
    func: F,
    data: Rc<S>,
}

impl<S, F> Clone for ReLUBackward<S, F>
where
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            func: self.func.clone(),
            data: Rc::clone(&self.data),
        }
    }
}

impl<S, F> Backward<S> for ReLUBackward<S, F>
where
    for<'a> &'a S: DReLU<Output = S> + Mul<Output = S> + Add<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(&res_grad * &self.data.drelu());
    }
}

impl<S, F> ReLU for TensorFunc<S, F>
where
    for<'a> &'a S: ReLU<Output = S>,
    F: Backward<S>,
{
    type Output = TensorFunc<S, ReLUBackward<S, F>>;
    fn relu(self) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.relu()),
            func: ReLUBackward {
                func: self.func,
                data: self.data,
            },
        }
    }
}
