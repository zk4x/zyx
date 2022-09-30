use crate::{ops::Exp, tensor::{Tensor, TensorGrad, TensorFunc, Backward}};
use std::{rc::Rc, ops::{Add, Mul}, cell::RefCell};

impl<S> Exp for Tensor<S>
where
    for<'a> &'a S: Exp<Output = S>,
{
    type Output = Tensor<S>;
    fn exp(self) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.exp()),
        }
    }
}

#[derive(Debug)]
pub struct ExpBackwardG<'g, S> {
    grad: &'g RefCell<S>,
    data: Rc<S>,
}

impl<'g, S> Clone for ExpBackwardG<'g, S> {
    fn clone(&self) -> Self {
        Self {
            grad: self.grad,
            data: Rc::clone(&self.data),
        }
    }
}

impl<'g, S> Backward<S> for ExpBackwardG<'g, S>
where
    for<'a> &'a S: Exp<Output = S> + Mul<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_with(|grad| &*grad + &(&res_grad * &self.data));
    }
}

impl<'g, S> Exp for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Exp<Output = S>,
{
    type Output = TensorFunc<S, ExpBackwardG<'g, S>>;
    fn exp(self) -> Self::Output {
        let data = Rc::new(self.data.borrow().exp());
        TensorFunc {
            data: Rc::clone(&data),
            func: ExpBackwardG {
                grad: &self.grad,
                data,
            },
        }
    }
}

#[derive(Debug)]
pub struct ExpBackwardF<S, F> {
    func: F,
    data: Rc<S>,
}

impl<S, F> Clone for ExpBackwardF<S, F>
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

impl<S, F> Backward<S> for ExpBackwardF<S, F>
where
    for<'a> &'a S: Exp<Output = S> + Mul<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(&res_grad * &self.data);
    }
}

impl<S, F> Exp for TensorFunc<S, F>
where
    for<'a> &'a S: Exp<Output = S>,
    F: Backward<S>,
{
    type Output = TensorFunc<S, ExpBackwardF<S, F>>;
    fn exp(self) -> Self::Output {
        let data = Rc::new(self.data.exp());
        TensorFunc {
            data: Rc::clone(&data),
            func: ExpBackwardF {
                func: self.func,
                data,
            },
        }
    }
}