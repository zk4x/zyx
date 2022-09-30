use crate::{ops::{Ln, Ones, Pow}, tensor::{Tensor, TensorGrad, TensorFunc, Backward}};
use std::{rc::Rc, ops::{Add, Mul, Neg}, cell::RefCell};

impl<S> Ln for Tensor<S>
where
    for<'a> &'a S: Ln<Output = S>,
{
    type Output = Tensor<S>;
    fn ln(self) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.ln()),
        }
    }
}

#[derive(Debug)]
pub struct LnBackwardLeaf<'g, S> {
    grad: &'g RefCell<S>,
    data: Rc<S>,
}

impl<'g, S> Clone for LnBackwardLeaf<'g, S> {
    fn clone(&self) -> Self {
        Self {
            grad: self.grad,
            data: Rc::clone(&self.data),
        }
    }
}

impl<'g, S> Backward<S> for LnBackwardLeaf<'g, S>
where
    for<'a> &'a S: Ln<Output = S> + Mul<Output = S> + Add<Output = S> + Pow<Output = S> + std::ops::Neg<Output = S>,
    S: Ones,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_with(|grad| &*grad + &(&res_grad * &self.data.pow(&-&S::ones(&[1]))));
    }
}

impl<'g, S> Ln for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Ln<Output = S>,
{
    type Output = TensorFunc<S, LnBackwardLeaf<'g, S>>;
    fn ln(self) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.borrow().ln()),
            func: LnBackwardLeaf {
                grad: &self.grad,
                data: Rc::clone(&self.data.borrow()),
            },
        }
    }
}

#[derive(Debug)]
pub struct LnBackward<S, F> {
    func: F,
    data: Rc<S>,
}

impl<S, F> Clone for LnBackward<S, F>
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

impl<S, F> Backward<S> for LnBackward<S, F>
where
    for<'a> &'a S: Ln<Output = S> + std::ops::Mul<Output = S> + Pow<Output = S> + Neg<Output = S>,
    S: Ones,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(&res_grad * &self.data.pow(&-&S::ones(&[1])));
    }
}

impl<S, F> Ln for TensorFunc<S, F>
where
    for<'a> &'a S: Ln<Output = S>,
{
    type Output = TensorFunc<S, LnBackward<S, F>>;
    fn ln(self) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.ln()),
            func: LnBackward {
                func: self.func,
                data: self.data,
            }
        }
    }
}
