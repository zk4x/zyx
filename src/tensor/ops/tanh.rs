use crate::{ops::{Tanh, Ones, Pow}, tensor::{Tensor, TensorGrad, TensorFunc, Backward}};
use std::{rc::Rc, ops::{Add, Mul, Neg}, cell::RefCell};

impl<S> Tanh for Tensor<S>
where
    for<'a> &'a S: Tanh<Output = S>,
{
    type Output = Tensor<S>;
    fn tanh(self) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.tanh()),
        }
    }
}

#[derive(Debug)]
pub struct TanhBackwardG<'g, S> {
    grad: &'g RefCell<S>,
    data: Rc<S>,
}

impl<'g, S> Backward<S> for TanhBackwardG<'g, S>
where
    for<'a> &'a S: Tanh<Output = S> + Mul<Output = S> + Add<Output = S> + Pow<Output = S> + Neg<Output = S>,
    S: Ones,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_with(|grad| &*grad + &(&res_grad * &(&-&self.data.pow(&(&S::ones(&[1]) + &S::ones(&[1]))) + &S::ones(&[1]))));
    }
}

impl<'g, S> Tanh for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Tanh<Output = S>,
{
    type Output = TensorFunc<S, TanhBackwardG<'g, S>>;
    fn tanh(self) -> Self::Output {
        let data = Rc::new(self.data.borrow().tanh());
        TensorFunc {
            data: Rc::clone(&data),
            func: TanhBackwardG {
                grad: &self.grad,
                data,
            }
        }
    }
}

#[derive(Debug)]
pub struct TanhBackwardF<S, F> {
    func: F,
    data: Rc<S>,
}

impl<S, F> Backward<S> for TanhBackwardF<S, F>
where
    for<'a> &'a S: Tanh<Output = S> + Mul<Output = S> + Add<Output = S> + Pow<Output = S> + Neg<Output = S>,
    S: Ones,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(&res_grad * &(&-&self.data.pow(&(&S::ones(&[1]) + &S::ones(&[1]))) + &S::ones(&[1])));
    }
}

impl<S, F> Tanh for TensorFunc<S, F>
where
    for<'a> &'a S: Tanh<Output = S>,
{
    type Output = TensorFunc<S, TanhBackwardF<S, F>>;
    fn tanh(self) -> Self::Output {
        let data = Rc::new(self.data.tanh());
        TensorFunc {
            data: Rc::clone(&data),
            func: TanhBackwardF {
                func: self.func,
                data,
            },
        }
    }
}