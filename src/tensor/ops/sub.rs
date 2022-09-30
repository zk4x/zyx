use crate::{tensor::{Tensor, TensorGrad, TensorFunc}};
use std::{rc::Rc, ops::{Add, Sub}};

impl<S> Sub<Tensor<S>> for Tensor<S>
where
    for<'a> &'a S: Sub<Output = S>,
{
    type Output = Tensor<S>;
    fn sub(self, rhs: Tensor<S>) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.as_ref() - rhs.data.as_ref()),
        }
    }
}

impl<'g, S> Sub<&'g TensorGrad<S>> for Tensor<S>
where
    S: 'g,
    for<'a> &'a S: Sub<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sub(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        let rhs_grad = &rhs.grad;
        TensorFunc {
            data: Rc::new(self.data.as_ref() - rhs.data().as_ref()),
            func: move |res_grad| {
                rhs_grad.replace_with(|grad| &*grad - &res_grad);
            },
        }
    }
}

impl<S, F> Sub<TensorFunc<S, F>> for Tensor<S>
where
    F: FnOnce(S),
    for<'a> &'a S: Sub<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sub(self, rhs: TensorFunc<S, F>) -> Self::Output {
        let rhs_func = rhs.func;
        TensorFunc {
            data: Rc::new(self.data.as_ref() - rhs.data.as_ref()),
            func: move |res_grad| {
                rhs_func(res_grad);
            },
        }
    }
}

impl<'g, S> Sub<Tensor<S>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Sub<Output = S> + Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sub(self, rhs: Tensor<S>) -> Self::Output {
        let self_grad = &self.grad;
        TensorFunc {
            data: Rc::new(self.data().as_ref() - rhs.data.as_ref()),
            func: move |res_grad| { self_grad.replace_with(|grad| &*grad + &res_grad); },
        }
    }
}

impl<'g, S> Sub<&'g TensorGrad<S>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Sub<Output = S> + Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sub(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        let self_grad = &self.grad;
        let rhs_grad = &rhs.grad;
        TensorFunc {
            data: Rc::new(self.data().as_ref() - rhs.data().as_ref()),
            func: move |res_grad| {
                self_grad.replace_with(|grad| &*grad + &res_grad);
                rhs_grad.replace_with(|grad| &*grad - &res_grad);
            },
        }
    }
}

impl<'g, S, F> Sub<TensorFunc<S, F>> for &'g TensorGrad<S>
where
    S: 'g,
    F: FnOnce(S),
    for<'a> &'a S: Sub<Output = S> + Add<Output = S> + std::ops::Neg<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sub(self, rhs: TensorFunc<S, F>) -> Self::Output {
        let self_grad = &self.grad;
        let rhs_func = rhs.func;
        TensorFunc {
            data: Rc::new(self.data().as_ref() - rhs.data.as_ref()),
            func: move |res_grad| {
                self_grad.replace_with(|grad| &*grad + &res_grad);
                rhs_func(-&res_grad);
            },
        }
    }
}

impl<S, F> Sub<Tensor<S>> for TensorFunc<S, F>
where
    for<'a> &'a S: Sub<Output = S>,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sub(self, rhs: Tensor<S>) -> Self::Output {
        let self_func = self.func;
        TensorFunc {
            data: Rc::new(self.data.as_ref() - rhs.data.as_ref()),
            func: move |res_grad| {
                self_func(res_grad);
            },
        }
    }
}

impl<'g, S, F> Sub<&'g TensorGrad<S>> for TensorFunc<S, F>
where
    S: 'g,
    for<'a> &'a S: Sub<Output = S>,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sub(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        let self_func = self.func;
        let rhs_grad = &rhs.grad;
        TensorFunc {
            data: Rc::new(self.data.as_ref() - rhs.data().as_ref()),
            func: move |res_grad| {
                rhs_grad.replace_with(|grad| &*grad - &res_grad);
                self_func(res_grad);
            },
        }
    }
}

impl<S, F1, F2> Sub<TensorFunc<S, F2>> for TensorFunc<S, F1>
where
    F1: FnOnce(S),
    F2: FnOnce(S),
    for<'a> &'a S: Sub<Output = S>,
    for<'b> &'b S: std::ops::Neg<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sub(self, rhs: TensorFunc<S, F2>) -> Self::Output {
        let self_func = self.func;
        let rhs_func = rhs.func;
        TensorFunc {
            data: Rc::new(self.data.as_ref() - rhs.data.as_ref()),
            func: move |res_grad: S| {
                rhs_func(-&res_grad);
                self_func(res_grad);
            },
        }
    }
}