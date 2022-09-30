use crate::{tensor::{Tensor, TensorGrad, TensorFunc}};
use std::{rc::Rc, ops::Add};

impl<S> Add<Tensor<S>> for Tensor<S>
where
    for<'a> &'a S: Add<Output = S>,
{
    type Output = Tensor<S>;
    fn add(self, rhs: Tensor<S>) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.as_ref() + rhs.data.as_ref()),
        }
    }
}

impl<'g, S> Add<&'g TensorGrad<S>> for Tensor<S>
where
    S: 'g,
    for<'a> &'a S: Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn add(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        let rhs_grad = &rhs.grad;
        TensorFunc {
            data: Rc::new(self.data.as_ref() + rhs.data().as_ref()),
            func: move |res_grad| { rhs_grad.replace_with(|grad| &*grad + &res_grad);
            },
        }
    }
}

impl<S, F> Add<TensorFunc<S, F>> for Tensor<S>
where
    F: FnOnce(S),
    for<'a> &'a S: Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn add(self, rhs: TensorFunc<S, F>) -> Self::Output {
        let rhs_func = rhs.func;
        TensorFunc {
            data: Rc::new(self.data.as_ref() + rhs.data.as_ref()),
            func: move |res_grad| {
                rhs_func(res_grad);
            },
        }
    }
}

impl<'g, S> Add<Tensor<S>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn add(self, rhs: Tensor<S>) -> Self::Output {
        let self_grad = &self.grad;
        TensorFunc {
            data: Rc::new(self.data().as_ref() + rhs.data.as_ref()),
            func: move |res_grad| { self_grad.replace_with(|grad| &*grad + &res_grad); },
        }
    }
}

impl<'g, S> Add<&'g TensorGrad<S>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn add(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        let self_grad = &self.grad;
        let rhs_grad = &rhs.grad;
        TensorFunc {
            data: Rc::new(self.data().as_ref() + rhs.data().as_ref()),
            func: move |res_grad| {
                self_grad.replace_with(|grad| &*grad + &res_grad);
                rhs_grad.replace_with(|grad| &*grad + &res_grad);
            },
        }
    }
}

impl<'g, S, F> Add<TensorFunc<S, F>> for &'g TensorGrad<S>
where
    S: 'g,
    F: FnOnce(S),
    for<'a> &'a S: Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn add(self, rhs: TensorFunc<S, F>) -> Self::Output {
        let self_grad = &self.grad;
        let rhs_func = rhs.func;
        TensorFunc {
            data: Rc::new(self.data().as_ref() + rhs.data.as_ref()),
            func: move |res_grad| {
                self_grad.replace_with(|grad| &*grad + &res_grad);
                rhs_func(res_grad);
            },
        }
    }
}

impl<S, F> Add<Tensor<S>> for TensorFunc<S, F>
where
    for<'a> &'a S: Add<Output = S>,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn add(self, rhs: Tensor<S>) -> Self::Output {
        let self_func = self.func;
        TensorFunc {
            data: Rc::new(self.data.as_ref() + rhs.data.as_ref()),
            func: self_func,
        }
    }
}

impl<'g, S, F> Add<&'g TensorGrad<S>> for TensorFunc<S, F>
where
    S: 'g,
    for<'a> &'a S: Add<Output = S>,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn add(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        let self_func = self.func;
        let rhs_grad = &rhs.grad;
        TensorFunc {
            data: Rc::new(self.data.as_ref() + rhs.data().as_ref()),
            func: move |res_grad| {
                rhs_grad.replace_with(|grad| &*grad + &res_grad);
                self_func(res_grad);
            },
        }
    }
}

impl<S, F1, F2> Add<TensorFunc<S, F2>> for TensorFunc<S, F1>
where
    S: Clone,
    F1: FnOnce(S),
    F2: FnOnce(S),
    for<'a> &'a S: Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn add(self, rhs: TensorFunc<S, F2>) -> Self::Output {
        let self_func = self.func;
        let rhs_func = rhs.func;
        TensorFunc {
            data: Rc::new(self.data.as_ref() + rhs.data.as_ref()),
            func: move |res_grad: S| {
                // With impl FnOnce(Rc<S>) this is not a copy, but hey, advantages from passing S
                // by value and potentially doing operations in place are bigger than this copy
                // (at least hope so)
                self_func(res_grad.clone());
                rhs_func(res_grad);
            },
        }
    }
}