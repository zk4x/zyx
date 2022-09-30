use crate::{tensor::{Tensor, TensorGrad, TensorFunc, Backward}};
use std::{rc::Rc, ops::Add, cell::RefCell};

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

#[derive(Debug)]
pub struct AddBackwardFG<'g, S, XF> {
    xfunc: XF,
    ygrad: &'g RefCell<S>,
}

impl<'g, S, XF> Backward<S> for AddBackwardFG<'g, S, XF>
where
    for<'a> &'a S: Add<Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_with(|grad| &*grad + &res_grad);
        self.xfunc.backward(res_grad);
    }
}

impl<'g, S, XF> Add<&'g TensorGrad<S>> for TensorFunc<S, XF>
where
    S: 'g,
    for<'a> &'a S: Add<Output = S>,
{
    type Output = TensorFunc<S, AddBackwardFG<'g, S, XF>>;
    fn add(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.as_ref() + rhs.data().as_ref()),
            func: AddBackwardFG {
                xfunc: self.func,
                ygrad: &rhs.grad,
            },
        }
    }
}

#[derive(Debug)]
pub struct AddBackwardFF<XF, YF> {
    xfunc: XF,
    yfunc: YF,
}

impl<S, XF, YF> Backward<S> for AddBackwardFF<XF, YF>
where
    S: Clone,
    for<'a> &'a S: Add<Output = S>,
    XF: Backward<S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        // If res_grad is &S this is not a copy, but hey, advantages from passing S
        // by value and potentially doing operations in place are bigger than this copy
        // (at least hope so), if not, this will be changed
        self.xfunc.backward(res_grad.clone());
        self.yfunc.backward(res_grad);
    }
}

impl<S, XF, YF> Add<TensorFunc<S, YF>> for TensorFunc<S, XF>
where
    for<'a> &'a S: Add<Output = S>,
{
    type Output = TensorFunc<S, AddBackwardFF<XF, YF>>;
    fn add(self, rhs: TensorFunc<S, YF>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.as_ref() + rhs.data.as_ref()),
            func: AddBackwardFF {
                xfunc: self.func,
                yfunc: rhs.func,
            }
        }
    }
}