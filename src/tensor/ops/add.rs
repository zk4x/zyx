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

#[derive(Debug, Clone)]
pub struct AddBackwardTG<'g, S> {
    ygrad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for AddBackwardTG<'g, S>
where
    for<'a> &'a S: Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_with(|grad| &*grad + &res_grad);
    }
}

impl<'g, S> Add<&'g TensorGrad<S>> for Tensor<S>
where
    S: 'g,
    for<'a> &'a S: Add<Output = S>,
{
    type Output = TensorFunc<S, AddBackwardTG<'g, S>>;
    fn add(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.as_ref() + rhs.data().as_ref()),
            func: AddBackwardTG {
                ygrad: &rhs.grad,
            }
        }
    }
}

impl<S, F> Add<TensorFunc<S, F>> for Tensor<S>
where
    F: FnOnce(S),
    for<'a> &'a S: Add<Output = S>,
{
    type Output = TensorFunc<S, F>;
    fn add(self, rhs: TensorFunc<S, F>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.as_ref() + rhs.data.as_ref()),
            func: rhs.func,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AddBackwardGT<'g, S> {
    xgrad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for AddBackwardGT<'g, S>
where
    for<'a> &'a S: Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_with(|grad| &*grad + &res_grad);
    }
}

impl<'g, S> Add<Tensor<S>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Add<Output = S>,
{
    type Output = TensorFunc<S, AddBackwardGT<'g, S>>;
    fn add(self, rhs: Tensor<S>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data().as_ref() + rhs.data.as_ref()),
            func: AddBackwardGT {
                xgrad: &self.grad,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct AddBackwardGG<'g, S> {
    xgrad: &'g RefCell<S>,
    ygrad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for AddBackwardGG<'g, S>
where
    for<'a> &'a S: Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_with(|grad| &*grad + &res_grad);
        self.ygrad.replace_with(|grad| &*grad + &res_grad);
    }
}

impl<'g, S> Add<&'g TensorGrad<S>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Add<Output = S>,
{
    type Output = TensorFunc<S, AddBackwardGG<'g, S>>;
    fn add(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data().as_ref() + rhs.data().as_ref()),
            func: AddBackwardGG {
                xgrad: &self.grad,
                ygrad: &rhs.grad,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct AddBackwardGF<'g, S, YF> {
    xgrad: &'g RefCell<S>,
    yfunc: YF,
}

impl<'g, S, YF> Backward<S> for AddBackwardGF<'g, S, YF>
where
    for<'a> &'a S: Add<Output = S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_with(|grad| &*grad + &res_grad);
        self.yfunc.backward(res_grad);
    }
}

impl<'g, S, F> Add<TensorFunc<S, F>> for &'g TensorGrad<S>
where
    S: 'g,
    F: FnOnce(S),
    for<'a> &'a S: Add<Output = S>,
{
    type Output = TensorFunc<S, AddBackwardGF<'g, S, F>>;
    fn add(self, rhs: TensorFunc<S, F>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data().as_ref() + rhs.data.as_ref()),
            func: AddBackwardGF {
                xgrad: &self.grad,
                yfunc: rhs.func,
            }
        }
    }
}

impl<S, F> Add<Tensor<S>> for TensorFunc<S, F>
where
    for<'a> &'a S: Add<Output = S>,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, F>;
    fn add(self, rhs: Tensor<S>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.as_ref() + rhs.data.as_ref()),
            func: self.func,
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