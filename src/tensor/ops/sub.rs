use crate::{tensor::{Tensor, TensorGrad, TensorFunc, Backward}};
use std::{rc::Rc, ops::{Add, Sub, Neg}};

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

#[derive(Debug)]
pub struct SubBackwardFT<XF> {
    xfunc: XF,
}

impl<S, XF> Backward<S> for SubBackwardFT<XF>
where
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xfunc.backward(res_grad);
    }
}

impl<S, XF> Sub<Tensor<S>> for TensorFunc<S, XF>
where
    for<'a> &'a S: Sub<Output = S>,
{
    type Output = TensorFunc<S, SubBackwardFT<XF>>;
    fn sub(self, rhs: Tensor<S>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.as_ref() - rhs.data.as_ref()),
            func: SubBackwardFT {
                xfunc: self.func,
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

#[derive(Debug)]
pub struct SubBackwardFF<XF, YF> {
    xfunc: XF,
    yfunc: YF,
}

impl<S, XF, YF> Backward<S> for SubBackwardFF<XF, YF>
where
    S: Clone,
    for<'a> &'a S: Add<Output = S> + Neg<Output = S>,
    XF: Backward<S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xfunc.backward(-&res_grad);
        self.yfunc.backward(res_grad);
    }
}

impl<S, XF, YF> Sub<TensorFunc<S, YF>> for TensorFunc<S, XF>
where
    for<'a> &'a S: Sub<Output = S>,
    for<'b> &'b S: std::ops::Neg<Output = S>,
{
    type Output = TensorFunc<S, SubBackwardFF<XF, YF>>;
    fn sub(self, rhs: TensorFunc<S, YF>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.as_ref() - rhs.data.as_ref()),
            func: SubBackwardFF {
                xfunc: self.func,
                yfunc: rhs.func,
            },
        }
    }
}