use crate::{ops::{Pow, Ln}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{cell::RefCell, ops::{Add, Mul, Div}};

#[derive(Debug, Clone, Copy)]
pub struct PowBackwardSV<'g, S> {
    ygrad: &'g RefCell<S>,
    ytemp: S,
}

impl<'g, S> Backward<S> for PowBackwardSV<'g, S>
where
    S: Default + Add<Output = S> + Mul<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad + res_grad * self.ytemp);
    }
}

impl<'g, S> Pow<&'g Variable<S>> for S
where
    S: 'g + Clone + Pow<Output = S> + Mul<Output = S> + Ln<Output = S>,
{
    type Output = Tensor<S, PowBackwardSV<'g, S>>;
    fn pow(self, rhs: &'g Variable<S>) -> Self::Output {
        let res = self.clone().pow(rhs.data().clone());
        Tensor {
            data: res.clone(),
            grad_fn: PowBackwardSV {
                ygrad: &rhs.grad,
                ytemp: res * self.ln(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PowBackwardST<S, YF> {
    ygrad_fn: YF,
    ytemp: S,
}

impl<YF, S> Backward<S> for PowBackwardST<S, YF>
where
    S: Default + Add<Output = S> + Mul<Output = S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad_fn.backward(res_grad * self.ytemp);
    }
}

impl<S, F> Pow<Tensor<S, F>> for S
where
    S: Clone + Pow<Output = S> + Ln<Output = S> + Mul<Output = S>,
{
    type Output = Tensor<S, PowBackwardST<S, F>>;
    fn pow(self, rhs: Tensor<S, F>) -> Self::Output {
        let res = self.clone().pow(rhs.data);
        Tensor {
            data: res.clone(),
            grad_fn: PowBackwardST {
                ygrad_fn: rhs.grad_fn,
                ytemp: res * self.ln(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PowBackwardVS<'g, S> {
    xgrad: &'g RefCell<S>,
    xtemp: S,
}

impl<'g, S> Backward<S> for PowBackwardVS<'g, S>
where
    S: Default + Add<Output = S> + Mul<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + self.xtemp * res_grad);
    }
}

impl<'g, S> Pow<S> for &'g Variable<S>
where
    S: 'g + Clone + Mul<Output = S> + Div<Output = S> + Pow<Output = S>,
{
    type Output = Tensor<S, PowBackwardVS<'g, S>>;
    fn pow(self, rhs: S) -> Self::Output {
        let res = self.data().clone().pow(rhs.clone());
        Tensor {
            data: res.clone(),
            grad_fn: PowBackwardVS {
                xgrad: &self.grad,
                xtemp: rhs * res/self.data().clone(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PowBackwardVV<'g, S> {
    xgrad: &'g RefCell<S>,
    xtemp: S,
    ygrad: &'g RefCell<S>,
    ytemp: S,
}

impl<'g, S> Backward<S> for PowBackwardVV<'g, S>
where
    S: Default + Clone + Add<Output = S> + Mul<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + self.xtemp * res_grad.clone());
        self.ygrad.replace_take(|grad| grad + self.ytemp * res_grad);
    }
}

impl<'g, S> Pow<&'g Variable<S>> for &'g Variable<S>
where
    S: 'g + Clone + Mul<Output = S> + Div<Output = S> + Pow<Output = S> + Ln<Output = S>,
{
    type Output = Tensor<S, PowBackwardVV<'g, S>>;
    fn pow(self, rhs: &'g Variable<S>) -> Self::Output {
        let res = self.data().clone().pow(rhs.data().clone());
        Tensor {
            data: res.clone(),
            grad_fn: PowBackwardVV {
                xgrad: &self.grad,
                xtemp: rhs.data().clone() * res.clone()/self.data().clone(),
                ygrad: &rhs.grad,
                ytemp: res * self.data().clone().ln(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PowBackwardVT<'g, S, YF> {
    xgrad: &'g RefCell<S>,
    xtemp: S,
    ygrad_fn: YF,
    ytemp: S,
}

impl<'g, S, YF> Backward<S> for PowBackwardVT<'g, S, YF>
where
    S: Default + Clone + Add<Output = S> + Mul<Output = S> + Pow<Output = S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + self.xtemp * res_grad.clone());
        self.ygrad_fn.backward(res_grad * self.ytemp);
    }
}

impl<'g, S, F> Pow<Tensor<S, F>> for &'g Variable<S>
where
    S: 'g + Clone + Mul<Output = S> + Div<Output = S> + Pow<Output = S> + Ln<Output = S>,
{
    type Output = Tensor<S, PowBackwardVT<'g, S, F>>;
    fn pow(self, rhs: Tensor<S, F>) -> Self::Output {
        let res = self.data().clone().pow(rhs.data.clone());
        Tensor {
            data: res.clone(),
            grad_fn: PowBackwardVT {
                xgrad: &self.grad,
                xtemp: rhs.data * res.clone()/self.data().clone(),
                ygrad_fn: rhs.grad_fn,
                ytemp: res * self.data().clone().ln(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PowBackwardTS<S, XF> {
    xgrad_fn: XF,
    xtemp: S,
}

impl<S, XF> Backward<S> for PowBackwardTS<S, XF>
where
    S: Default + Mul<Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(self.xtemp * res_grad);
    }
}

impl<S, F> Pow<S> for Tensor<S, F>
where
    S: Clone + Mul<Output = S> + Div<Output = S> + Pow<Output = S>,
{
    type Output = Tensor<S, PowBackwardTS<S, F>>;
    fn pow(self, rhs: S) -> Self::Output {
        let res = self.data.clone().pow(rhs.clone());
        Tensor {
            data: res.clone(),
            grad_fn: PowBackwardTS {
                xgrad_fn: self.grad_fn,
                xtemp: rhs * res/self.data,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PowBackwardTV<'g, S, XF> {
    xgrad_fn: XF,
    xtemp: S,
    ygrad: &'g RefCell<S>,
    ytemp: S,
}

impl<'g, S, XF> Backward<S> for PowBackwardTV<'g, S, XF>
where
    S: Default + Clone + Add<Output = S> + Mul<Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad + res_grad.clone() * self.ytemp);
        self.xgrad_fn.backward(self.xtemp * res_grad);
    }
}

impl<'g, S, XF> Pow<&'g Variable<S>> for Tensor<S, XF>
where
    S: 'g + Clone + Mul<Output = S> + Div<Output = S> + Pow<Output = S> + Ln<Output = S>,
{
    type Output = Tensor<S, PowBackwardTV<'g, S, XF>>;
    fn pow(self, rhs: &'g Variable<S>) -> Self::Output {
        let res = self.data.clone().pow(rhs.data().clone());
        Tensor {
            data: res.clone(),
            grad_fn: PowBackwardTV {
                xgrad_fn: self.grad_fn,
                xtemp: rhs.data().clone() * res.clone()/self.data.clone(),
                ygrad: &rhs.grad,
                ytemp: res * self.data.ln(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PowBackwardTT<S, XF, YF> {
    xgrad_fn: XF,
    xtemp: S,
    ygrad_fn: YF,
    ytemp: S,
}

impl<S, XF, YF> Backward<S> for PowBackwardTT<S, XF, YF>
where
    S: Clone + Mul<Output = S>,
    XF: Backward<S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad_fn.backward(self.ytemp * res_grad.clone());
        self.xgrad_fn.backward(self.xtemp * res_grad);
    }
}

impl<S, XF, YF> Pow<Tensor<S, YF>> for Tensor<S, XF>
where
    S: Clone + Mul<Output = S> + Div<Output = S> + Pow<Output = S> + Ln<Output = S>,
{
    type Output = Tensor<S, PowBackwardTT<S, XF, YF>>;
    fn pow(self, rhs: Tensor<S, YF>) -> Self::Output {
        let res = self.data.clone().pow(rhs.data.clone());
        Tensor {
            data: res.clone(),
            grad_fn: PowBackwardTT {
                xgrad_fn: self.grad_fn,
                xtemp: rhs.data().clone() * res.clone()/self.data.clone(),
                ygrad_fn: rhs.grad_fn,
                ytemp: res * self.data.ln(),
            }
        }
    }
}
