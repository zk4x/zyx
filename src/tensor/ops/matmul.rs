use crate::{ops::{MatMul, Transpose}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardSV<'g, S> {
    xdata: S,
    ygrad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for MatMulBackwardSV<'g, S>
where
    S: Default + MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad + self.xdata.transpose().matmul(res_grad));
    }
}

impl<'g, S> MatMul<&'g Variable<S>> for S
where
    S: 'g + Clone + MatMul<Output = S>,
{
    type Output = Tensor<S, MatMulBackwardSV<'g, S>>;
    fn matmul(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.clone().matmul(rhs.data().clone()),
            grad_fn: MatMulBackwardSV {
                xdata: self,
                ygrad: &rhs.grad,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardST<S, F> {
    xdata: S,
    ygrad_fn: F,
}

impl<S, F> Backward<S> for MatMulBackwardST<S, F>
where
    S: MatMul<Output = S> + Transpose<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad_fn.backward(self.xdata.transpose().matmul(res_grad));
    }
}

impl<S, F> MatMul<Tensor<S, F>> for S
where
    S: Clone + MatMul<Output = S>,
{
    type Output = Tensor<S, MatMulBackwardST<S, F>>;
    fn matmul(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self.clone().matmul(rhs.data),
            grad_fn: MatMulBackwardST {
                xdata: self,
                ygrad_fn: rhs.grad_fn,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardVS<'g, S> {
    xgrad: &'g RefCell<S>,
    ydata: S,
}

impl<'g, S> Backward<S> for MatMulBackwardVS<'g, S>
where
    S: Default + MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad.matmul(self.ydata.transpose()));
    }
}

impl<'g, S> MatMul<S> for &'g Variable<S>
where
    S: 'g + Clone + MatMul<Output = S>,
{
    type Output = Tensor<S, MatMulBackwardVS<'g, S>>;
    fn matmul(self, rhs: S) -> Self::Output {
        Tensor {
            data: self.data.borrow().clone().matmul(rhs.clone()),
            grad_fn: MatMulBackwardVS {
                xgrad: &self.grad,
                ydata: rhs,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardVV<'g, S> {
    xgrad: &'g RefCell<S>,
    xdata: S,
    ygrad: &'g RefCell<S>,
    ydata: S,
}

impl<'g, S> Backward<S> for MatMulBackwardVV<'g, S>
where
    S: Default + Clone + MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad + self.xdata.transpose().matmul(res_grad.clone()));
        self.xgrad.replace_take(|grad| grad + res_grad.matmul(self.ydata.transpose()));
    }
}

impl<'g, S> MatMul<&'g Variable<S>> for &'g Variable<S>
where
    S: 'g + Clone + MatMul<Output = S>,
{
    type Output = Tensor<S, MatMulBackwardVV<'g, S>>;
    fn matmul(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.data().clone().matmul(rhs.data().clone()),
            grad_fn: MatMulBackwardVV {
                xgrad: &self.grad,
                xdata: self.data().clone(),
                ygrad: &rhs.grad,
                ydata: rhs.data().clone(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardVT<'g, S, YF> {
    xgrad: &'g RefCell<S>,
    xdata: S,
    ygrad_fn: YF,
    ydata: S,
}

impl<'g, S, YF> Backward<S> for MatMulBackwardVT<'g, S, YF>
where
    S: Default + Clone + MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad_fn.backward(self.xdata.transpose().matmul(res_grad.clone()));
        self.xgrad.replace_take(|grad| grad + res_grad.transpose().matmul(self.ydata));
    }
}

impl<'g, S, F> MatMul<Tensor<S, F>> for &'g Variable<S>
where
    S: 'g + Clone + MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    type Output = Tensor<S, MatMulBackwardVT<'g, S, F>>;
    fn matmul(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self.data().clone().matmul(rhs.data.clone()),
            grad_fn: MatMulBackwardVT {
                xgrad: &self.grad,
                xdata: self.data().clone(),
                ygrad_fn: rhs.grad_fn,
                ydata: rhs.data,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardTS<S, XF> {
    xgrad_fn: XF,
    ydata: S,
}

impl<S, XF> Backward<S> for MatMulBackwardTS<S, XF>
where
    S: MatMul<Output = S> + Transpose<Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(self.ydata.transpose().matmul(res_grad));
    }
}

impl<S, F> MatMul<S> for Tensor<S, F>
where
    S: Clone + MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    type Output = Tensor<S, MatMulBackwardTS<S, F>>;
    fn matmul(self, rhs: S) -> Self::Output {
        Tensor {
            data: self.data.matmul(rhs.clone()),
            grad_fn: MatMulBackwardTS {
                xgrad_fn: self.grad_fn,
                ydata: rhs,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardTV<'g, S, XF> {
    xgrad_fn: XF,
    xdata: S,
    ygrad: &'g RefCell<S>,
    ydata: S,
}

impl<'g, S, XF> Backward<S> for MatMulBackwardTV<'g, S, XF>
where
    S: Default + Clone + MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(res_grad.clone().matmul(self.ydata.transpose()));
        self.ygrad.replace_take(|grad| grad + self.xdata.transpose().matmul(res_grad));
    }
}

impl<'g, S, F> MatMul<&'g Variable<S>> for Tensor<S, F>
where
    S: 'g + Clone + MatMul<Output = S>,
{
    type Output = Tensor<S, MatMulBackwardTV<'g, S, F>>;
    fn matmul(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.data.clone().matmul(rhs.data().clone()),
            grad_fn: MatMulBackwardTV {
                xgrad_fn: self.grad_fn,
                xdata: self.data,
                ygrad: &rhs.grad,
                ydata: rhs.data().clone(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardTT<S, XF, YF> {
    xgrad_fn: XF,
    xdata: S,
    ygrad_fn: YF,
    ydata: S,
}

impl<S, XF, YF> Backward<S> for MatMulBackwardTT<S, XF, YF>
where
    S: Clone + MatMul<Output = S> + Transpose<Output = S>,
    XF: Backward<S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(res_grad.clone().matmul(self.ydata.transpose()));
        self.ygrad_fn.backward(self.xdata.transpose().matmul(res_grad));
    }
}

impl<S, XF, YF> MatMul<Tensor<S, YF>> for Tensor<S, XF>
where
    S: Clone + MatMul<Output = S>,
{
    type Output = Tensor<S, MatMulBackwardTT<S, XF, YF>>;
    fn matmul(self, rhs: Tensor<S, YF>) -> Self::Output {
        Tensor {
            data: self.data.clone().matmul(rhs.data.clone()),
            grad_fn: MatMulBackwardTT {
                xgrad_fn: self.grad_fn,
                xdata: self.data,
                ygrad_fn: rhs.grad_fn,
                ydata: rhs.data,
            }
        }
    }
}
