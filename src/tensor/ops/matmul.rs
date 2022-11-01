use crate::{ops::{MatMul, Transpose}, tensor::{Variable, Tensor, Backward, Gradient}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardSV<'g, XS, YG> {
    xdata: XS,
    ygrad: &'g Gradient<YG>,
}

impl<S, XS, YG> Backward<S> for MatMulBackwardSV<'_, XS, YG>
where
    XS: Transpose,
    <XS as Transpose>::Output: MatMul<S, Output = YG>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.accumulate(self.xdata.transpose().matmul(res_grad));
    }
}

impl<'g, XS, YS, YG> MatMul<&'g Variable<YS, YG>> for XS
where
    XS: Clone + MatMul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as MatMul<YS>>::Output, MatMulBackwardSV<'g, XS, YG>>;
    fn matmul(self, rhs: &'g Variable<YS, YG>) -> Self::Output {
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
pub struct MatMulBackwardST<XS, YF> {
    xdata: XS,
    ygrad_fn: YF,
}

impl<S, XS, YF> Backward<S> for MatMulBackwardST<XS, YF>
where
    XS: Transpose,
    <XS as Transpose>::Output: MatMul<S>,
    YF: Backward<<<XS as Transpose>::Output as MatMul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.ygrad_fn.backward(self.xdata.transpose().matmul(res_grad));
    }
}

impl<XS, YS, YF> MatMul<Tensor<YS, YF>> for XS
where
    XS: Clone + MatMul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as MatMul<YS>>::Output, MatMulBackwardST<XS, YF>>;
    fn matmul(self, rhs: Tensor<YS, YF>) -> Self::Output {
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
pub struct MatMulBackwardVS<'g, XG, YS> {
    xgrad: &'g Gradient<XG>,
    ydata: YS,
}

impl<S, XG, YS> Backward<S> for MatMulBackwardVS<'_, XG, YS>
where
    YS: Transpose,
    S: MatMul<<YS as Transpose>::Output, Output = XG>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(res_grad.matmul(self.ydata.transpose()));
    }
}

impl<'g, XS, XG, YS> MatMul<YS> for &'g Variable<XS, XG>
where
    XS: Clone + MatMul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as MatMul<YS>>::Output, MatMulBackwardVS<'g, XG, YS>>;
    fn matmul(self, rhs: YS) -> Self::Output {
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
pub struct MatMulBackwardVV<'g, XS, XG, YS, YG> {
    xgrad: &'g Gradient<XG>,
    xdata: XS,
    ygrad: &'g Gradient<YG>,
    ydata: YS,
}

impl<S, XS, XG, YS, YG> Backward<S> for MatMulBackwardVV<'_, XS, XG, YS, YG>
where
{
    fn backward(self, res_grad: S) {
        self.ygrad.accumulate(self.xdata.transpose().matmul(res_grad.clone()));
        self.xgrad.accumulate(res_grad.matmul(self.ydata.transpose()));
    }
}

impl<'g, XS, XG, YS, YG> MatMul<&'g Variable<YS, YG>> for &'g Variable<XS, XG>
where
    XS: Clone + MatMul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as MatMul<YS>>::Output, MatMulBackwardVV<'g, XS, XG, YS, YG>>;
    fn matmul(self, rhs: &'g Variable<YS, YG>) -> Self::Output {
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
pub struct MatMulBackwardVT<'g, XG, XS, YS, YF> {
    xgrad: &'g Gradient<XG>,
    xdata: XS,
    ygrad_fn: YF,
    ydata: YS,
}

impl<S, XG, XS, YS, YF> Backward<S> for MatMulBackwardVT<'_, XG, XS, YS, YF>
where
    YF: Backward<()>,
{
    fn backward(self, res_grad: S) {
        self.ygrad_fn.backward(self.xdata.transpose().matmul(res_grad.clone()));
        self.xgrad.accumulate(res_grad.transpose().matmul(self.ydata));
    }
}

impl<'g, XS, XG, YS, YF> MatMul<Tensor<YS, YF>> for &'g Variable<XS, XG>
where
    XS: Clone + MatMul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as MatMul<YS>>::Output, MatMulBackwardVT<'g, XG, XS, YS, YF>>;
    fn matmul(self, rhs: Tensor<YS, YF>) -> Self::Output {
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
pub struct MatMulBackwardTS<YS, XF> {
    xgrad_fn: XF,
    ydata: YS,
}

impl<S, YS, XF> Backward<S> for MatMulBackwardTS<YS, XF>
where
    XF: Backward<()>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(self.ydata.transpose().matmul(res_grad));
    }
}

impl<XS, YS, XF> MatMul<YS> for Tensor<XS, XF>
where
    XS: MatMul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as MatMul<YS>>::Output, MatMulBackwardTS<YS, XF>>;
    fn matmul(self, rhs: YS) -> Self::Output {
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
pub struct MatMulBackwardTV<'g, YG, XS, YS, XF> {
    xgrad_fn: XF,
    xdata: XS,
    ygrad: &'g Gradient<YG>,
    ydata: YS,
}

impl<S, YG, XS, YS, XF> Backward<S> for MatMulBackwardTV<'_, YG, XS, YS, XF>
where
    XF: Backward<()>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(res_grad.clone().matmul(self.ydata.transpose()));
        self.ygrad.accumulate(self.xdata.transpose().matmul(res_grad));
    }
}

impl<'g, XS, XF, YS, YG> MatMul<&'g Variable<YS, YG>> for Tensor<XS, XF>
where
    XS: Clone + MatMul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as MatMul<YS>>::Output, MatMulBackwardTV<'g, YG, XS, YS, XF>>;
    fn matmul(self, rhs: &'g Variable<YS, YG>) -> Self::Output {
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
pub struct MatMulBackwardTT<XS, YS, XF, YF> {
    xgrad_fn: XF,
    xdata: XS,
    ygrad_fn: YF,
    ydata: YS,
}

impl<S, XS, YS, XF, YF> Backward<S> for MatMulBackwardTT<XS, YS, XF, YF>
where
    XF: Backward<()>,
    YF: Backward<()>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(res_grad.clone().matmul(self.ydata.transpose()));
        self.ygrad_fn.backward(self.xdata.transpose().matmul(res_grad));
    }
}

impl<XS, YS, XF, YF> MatMul<Tensor<YS, YF>> for Tensor<XS, XF>
where
    XS: Clone + MatMul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as MatMul<YS>>::Output, MatMulBackwardTT<XS, YS, XF, YF>>;
    fn matmul(self, rhs: Tensor<YS, YF>) -> Self::Output {
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
