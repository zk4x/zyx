use crate::{ops::{MatMul, Transpose}, tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}};
use std::{ops::Add, cell::RefCell};

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardTG<'g, S> {
    xdata: S,
    ygrad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for MatMulBackwardTG<'g, S>
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
    type Output = Tensor<S, MatMulBackwardTG<'g, S>>;
    fn matmul(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.clone().matmul(rhs.data().clone()),
            func: MatMulBackwardTG {
                xdata: self,
                ygrad: &rhs.grad,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardTF<S, F> {
    xdata: S,
    yfunc: F,
}

impl<S, F> Backward<S> for MatMulBackwardTF<S, F>
where
    S: MatMul<Output = S> + Transpose<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.yfunc.backward(self.xdata.transpose().matmul(res_grad));
    }
}

impl<S, F> MatMul<Tensor<S, F>> for S
where
    S: Clone + MatMul<Output = S>,
{
    type Output = Tensor<S, MatMulBackwardTF<S, F>>;
    fn matmul(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self.clone().matmul(rhs.data),
            func: MatMulBackwardTF {
                xdata: self,
                yfunc: rhs.func,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardGT<'g, S> {
    xgrad: &'g RefCell<S>,
    ydata: S,
}

impl<'g, S> Backward<S> for MatMulBackwardGT<'g, S>
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
    type Output = Tensor<S, MatMulBackwardGT<'g, S>>;
    fn matmul(self, rhs: S) -> Self::Output {
        Tensor {
            data: self.data.borrow().clone().matmul(rhs.clone()),
            func: MatMulBackwardGT {
                xgrad: &self.grad,
                ydata: rhs,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardGG<'g, S> {
    xgrad: &'g RefCell<S>,
    xdata: S,
    ygrad: &'g RefCell<S>,
    ydata: S,
}

impl<'g, S> Backward<S> for MatMulBackwardGG<'g, S>
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
    type Output = Tensor<S, MatMulBackwardGG<'g, S>>;
    fn matmul(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.data().clone().matmul(rhs.data().clone()),
            func: MatMulBackwardGG {
                xgrad: &self.grad,
                xdata: self.data().clone(),
                ygrad: &rhs.grad,
                ydata: rhs.data().clone(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardGF<'g, S, YF> {
    xgrad: &'g RefCell<S>,
    xdata: S,
    yfunc: YF,
    ydata: S,
}

impl<'g, S, YF> Backward<S> for MatMulBackwardGF<'g, S, YF>
where
    S: Default + Clone + MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.yfunc.backward(self.xdata.transpose().matmul(res_grad.clone()));
        self.xgrad.replace_take(|grad| grad + res_grad.transpose().matmul(self.ydata));
    }
}

impl<'g, S, F> MatMul<Tensor<S, F>> for &'g Variable<S>
where
    S: 'g + Clone + MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    type Output = Tensor<S, MatMulBackwardGF<'g, S, F>>;
    fn matmul(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self.data().clone().matmul(rhs.data.clone()),
            func: MatMulBackwardGF {
                xgrad: &self.grad,
                xdata: self.data().clone(),
                yfunc: rhs.func,
                ydata: rhs.data,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardFT<S, XF> {
    xfunc: XF,
    ydata: S,
}

impl<S, XF> Backward<S> for MatMulBackwardFT<S, XF>
where
    S: MatMul<Output = S> + Transpose<Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xfunc.backward(self.ydata.transpose().matmul(res_grad));
    }
}

impl<S, F> MatMul<S> for Tensor<S, F>
where
    S: Clone + MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    type Output = Tensor<S, MatMulBackwardFT<S, F>>;
    fn matmul(self, rhs: S) -> Self::Output {
        Tensor {
            data: self.data.matmul(rhs.clone()),
            func: MatMulBackwardFT {
                xfunc: self.func,
                ydata: rhs,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardFG<'g, S, XF> {
    xfunc: XF,
    xdata: S,
    ygrad: &'g RefCell<S>,
    ydata: S,
}

impl<'g, S, XF> Backward<S> for MatMulBackwardFG<'g, S, XF>
where
    S: Default + Clone + MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xfunc.backward(res_grad.clone().matmul(self.ydata.transpose()));
        self.ygrad.replace_take(|grad| grad + self.xdata.transpose().matmul(res_grad));
    }
}

impl<'g, S, F> MatMul<&'g Variable<S>> for Tensor<S, F>
where
    S: 'g + Clone + MatMul<Output = S>,
{
    type Output = Tensor<S, MatMulBackwardFG<'g, S, F>>;
    fn matmul(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.data.clone().matmul(rhs.data().clone()),
            func: MatMulBackwardFG {
                xfunc: self.func,
                xdata: self.data,
                ygrad: &rhs.grad,
                ydata: rhs.data().clone(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardFF<S, XF, YF> {
    xfunc: XF,
    xdata: S,
    yfunc: YF,
    ydata: S,
}

impl<S, XF, YF> Backward<S> for MatMulBackwardFF<S, XF, YF>
where
    S: Clone + MatMul<Output = S> + Transpose<Output = S>,
    XF: Backward<S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xfunc.backward(res_grad.clone().matmul(self.ydata.transpose()));
        self.yfunc.backward(self.xdata.transpose().matmul(res_grad));
    }
}

impl<S, XF, YF> MatMul<Tensor<S, YF>> for Tensor<S, XF>
where
    S: Clone + MatMul<Output = S>,
{
    type Output = Tensor<S, MatMulBackwardFF<S, XF, YF>>;
    fn matmul(self, rhs: Tensor<S, YF>) -> Self::Output {
        Tensor {
            data: self.data.clone().matmul(rhs.data.clone()),
            func: MatMulBackwardFF {
                xfunc: self.func,
                xdata: self.data,
                yfunc: rhs.func,
                ydata: rhs.data,
            }
        }
    }
}
