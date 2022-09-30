use crate::{ops::{MatMul, Transpose}, tensor::{Tensor, TensorGrad, TensorFunc, Backward}};
use std::{rc::Rc, ops::Add, cell::RefCell};

impl<S> MatMul<Tensor<S>> for Tensor<S>
where
    for<'a> &'a S: MatMul<Output = S>,
{
    type Output = Tensor<S>;
    fn matmul(self, rhs: Tensor<S>) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.matmul(&rhs.data)),
        }
    }
}

#[derive(Debug)]
pub struct MatMulBackwardTG<'g, S> {
    xdata: Rc<S>,
    ygrad: &'g RefCell<S>,
}

impl<'g, S> Clone for MatMulBackwardTG<'g, S> {
    fn clone(&self) -> Self {
        Self {
            xdata: Rc::clone(&self.xdata),
            ygrad: self.ygrad,
        }
    }
}

impl<'g, S> Backward<S> for MatMulBackwardTG<'g, S>
where
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_with(|grad| &*grad + &self.xdata.transpose().matmul(&res_grad));
    }
}

impl<'g, S> MatMul<&'g TensorGrad<S>> for Tensor<S>
where
    S: 'g,
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    type Output = TensorFunc<S, MatMulBackwardTG<'g, S>>;
    fn matmul(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.matmul(&rhs.data())),
            func: MatMulBackwardTG {
                xdata: self.data,
                ygrad: &rhs.grad,
            }
        }
    }
}

#[derive(Debug)]
pub struct MatMulBackwardTF<S, F> {
    xdata: Rc<S>,
    yfunc: F,
}

impl<S, F> Clone for MatMulBackwardTF<S, F>
where
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            xdata: Rc::clone(&self.xdata),
            yfunc: self.yfunc.clone(),
        }
    }
}

impl<S, F> Backward<S> for MatMulBackwardTF<S, F>
where
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.yfunc.backward(self.xdata.transpose().matmul(&res_grad));
    }
}

impl<S, F> MatMul<TensorFunc<S, F>> for Tensor<S>
where
    for<'a> &'a S: MatMul<Output = S>,
{
    type Output = TensorFunc<S, MatMulBackwardTF<S, F>>;
    fn matmul(self, rhs: TensorFunc<S, F>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.matmul(&rhs.data)),
            func: MatMulBackwardTF {
                xdata: self.data,
                yfunc: rhs.func,
            }
        }
    }
}

#[derive(Debug)]
pub struct MatMulBackwardGT<'g, S> {
    xgrad: &'g RefCell<S>,
    ydata: Rc<S>,
}

impl<'g, S> Clone for MatMulBackwardGT<'g, S> {
    fn clone(&self) -> Self {
        Self {
            xgrad: self.xgrad,
            ydata: Rc::clone(&self.ydata),
        }
    }
}

impl<'g, S> Backward<S> for MatMulBackwardGT<'g, S>
where
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_with(|grad| &*grad + &res_grad.matmul(&self.ydata.transpose()));
    }
}

impl<'g, S> MatMul<Tensor<S>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: MatMul<Output = S>,
{
    type Output = TensorFunc<S, MatMulBackwardGT<'g, S>>;
    fn matmul(self, rhs: Tensor<S>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.borrow().matmul(&rhs.data)),
            func: MatMulBackwardGT {
                xgrad: &self.grad,
                ydata: rhs.data,
            }
        }
    }
}

#[derive(Debug)]
pub struct MatMulBackwardGG<'g, S> {
    xgrad: &'g RefCell<S>,
    xdata: Rc<S>,
    ygrad: &'g RefCell<S>,
    ydata: Rc<S>,
}

impl<'g, S> Clone for MatMulBackwardGG<'g, S> {
    fn clone(&self) -> Self {
        Self {
            xgrad: self.xgrad,
            xdata: Rc::clone(&self.xdata),
            ygrad: self.ygrad,
            ydata: Rc::clone(&self.ydata),
        }
    }
}

impl<'g, S> Backward<S> for MatMulBackwardGG<'g, S>
where
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_with(|grad| &*grad + &self.xdata.transpose().matmul(&res_grad));
        self.xgrad.replace_with(|grad| &*grad + &res_grad.matmul(&self.ydata.transpose()));
    }
}

impl<'g, S> MatMul<&'g TensorGrad<S>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: MatMul<Output = S>,
{
    type Output = TensorFunc<S, MatMulBackwardGG<'g, S>>;
    fn matmul(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.borrow().matmul(&rhs.data.borrow())),
            func: MatMulBackwardGG {
                xgrad: &self.grad,
                xdata: Rc::clone(&self.data.borrow()),
                ygrad: &rhs.grad,
                ydata: Rc::clone(&rhs.data.borrow()),
            }
        }
    }
}

#[derive(Debug)]
//#[derivative(Clone(bound=""))]
pub struct MatMulBackwardGF<'g, S, YF> {
    xgrad: &'g RefCell<S>,
    xdata: Rc<S>,
    yfunc: YF,
    ydata: Rc<S>,
}

impl<'g, S, YF> Backward<S> for MatMulBackwardGF<'g, S, YF>
where
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.yfunc.backward(self.xdata.transpose().matmul(&res_grad));
        self.xgrad.replace_with(|grad| &*grad + &res_grad.transpose().matmul(&self.ydata));
    }
}

impl<'g, S, F> MatMul<TensorFunc<S, F>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    type Output = TensorFunc<S, MatMulBackwardGF<'g, S, F>>;
    fn matmul(self, rhs: TensorFunc<S, F>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.borrow().matmul(&rhs.data)),
            func: MatMulBackwardGF {
                xgrad: &self.grad,
                xdata: Rc::clone(&self.data.borrow()),
                yfunc: rhs.func,
                ydata: rhs.data,
            }
        }
    }
}

#[derive(Debug)]
pub struct MatMulBackwardFT<S, XF> {
    xfunc: XF,
    ydata: Rc<S>,
}

impl<S, XF> Backward<S> for MatMulBackwardFT<S, XF>
where
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xfunc.backward(self.ydata.transpose().matmul(&res_grad));
    }
}

impl<S, F> MatMul<Tensor<S>> for TensorFunc<S, F>
where
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    type Output = TensorFunc<S, MatMulBackwardFT<S, F>>;
    fn matmul(self, rhs: Tensor<S>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.matmul(&rhs.data)),
            func: MatMulBackwardFT {
                xfunc: self.func,
                ydata: rhs.data,
            }
        }
    }
}

#[derive(Debug)]
pub struct MatMulBackwardFG<'g, S, XF> {
    xfunc: XF,
    xdata: Rc<S>,
    ygrad: &'g RefCell<S>,
    ydata: Rc<S>,
}

impl<'g, S, XF> Backward<S> for MatMulBackwardFG<'g, S, XF>
where
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xfunc.backward(res_grad.matmul(&self.ydata.transpose()));
        self.ygrad.replace_with(|grad| &*grad + &self.xdata.transpose().matmul(&res_grad));
    }
}

impl<'g, S, F> MatMul<&'g TensorGrad<S>> for TensorFunc<S, F>
where
    S: 'g,
    for<'a> &'a S: MatMul<Output = S>,
{
    type Output = TensorFunc<S, MatMulBackwardFG<'g, S, F>>;
    fn matmul(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.matmul(&rhs.data.borrow())),
            func: MatMulBackwardFG {
                xfunc: self.func,
                xdata: self.data,
                ygrad: &rhs.grad,
                ydata: Rc::clone(&rhs.data.borrow()),
            }
        }
    }
}

#[derive(Debug)]
pub struct MatMulBackwardFF<S, XF, YF> {
    xfunc: XF,
    xdata: Rc<S>,
    yfunc: YF,
    ydata: Rc<S>,
}

impl<S, XF, YF> Backward<S> for MatMulBackwardFF<S, XF, YF>
where
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
    XF: Backward<S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xfunc.backward(res_grad.matmul(&self.ydata.transpose()));
        self.yfunc.backward(self.xdata.transpose().matmul(&res_grad));
    }
}

impl<S, XF, YF> MatMul<TensorFunc<S, YF>> for TensorFunc<S, XF>
where
    for<'a> &'a S: MatMul<Output = S>,
{
    type Output = TensorFunc<S, MatMulBackwardFF<S, XF, YF>>;
    fn matmul(self, rhs: TensorFunc<S, YF>) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.matmul(&rhs.data)),
            func: MatMulBackwardFF {
                xfunc: self.func,
                xdata: self.data,
                yfunc: rhs.func,
                ydata: rhs.data,
            }
        }
    }
}
