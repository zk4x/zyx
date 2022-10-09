use crate::tensor::{B, Variable, Tensor, Backward, ops::RefCellReplaceTake};
use std::{cell::RefCell, ops::{Add, Mul}};

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardSV<'g, S> {
    xdata: S,
    ygrad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for MulBackwardSV<'g, S>
where
    S: Default + Mul<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad + self.xdata * res_grad);
    }
}

// If you wanted to Mul Variable or Tensor to S, you need to wrap it inside B(),
// but you can Mul S to Variable or Tensor
impl<'g, S> Mul<&'g Variable<S>> for B<S>
where
    S: 'g + Clone + Mul<Output = S>,
{
    type Output = Tensor<S, MulBackwardSV<'g, S>>;
    fn mul(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.0.clone() * rhs.data().clone(),
            func: MulBackwardSV {
                xdata: self.0,
                ygrad: &rhs.grad,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardST<S, YF> {
    xdata: S,
    yfunc: YF,
}

impl<S, YF> Backward<S> for MulBackwardST<S, YF>
where
    S: Mul<Output = S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.yfunc.backward(self.xdata * res_grad);
    }
}

impl<S, F> Mul<Tensor<S, F>> for B<S>
where
    S: Clone + Mul<Output = S>,
{
    type Output = Tensor<S, MulBackwardST<S, F>>;
    fn mul(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self.0.clone() * rhs.data,
            func: MulBackwardST {
                xdata: self.0,
                yfunc: rhs.func,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardVS<'g, S> {
    xgrad: &'g RefCell<S>,
    ydata: S,
}

impl<'g, S> Backward<S> for MulBackwardVS<'g, S>
where
    S: Default + Mul<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + self.ydata * res_grad);
    }
}

impl<'g, S> Mul<S> for &'g Variable<S>
where
    S: 'g + Clone + Mul<Output = S>,
{
    type Output = Tensor<S, MulBackwardVS<'g, S>>;
    fn mul(self, rhs: S) -> Self::Output {
        Tensor {
            data: self.data().clone() * rhs.clone(),
            func: MulBackwardVS {
                xgrad: &self.grad,
                ydata: rhs,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardVV<'g, S> {
    xgrad: &'g RefCell<S>,
    xdata: S,
    ygrad: &'g RefCell<S>,
    ydata: S,
}

impl<'g, S> Backward<S> for MulBackwardVV<'g, S>
where
    S: Default + Clone + Mul<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + self.ydata * res_grad.clone());
        self.ygrad.replace_take(|grad| grad + self.xdata * res_grad);
    }
}

impl<'g, S> Mul<&'g Variable<S>> for &'g Variable<S>
where
    S: 'g + Clone + Mul<Output = S>,
{
    type Output = Tensor<S, MulBackwardVV<'g, S>>;
    fn mul(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.data().clone() * rhs.data().clone(),
            func: MulBackwardVV {
                xgrad: &self.grad,
                xdata: self.data().clone(),
                ygrad: &rhs.grad,
                ydata: rhs.data().clone(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardVT<'g, S, YF> {
    xgrad: &'g RefCell<S>,
    xdata: S,
    yfunc: YF,
    ydata: S,
}

impl<'g, S, YF> Backward<S> for MulBackwardVT<'g, S, YF>
where
    S: Default + Clone + Mul<Output = S> + Add<Output = S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + self.ydata * res_grad.clone());
        self.yfunc.backward(self.xdata * res_grad);
    }
}

impl<'g, S, F> Mul<Tensor<S, F>> for &'g Variable<S>
where
    S: 'g + Clone + Mul<Output = S>,
{
    type Output = Tensor<S, MulBackwardVT<'g, S, F>>;
    fn mul(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self.data().clone() * rhs.data.clone(),
            func: MulBackwardVT {
                xgrad: &self.grad,
                xdata: self.data().clone(),
                yfunc: rhs.func,
                ydata: rhs.data,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardTS<S, XF> {
    xfunc: XF,
    ydata: S,
}

impl<S, XF> Backward<S> for MulBackwardTS<S, XF>
where
    S: Mul<Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xfunc.backward(self.ydata * res_grad);
    }
}

impl<S, F> Mul<S> for Tensor<S, F>
where
    S: Clone + Mul<Output = S>,
{
    type Output = Tensor<S, MulBackwardTS<S, F>>;
    fn mul(self, rhs: S) -> Self::Output {
        Tensor {
            data: self.data * rhs.clone(),
            func: MulBackwardTS {
                xfunc: self.func,
                ydata: rhs,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardTV<'g, S, XF> {
    xfunc: XF,
    xdata: S,
    ygrad: &'g RefCell<S>,
    ydata: S,
}

impl<'g, S, XF> Backward<S> for MulBackwardTV<'g, S, XF>
where
    S: Default + Clone + Mul<Output = S> + Add<Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad + self.xdata * res_grad.clone());
        // this way it is tail recursive
        self.xfunc.backward(self.ydata * res_grad);
    }
}

impl<'g, S, XF> Mul<&'g Variable<S>> for Tensor<S, XF>
where
    S: 'g + Clone + Mul<Output = S>,
{
    type Output = Tensor<S, MulBackwardTV<'g, S, XF>>;
    fn mul(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.data.clone() * rhs.data().clone(),
            func: MulBackwardTV {
                xfunc: self.func,
                xdata: self.data,
                ygrad: &rhs.grad,
                ydata: rhs.data().clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardTT<S, XF, YF> {
    xfunc: XF,
    xdata: S,
    yfunc: YF,
    ydata: S,
}

impl<S, XF, YF> Backward<S> for MulBackwardTT<S, XF, YF>
where
    S: Clone + Mul<Output = S>,
    XF: Backward<S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        // If res_grad is &S this is not a copy, but hey, advantages from passing S
        // by value and potentially doing operations in place are bigger than this copy
        // (at least hope so), if not, this will be changed
        self.xfunc.backward(self.ydata * res_grad.clone());
        self.yfunc.backward(self.xdata * res_grad);
    }
}

impl<S, XF, YF> Mul<Tensor<S, YF>> for Tensor<S, XF>
where
    S: Clone + Mul<Output = S>,
{
    type Output = Tensor<S, MulBackwardTT<S, XF, YF>>;
    fn mul(self, rhs: Tensor<S, YF>) -> Self::Output {
        Tensor {
            data: self.data.clone() * rhs.data.clone(),
            func: MulBackwardTT {
                xfunc: self.func,
                xdata: self.data,
                yfunc: rhs.func,
                ydata: rhs.data,
            }
        }
    }
}
