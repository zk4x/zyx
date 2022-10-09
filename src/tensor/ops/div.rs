use crate::tensor::{B, Variable, Tensor, Backward, ops::RefCellReplaceTake};
use std::{cell::RefCell, ops::{Neg, Add, Sub, Mul, Div}};

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardSV<'g, S> {
    res: S,
    ygrad: &'g RefCell<S>,
    ydata: S,
}

impl<'g, S> Backward<S> for DivBackwardSV<'g, S>
where
    S: Default + Div<Output = S> + Sub<Output = S> + Mul<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad - self.res / self.ydata * res_grad);
    }
}

// If you wanted to Div Variable or Tensor to S, you need to wrap it inside B(),
// but you can Div S to Variable or Tensor
impl<'g, S> Div<&'g Variable<S>> for B<S>
where
    S: 'g + Clone + Div<Output = S>,
{
    type Output = Tensor<S, DivBackwardSV<'g, S>>;
    fn div(self, rhs: &'g Variable<S>) -> Self::Output {
        let res = self.0 / rhs.data().clone();
        Tensor {
            data: res.clone(),
            func: DivBackwardSV {
                res,
                ygrad: &rhs.grad,
                ydata: rhs.data().clone(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardST<S, YF> {
    res: S,
    yfunc: YF,
    ydata: S,
}

impl<S, YF> Backward<S> for DivBackwardST<S, YF>
where
    S: Neg<Output = S> + Mul<Output = S> + Div<Output = S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.yfunc.backward(-self.res / self.ydata * res_grad);
    }
}

impl<S, F> Div<Tensor<S, F>> for B<S>
where
    S: Clone + Div<Output = S>,
{
    type Output = Tensor<S, DivBackwardST<S, F>>;
    fn div(self, rhs: Tensor<S, F>) -> Self::Output {
        let res = self.0 / rhs.data.clone();
        Tensor {
            data: res.clone(),
            func: DivBackwardST {
                res,
                yfunc: rhs.func,
                ydata: rhs.data,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardVS<'g, S> {
    xgrad: &'g RefCell<S>,
    ydata: S,
}

impl<'g, S> Backward<S> for DivBackwardVS<'g, S>
where
    S: Default + Div<Output = S> + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad / self.ydata);
    }
}

impl<'g, S> Div<S> for &'g Variable<S>
where
    S: 'g + Clone + Div<Output = S>,
{
    type Output = Tensor<S, DivBackwardVS<'g, S>>;
    fn div(self, rhs: S) -> Self::Output {
        Tensor {
            data: self.data().clone() / rhs.clone(),
            func: DivBackwardVS {
                xgrad: &self.grad,
                ydata: rhs,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardVV<'g, S> {
    res: S,
    xgrad: &'g RefCell<S>,
    ygrad: &'g RefCell<S>,
    ydata: S,
}

impl<'g, S> Backward<S> for DivBackwardVV<'g, S>
where
    S: Default + Clone + Add<Output = S> + Sub<Output = S> + Mul<Output = S> + Div<Output = S>,
{
    fn backward(self, res_grad: S) {
        let temp = res_grad / self.ydata;
        self.xgrad.replace_take(|grad| grad + temp.clone());
        self.ygrad.replace_take(|grad| grad - self.res * temp);
    }
}

impl<'g, S> Div<&'g Variable<S>> for &'g Variable<S>
where
    S: 'g + Clone + Div<Output = S>,
{
    type Output = Tensor<S, DivBackwardVV<'g, S>>;
    fn div(self, rhs: &'g Variable<S>) -> Self::Output {
        let res = self.data().clone() / rhs.data().clone();
        Tensor {
            data: res.clone(),
            func: DivBackwardVV {
                xgrad: &self.grad,
                res,
                ygrad: &rhs.grad,
                ydata: rhs.data().clone(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardVT<'g, S, YF> {
    res: S,
    xgrad: &'g RefCell<S>,
    yfunc: YF,
    ydata: S,
}

impl<'g, S, YF> Backward<S> for DivBackwardVT<'g, S, YF>
where
    S: Default + Clone + Neg<Output = S> + Add<Output = S> + Mul<Output = S> + Div<Output = S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        let temp = res_grad / self.ydata;
        self.xgrad.replace_take(|grad| grad + temp.clone());
        self.yfunc.backward(-self.res * temp);
    }
}

impl<'g, S, F> Div<Tensor<S, F>> for &'g Variable<S>
where
    S: 'g + Clone + Div<Output = S>,
{
    type Output = Tensor<S, DivBackwardVT<'g, S, F>>;
    fn div(self, rhs: Tensor<S, F>) -> Self::Output {
        let res = self.data().clone() / rhs.data.clone();
        Tensor {
            data: res.clone(),
            func: DivBackwardVT {
                res,
                xgrad: &self.grad,
                yfunc: rhs.func,
                ydata: rhs.data,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardTS<S, XF> {
    xfunc: XF,
    ydata: S,
}

impl<S, XF> Backward<S> for DivBackwardTS<S, XF>
where
    S: Div<Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xfunc.backward(res_grad / self.ydata);
    }
}

impl<S, F> Div<S> for Tensor<S, F>
where
    S: Clone + Div<Output = S>,
{
    type Output = Tensor<S, DivBackwardTS<S, F>>;
    fn div(self, rhs: S) -> Self::Output {
        Tensor {
            data: self.data / rhs.clone(),
            func: DivBackwardTS {
                xfunc: self.func,
                ydata: rhs,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardTV<'g, S, XF> {
    res: S,
    xfunc: XF,
    ygrad: &'g RefCell<S>,
    ydata: S,
}

impl<'g, S, XF> Backward<S> for DivBackwardTV<'g, S, XF>
where
    S: Default + Clone + Sub<Output = S> + Mul<Output = S> + Div<Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        let temp = res_grad / self.ydata;
        self.ygrad.replace_take(|grad| grad - self.res * temp.clone());
        self.xfunc.backward(temp);
    }
}

impl<'g, S, XF> Div<&'g Variable<S>> for Tensor<S, XF>
where
    S: 'g + Clone + Div<Output = S>,
{
    type Output = Tensor<S, DivBackwardTV<'g, S, XF>>;
    fn div(self, rhs: &'g Variable<S>) -> Self::Output {
        let res = self.data / rhs.data().clone();
        Tensor {
            data: res.clone(),
            func: DivBackwardTV {
                res,
                xfunc: self.func,
                ygrad: &rhs.grad,
                ydata: rhs.data().clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardTT<S, XF, YF> {
    res: S,
    xfunc: XF,
    yfunc: YF,
    ydata: S,
}

impl<S, XF, YF> Backward<S> for DivBackwardTT<S, XF, YF>
where
    S: Clone + Neg<Output = S> + Mul<Output = S> + Div<Output = S>,
    XF: Backward<S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        // If res_grad is &S this is not a copy, but hey, advantages from passing S
        // by value and potentially doing operations in place are bigger than this copy
        // (at least hope so), if not, this will be changed
        let temp = res_grad / self.ydata;
        self.xfunc.backward(temp.clone());
        self.yfunc.backward(- self.res * temp);
    }
}

impl<S, XF, YF> Div<Tensor<S, YF>> for Tensor<S, XF>
where
    S: Clone + Div<Output = S>,
{
    type Output = Tensor<S, DivBackwardTT<S, XF, YF>>;
    fn div(self, rhs: Tensor<S, YF>) -> Self::Output {
        let res = self.data / rhs.data.clone();
        Tensor {
            data: res.clone(),
            func: DivBackwardTT {
                res,
                xfunc: self.func,
                yfunc: rhs.func,
                ydata: rhs.data,
            }
        }
    }
}
