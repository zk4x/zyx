use crate::{tensor::{B, Variable, Tensor, Backward, ops::RefCellReplaceTake}, dtype::DType};
use std::{cell::RefCell, ops::{Neg, Add, Sub, Mul, Div}};
use duplicate::duplicate_item;
use crate::accel::cpu::Buffer;

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
            grad_fn: DivBackwardSV {
                res,
                ygrad: &rhs.grad,
                ydata: rhs.data().clone(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardST<S, S2, YF> {
    res: S,
    ygrad_fn: YF,
    ydata: S2,
}

impl<S, S2, S3, YF> Backward<S3> for DivBackwardST<S, S2, YF>
where
    S: Neg,
    <S as Neg>::Output: Div<S2>,
    <<S as Neg>::Output as Div<S2>>::Output: Mul<S3, Output = S2>,
    YF: Backward<S2>,
{
    fn backward(self, res_grad: S3) {
        self.ygrad_fn.backward(-self.res / self.ydata * res_grad);
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i32]; [i64]; [i128]; [u8]; [u16]; [u32]; [u64]; [u128]; [bool];
    [Buffer<f32>]; [Buffer<f64>]; [Buffer<i32>]; [Buffer<i64>]; [Buffer<i128>]; [Buffer<u8>]; [Buffer<u16>]; [Buffer<u32>]; [Buffer<u64>]; [Buffer<u128>]; [Buffer<bool>];)]
impl<S, F> Div<Tensor<S, F>> for dtype
where
    S: Clone,
    Self: Div<S, Output = S>,
    // TODO: why does this overflow during compilation?
    //<Self as Div<S>>::Output: Clone,
{
    type Output = Tensor<<Self as Div<S>>::Output, DivBackwardST<<Self as Div<S>>::Output, S, F>>;
    fn div(self, rhs: Tensor<S, F>) -> Self::Output {
        let res = self / rhs.data.clone();
        Tensor {
            data: res.clone(),
            grad_fn: DivBackwardST {
                res,
                ygrad_fn: rhs.grad_fn,
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
            grad_fn: DivBackwardVS {
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
            grad_fn: DivBackwardVV {
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
    ygrad_fn: YF,
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
        self.ygrad_fn.backward(-self.res * temp);
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
            grad_fn: DivBackwardVT {
                res,
                xgrad: &self.grad,
                ygrad_fn: rhs.grad_fn,
                ydata: rhs.data,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardTS<S, XF> {
    xgrad_fn: XF,
    ydata: S,
}

impl<S, S2, XF> Backward<S2> for DivBackwardTS<S, XF>
where
    S2: Div<S>,
    XF: Backward<<S2 as Div<S>>::Output>,
{
    fn backward(self, res_grad: S2) {
        self.xgrad_fn.backward(res_grad / self.ydata);
    }
}

impl<S, S2, F> Div<S2> for Tensor<S, F>
where
    S2: DType + Clone,
    S: Clone + Div<S2>,
{
    type Output = Tensor<<S as Div<S2>>::Output, DivBackwardTS<S2, F>>;
    fn div(self, rhs: S2) -> Self::Output {
        Tensor {
            data: self.data / rhs.clone(),
            grad_fn: DivBackwardTS {
                xgrad_fn: self.grad_fn,
                ydata: rhs,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardTV<'g, S, XF> {
    res: S,
    xgrad_fn: XF,
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
        self.xgrad_fn.backward(temp);
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
            grad_fn: DivBackwardTV {
                res,
                xgrad_fn: self.grad_fn,
                ygrad: &rhs.grad,
                ydata: rhs.data().clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardTT<S, XF, YF> {
    res: S,
    xgrad_fn: XF,
    ygrad_fn: YF,
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
        self.xgrad_fn.backward(temp.clone());
        self.ygrad_fn.backward(- self.res * temp);
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
            grad_fn: DivBackwardTT {
                res,
                xgrad_fn: self.grad_fn,
                ygrad_fn: rhs.grad_fn,
                ydata: rhs.data,
            }
        }
    }
}
