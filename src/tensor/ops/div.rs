use crate::{tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}, dtype::DType, accel::cpu};
use std::{cell::RefCell, ops::{Neg, Add, Sub, Mul, Div}};
use duplicate::duplicate_item;

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardSV<'g, S, S2> {
    res: S2,
    ygrad: &'g RefCell<S>,
    ydata: S,
}

impl<S, S2> Backward<S> for DivBackwardSV<'_, S, S2>
where
    S: Default + Sub<<<S2 as Div<S>>::Output as Mul<S>>::Output, Output = S>,
    S2: Div<S>,
    <S2 as Div<S>>::Output: Mul<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad - self.res / self.ydata * res_grad);
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];
    [cpu::Buffer<f32>]; [cpu::Buffer<f64>]; [cpu::Buffer<i32>]; [cpu::Buffer<i64>]; [cpu::Buffer<i128>];
    [cpu::Buffer<u8>]; [cpu::Buffer<u16>]; [cpu::Buffer<u32>]; [cpu::Buffer<u64>]; [cpu::Buffer<u128>]; [cpu::Buffer<bool>];)]
impl<'g, S> Div<&'g Variable<S>> for dtype
where
    Self: Div<S, Output = S>,
    S: Clone,
    //<Self as Div<S>>::Output: Clone,
{
    type Output = Tensor<<Self as Div<S>>::Output, DivBackwardSV<'g, S, <Self as Div<S>>::Output>>;
    fn div(self, rhs: &'g Variable<S>) -> Self::Output {
        let res = self / rhs.data().clone();
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

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];
    [cpu::Buffer<f32>]; [cpu::Buffer<f64>]; [cpu::Buffer<i32>]; [cpu::Buffer<i64>]; [cpu::Buffer<i128>];
    [cpu::Buffer<u8>]; [cpu::Buffer<u16>]; [cpu::Buffer<u32>]; [cpu::Buffer<u64>]; [cpu::Buffer<u128>]; [cpu::Buffer<bool>];)]
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
pub struct DivBackwardVS<'g, XS, YS> {
    xgrad: &'g RefCell<XS>,
    ydata: YS,
}

impl<S, XS, YS> Backward<S> for DivBackwardVS<'_, XS, YS>
where
    S: Div<YS>,
    XS: Default + Add<<S as Div<YS>>::Output, Output = XS>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad / self.ydata);
    }
}

impl<'g, XS, YS> Div<YS> for &'g Variable<XS>
where
    XS: Clone + Div<YS>,
    YS: Clone + DType,
{
    type Output = Tensor<<XS as Div<YS>>::Output, DivBackwardVS<'g, XS, YS>>;
    fn div(self, rhs: YS) -> Self::Output {
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
pub struct DivBackwardVV<'g, S, XS, YS> {
    res: S,
    xgrad: &'g RefCell<XS>,
    ygrad: &'g RefCell<YS>,
    ydata: YS,
}

impl<S, S2, XS, YS> Backward<S> for DivBackwardVV<'_, S2, XS, YS>
where
    S: Div<YS>,
    <S as Div<YS>>::Output: Clone,
    S2: Mul<<S as Div<YS>>::Output>,
    XS: Default + Add<<S as Div<YS>>::Output, Output = XS>,
    YS: Default + Sub<<S2 as Mul<<S as Div<YS>>::Output>>::Output, Output = YS>,
{
    fn backward(self, res_grad: S) {
        let temp = res_grad / self.ydata;
        self.xgrad.replace_take(|grad| grad + temp.clone());
        self.ygrad.replace_take(|grad| grad - self.res * temp);
    }
}

impl<'g, XS, YS> Div<&'g Variable<YS>> for &'g Variable<XS>
where
    XS: Clone + Div<YS>,
    YS: Clone,
    <XS as Div<YS>>::Output: Clone,
{
    type Output = Tensor<<XS as Div<YS>>::Output, DivBackwardVV<'g, <XS as Div<YS>>::Output, XS, YS>>;
    fn div(self, rhs: &'g Variable<YS>) -> Self::Output {
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
pub struct DivBackwardVT<'g, S, XS, YS, YF> {
    res: S,
    xgrad: &'g RefCell<XS>,
    ygrad_fn: YF,
    ydata: YS,
}

impl<S, S2, XS, YS, YF> Backward<S> for DivBackwardVT<'_, S2, XS, YS, YF>
where
    S: Div<YS>,
    XS: Default + Add<<S as Div<YS>>::Output, Output = XS>,
    <S as Div<YS>>::Output: Clone,
    S2: Neg,
    <S2 as Neg>::Output: Mul<<S as Div<YS>>::Output>,
    YF: Backward<<<S2 as Neg>::Output as Mul<<S as Div<YS>>::Output>>::Output>,
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
    type Output = Tensor<S, DivBackwardVT<'g, S, S, S, F>>;
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

impl<S, XF> Backward<S> for DivBackwardTV<'_, S, XF>
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
pub struct DivBackwardTT<S, S2, XF, YF> {
    res: S2,
    xgrad_fn: XF,
    ygrad_fn: YF,
    ydata: S,
}

impl<S, S2, S3, XF, YF> Backward<S> for DivBackwardTT<S2, S3, XF, YF>
where
    S: Div<S2>,
    <S as Div<S2>>::Output: Clone,
    S3: Neg,
    <S3 as Neg>::Output: Mul<<S as Div<S2>>::Output>,
    XF: Backward<<S as Div<S2>>::Output>,
    YF: Backward<<<S3 as Neg>::Output as Mul<<S as Div<S2>>::Output>>::Output>,
{
    fn backward(self, res_grad: S) {
        let temp = res_grad / self.ydata;
        self.xgrad_fn.backward(temp.clone());
        self.ygrad_fn.backward(- self.res * temp);
    }
}

impl<XS, YS, XF, YF> Div<Tensor<YS, YF>> for Tensor<XS, XF>
where
    XS: Div<YS>,
    YS: Clone,
    <XS as Div<YS>>::Output: Clone,
{
    type Output = Tensor<<XS as Div<YS>>::Output, DivBackwardTT<YS, <XS as Div<YS>>::Output, XF, YF>>;
    fn div(self, rhs: Tensor<YS, YF>) -> Self::Output {
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
