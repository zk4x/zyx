use crate::{tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}, accel::cpu, dtype::DType};
use std::{cell::RefCell, ops::{Add, Mul}};
use duplicate::duplicate_item;

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardSV<'g, S, S2> {
    xdata: S2,
    ygrad: &'g RefCell<S>,
}

impl<S, S2> Backward<S> for MulBackwardSV<'_, S, S2>
where
    S: Default + Add<<S2 as Mul<S>>::Output, Output = S>,
    S2: Mul<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad + self.xdata * res_grad);
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<'g, S> Mul<&'g Variable<S>> for dtype
where
    Self: Mul<S>,
    S: Clone,
{
    type Output = Tensor<<Self as Mul<S>>::Output, MulBackwardSV<'g, S, Self>>;
    fn mul(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self * rhs.data().clone(),
            grad_fn: MulBackwardSV {
                xdata: self,
                ygrad: &rhs.grad,
            }
        }
    }
}

#[duplicate_item( dtype; [cpu::Buffer<f32>]; [cpu::Buffer<f64>]; [cpu::Buffer<i32>]; [cpu::Buffer<i64>]; [cpu::Buffer<i128>];
    [cpu::Buffer<u8>]; [cpu::Buffer<u16>]; [cpu::Buffer<u32>]; [cpu::Buffer<u64>]; [cpu::Buffer<u128>]; [cpu::Buffer<bool>];)]
impl<'g, S> Mul<&'g Variable<S>> for dtype
where
    Self: Mul<S>,
    S: Clone,
{
    type Output = Tensor<<Self as Mul<S>>::Output, MulBackwardSV<'g, S, Self>>;
    fn mul(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.clone() * rhs.data().clone(),
            grad_fn: MulBackwardSV {
                xdata: self,
                ygrad: &rhs.grad,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardST<S, YF> {
    xdata: S,
    ygrad_fn: YF,
}

impl<S, S2, YF> Backward<S> for MulBackwardST<S2, YF>
where
    S2: Mul<S>,
    YF: Backward<<S2 as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.ygrad_fn.backward(self.xdata * res_grad);
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool]; )]
impl<S, F> Mul<Tensor<S, F>> for dtype
where
    Self: Mul<S>,
{
    type Output = Tensor<<Self as Mul<S>>::Output, MulBackwardST<Self, F>>;
    fn mul(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self * rhs.data,
            grad_fn: MulBackwardST {
                xdata: self,
                ygrad_fn: rhs.grad_fn,
            },
        }
    }
}

#[duplicate_item( dtype; [cpu::Buffer<f32>]; [cpu::Buffer<f64>]; [cpu::Buffer<i32>]; [cpu::Buffer<i64>]; [cpu::Buffer<i128>];
    [cpu::Buffer<u8>]; [cpu::Buffer<u16>]; [cpu::Buffer<u32>]; [cpu::Buffer<u64>]; [cpu::Buffer<u128>]; [cpu::Buffer<bool>];)]
impl<S, F> Mul<Tensor<S, F>> for dtype
where
    Self: Mul<S>,
{
    type Output = Tensor<<Self as Mul<S>>::Output, MulBackwardST<Self, F>>;
    fn mul(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self.clone() * rhs.data,
            grad_fn: MulBackwardST {
                xdata: self,
                ygrad_fn: rhs.grad_fn,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardVS<'g, XS, YS> {
    xgrad: &'g RefCell<XS>,
    ydata: YS,
}

impl<S, XS, YS> Backward<S> for MulBackwardVS<'_, XS, YS>
where
    XS: Default + Add<<YS as Mul<S>>::Output, Output = XS>,
    YS: Mul<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + self.ydata * res_grad);
    }
}

impl<'g, XS, YS> Mul<YS> for &'g Variable<XS>
where
    YS: Clone + DType,
    XS: Clone + Mul<YS>,
{
    type Output = Tensor<<XS as Mul<YS>>::Output, MulBackwardVS<'g, XS, YS>>;
    fn mul(self, rhs: YS) -> Self::Output {
        Tensor {
            data: self.data().clone() * rhs.clone(),
            grad_fn: MulBackwardVS {
                xgrad: &self.grad,
                ydata: rhs,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardVV<'g, XS, YS> {
    xgrad: &'g RefCell<XS>,
    xdata: XS,
    ygrad: &'g RefCell<YS>,
    ydata: YS,
}

impl<S, XS, YS> Backward<S> for MulBackwardVV<'_, XS, YS>
where
    S: Clone,
    XS: Default + Add<<YS as Mul<S>>::Output, Output = XS> + Mul<S>,
    YS: Default + Add<<XS as Mul<S>>::Output, Output = YS> + Mul<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + self.ydata * res_grad.clone());
        self.ygrad.replace_take(|grad| grad + self.xdata * res_grad);
    }
}

impl<'g, XS, YS> Mul<&'g Variable<YS>> for &'g Variable<XS>
where
    XS: Clone + Mul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as Mul<YS>>::Output, MulBackwardVV<'g, XS, YS>>;
    fn mul(self, rhs: &'g Variable<YS>) -> Self::Output {
        Tensor {
            data: self.data().clone() * rhs.data().clone(),
            grad_fn: MulBackwardVV {
                xgrad: &self.grad,
                xdata: self.data().clone(),
                ygrad: &rhs.grad,
                ydata: rhs.data().clone(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardVT<'g, XS, YS, YF> {
    xgrad: &'g RefCell<XS>,
    xdata: XS,
    ygrad_fn: YF,
    ydata: YS,
}

impl<S, XS, YS, YF> Backward<S> for MulBackwardVT<'_, XS, YS, YF>
where
    S: Clone,
    XS: Default + Mul<S> + Add<<YS as Mul<S>>::Output, Output = XS>,
    YS: Mul<S>,
    YF: Backward<<XS as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + self.ydata * res_grad.clone());
        self.ygrad_fn.backward(self.xdata * res_grad);
    }
}

impl<'g, XS, YS, F> Mul<Tensor<YS, F>> for &'g Variable<XS>
where
    XS: Clone + Mul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as Mul<YS>>::Output, MulBackwardVT<'g, XS, YS, F>>;
    fn mul(self, rhs: Tensor<YS, F>) -> Self::Output {
        Tensor {
            data: self.data().clone() * rhs.data.clone(),
            grad_fn: MulBackwardVT {
                xgrad: &self.grad,
                xdata: self.data().clone(),
                ygrad_fn: rhs.grad_fn,
                ydata: rhs.data,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardTS<S, XF> {
    xgrad_fn: XF,
    ydata: S,
}

impl<S, YS, XF> Backward<S> for MulBackwardTS<YS, XF>
where
    YS: Mul<S>,
    XF: Backward<<YS as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(self.ydata * res_grad);
    }
}

impl<XS, YS, F> Mul<YS> for Tensor<XS, F>
where
    XS: Mul<YS>,
    YS: DType + Clone,
{
    type Output = Tensor<<XS as Mul<YS>>::Output, MulBackwardTS<YS, F>>;
    fn mul(self, rhs: YS) -> Self::Output {
        Tensor {
            data: self.data * rhs.clone(),
            grad_fn: MulBackwardTS {
                xgrad_fn: self.grad_fn,
                ydata: rhs,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardTV<'g, XS, YS, XF> {
    xgrad_fn: XF,
    xdata: XS,
    ygrad: &'g RefCell<YS>,
    ydata: YS,
}

impl<S, XS, YS, XF> Backward<S> for MulBackwardTV<'_, XS, YS, XF>
where
    S: Clone,
    XS: Mul<S>,
    YS: Default + Add<<XS as Mul<S>>::Output, Output = YS> + Mul<S>,
    XF: Backward<<YS as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad + self.xdata * res_grad.clone());
        // this way it is tail recursive
        self.xgrad_fn.backward(self.ydata * res_grad);
    }
}

impl<'g, XS, YS, XF> Mul<&'g Variable<YS>> for Tensor<XS, XF>
where
    XS: Clone + Mul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as Mul<YS>>::Output, MulBackwardTV<'g, XS, YS, XF>>;
    fn mul(self, rhs: &'g Variable<YS>) -> Self::Output {
        Tensor {
            data: self.data.clone() * rhs.data().clone(),
            grad_fn: MulBackwardTV {
                xgrad_fn: self.grad_fn,
                xdata: self.data,
                ygrad: &rhs.grad,
                ydata: rhs.data().clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardTT<XS, YS, XF, YF> {
    xgrad_fn: XF,
    xdata: XS,
    ygrad_fn: YF,
    ydata: YS,
}

impl<S, XS, YS, XF, YF> Backward<S> for MulBackwardTT<XS, YS, XF, YF>
where
    S: Clone,
    XS: Mul<S>,
    YS: Mul<S>,
    XF: Backward<<YS as Mul<S>>::Output>,
    YF: Backward<<XS as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        // If res_grad is &S this is not a copy, but hey, advantages from passing S
        // by value and potentially doing operations in place are bigger than this copy
        // (at least hope so), if not, this will be changed
        self.xgrad_fn.backward(self.ydata * res_grad.clone());
        self.ygrad_fn.backward(self.xdata * res_grad);
    }
}

impl<XS, YS, XF, YF> Mul<Tensor<YS, YF>> for Tensor<XS, XF>
where
    XS: Clone + Mul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as Mul<YS>>::Output, MulBackwardTT<XS, YS, XF, YF>>;
    fn mul(self, rhs: Tensor<YS, YF>) -> Self::Output {
        Tensor {
            data: self.data.clone() * rhs.data.clone(),
            grad_fn: MulBackwardTT {
                xgrad_fn: self.grad_fn,
                xdata: self.data,
                ygrad_fn: rhs.grad_fn,
                ydata: rhs.data,
            }
        }
    }
}
