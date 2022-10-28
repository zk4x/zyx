use crate::{tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}, accel::cpu};
use std::{cell::RefCell, ops::{Add, Mul}};
use duplicate::duplicate_item;

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardSV<'g, S, S2> {
    xdata: S2,
    ygrad: &'g RefCell<S>,
}

impl<'g, S, S2> Backward<S> for MulBackwardSV<'g, S, S2>
where
    S: Default + Add<<S2 as Mul<S>>::Output, Output = S>,
    S2: Mul<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad + self.xdata * res_grad);
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];
    [cpu::Buffer<f32>]; [cpu::Buffer<f64>]; [cpu::Buffer<i32>]; [cpu::Buffer<i64>]; [cpu::Buffer<i128>];
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

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];
    [cpu::Buffer<f32>]; [cpu::Buffer<f64>]; [cpu::Buffer<i32>]; [cpu::Buffer<i64>]; [cpu::Buffer<i128>];
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
            grad_fn: MulBackwardVS {
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
pub struct MulBackwardVT<'g, S, YF> {
    xgrad: &'g RefCell<S>,
    xdata: S,
    ygrad_fn: YF,
    ydata: S,
}

impl<'g, S, YF> Backward<S> for MulBackwardVT<'g, S, YF>
where
    S: Default + Clone + Mul<Output = S> + Add<Output = S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + self.ydata * res_grad.clone());
        self.ygrad_fn.backward(self.xdata * res_grad);
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

impl<S, XF> Backward<S> for MulBackwardTS<S, XF>
where
    S: Mul<Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(self.ydata * res_grad);
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
            grad_fn: MulBackwardTS {
                xgrad_fn: self.grad_fn,
                ydata: rhs,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardTV<'g, S, XF> {
    xgrad_fn: XF,
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
        self.xgrad_fn.backward(self.ydata * res_grad);
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
pub struct MulBackwardTT<S, XF, YF> {
    xgrad_fn: XF,
    xdata: S,
    ygrad_fn: YF,
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
        self.xgrad_fn.backward(self.ydata * res_grad.clone());
        self.ygrad_fn.backward(self.xdata * res_grad);
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
            grad_fn: MulBackwardTT {
                xgrad_fn: self.grad_fn,
                xdata: self.data,
                ygrad_fn: rhs.grad_fn,
                ydata: rhs.data,
            }
        }
    }
}
