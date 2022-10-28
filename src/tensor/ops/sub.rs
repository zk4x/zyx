use crate::{tensor::{Tensor, Variable, Backward, ops::RefCellReplaceTake}, accel::cpu};
use std::{cell::RefCell, ops::{Sub, Neg, Add}};
use duplicate::duplicate_item;

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardSV<'g, S> {
    ygrad: &'g RefCell<S>,
}

impl<'g, S, S2> Backward<S> for SubBackwardSV<'g, S2>
where
    S2: Default + Sub<S, Output = S2>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad - res_grad);
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];
    [cpu::Buffer<f32>]; [cpu::Buffer<f64>]; [cpu::Buffer<i32>]; [cpu::Buffer<i64>]; [cpu::Buffer<i128>];
    [cpu::Buffer<u8>]; [cpu::Buffer<u16>]; [cpu::Buffer<u32>]; [cpu::Buffer<u64>]; [cpu::Buffer<u128>]; [cpu::Buffer<bool>];)]
impl<'g, S> Sub<&'g Variable<S>> for dtype
where
    Self: Sub<S>,
    S: Clone,
{
    type Output = Tensor<<Self as Sub<S>>::Output, SubBackwardSV<'g, S>>;
    fn sub(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self - rhs.data().clone(),
            grad_fn: SubBackwardSV {
                ygrad: &rhs.grad,
            },
        }
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];
    [cpu::Buffer<f32>]; [cpu::Buffer<f64>]; [cpu::Buffer<i32>]; [cpu::Buffer<i64>]; [cpu::Buffer<i128>];
    [cpu::Buffer<u8>]; [cpu::Buffer<u16>]; [cpu::Buffer<u32>]; [cpu::Buffer<u64>]; [cpu::Buffer<u128>]; [cpu::Buffer<bool>];)]
impl<S, F> Sub<Tensor<S, F>> for dtype
where
    Self: Sub<S>,
{
    type Output = Tensor<<Self as Sub<S>>::Output, F>;
    fn sub(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self - rhs.data,
            grad_fn: rhs.grad_fn,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardVS<'g, S> {
    xgrad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for SubBackwardVS<'g, S>
where
    S: Default + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad);
    }
}

impl<'g, S> Sub<S> for &'g Variable<S>
where
    S: 'g + Clone + Sub<Output = S>,
{
    type Output = Tensor<S, SubBackwardVS<'g, S>>;
    fn sub(self, rhs: S) -> Self::Output {
        Tensor {
            data: self.data().clone() - rhs,
            grad_fn: SubBackwardVS {
                xgrad: &self.grad,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardVV<'g, S> {
    xgrad: &'g RefCell<S>,
    ygrad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for SubBackwardVV<'g, S>
where
    S: Default + Clone + Add<Output = S> + Sub<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad.clone());
        self.ygrad.replace_take(|grad| grad - res_grad);
    }
}

impl<'g, S> Sub<&'g Variable<S>> for &'g Variable<S>
where
    S: 'g + Clone + Sub<Output = S>,
{
    type Output = Tensor<S, SubBackwardVV<'g, S>>;
    fn sub(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.data().clone() - rhs.data().clone(),
            grad_fn: SubBackwardVV {
                xgrad: &self.grad,
                ygrad: &rhs.grad,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardVT<'g, S, YF> {
    xgrad: &'g RefCell<S>,
    ygrad_fn: YF,
}

impl<'g, S, YF> Backward<S> for SubBackwardVT<'g, S, YF>
where
    S: Default + Clone + Add<Output = S> + Neg<Output = S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad.clone());
        self.ygrad_fn.backward(-res_grad);
    }
}

impl<'g, S, F> Sub<Tensor<S, F>> for &'g Variable<S>
where
    S: 'g + Clone + Sub<Output = S>,
{
    type Output = Tensor<S, SubBackwardVT<'g, S, F>>;
    fn sub(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self.data().clone() - rhs.data,
            grad_fn: SubBackwardVT {
                xgrad: &self.grad,
                ygrad_fn: rhs.grad_fn,
            },
        }
    }
}

impl<S, XF> Sub<S> for Tensor<S, XF>
where
    S: Sub<Output = S>,
{
    type Output = Tensor<S, XF>;
    fn sub(self, rhs: S) -> Self::Output {
        Tensor {
            data: self.data - rhs,
            grad_fn: self.grad_fn,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardTV<'g, S, XF> {
    xgrad_fn: XF,
    ygrad: &'g RefCell<S>,
}

impl<'g, S, XF> Backward<S> for SubBackwardTV<'g, S, XF>
where
    S: Default + Clone + Sub<Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(res_grad.clone());
        self.ygrad.replace_take(|grad| grad - res_grad);
    }
}

impl<'g, S, F> Sub<&'g Variable<S>> for Tensor<S, F>
where
    S: 'g + Clone + Sub<Output = S>,
{
    type Output = Tensor<S, SubBackwardTV<'g, S, F>>;
    fn sub(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.data - rhs.data().clone(),
            grad_fn: SubBackwardTV {
                xgrad_fn: self.grad_fn,
                ygrad: &rhs.grad,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardTT<XF, YF> {
    xgrad_fn: XF,
    ygrad_fn: YF,
}

impl<S, XF, YF> Backward<S> for SubBackwardTT<XF, YF>
where
    S: Clone + Neg<Output = S>,
    XF: Backward<S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(-res_grad.clone());
        self.ygrad_fn.backward(res_grad);
    }
}

impl<S, XF, YF> Sub<Tensor<S, YF>> for Tensor<S, XF>
where
    S: Sub<Output = S>,
{
    type Output = Tensor<S, SubBackwardTT<XF, YF>>;
    fn sub(self, rhs: Tensor<S, YF>) -> Self::Output {
        Tensor {
            data: self.data - rhs.data,
            grad_fn: SubBackwardTT {
                xgrad_fn: self.grad_fn,
                ygrad_fn: rhs.grad_fn,
            },
        }
    }
}
