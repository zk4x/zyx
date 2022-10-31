use crate::{tensor::{Tensor, Variable, Backward, ops::RefCellReplaceTake}, accel::cpu, dtype::DType};
use std::{cell::RefCell, ops::{Sub, Neg, Add}};
use duplicate::duplicate_item;

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardSV<'g, YS> {
    ygrad: &'g RefCell<YS>,
}

impl<S, YS> Backward<S> for SubBackwardSV<'_, YS>
where
    YS: Default + Sub<S, Output = YS>,
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
    S: Clone + DType,
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
    S: DType,
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
pub struct SubBackwardVS<'g, XS> {
    xgrad: &'g RefCell<XS>,
}

impl<S, XS> Backward<S> for SubBackwardVS<'_, XS>
where
    XS: Default + Add<S, Output = XS>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad);
    }
}

impl<'g, XS, YS> Sub<YS> for &'g Variable<XS>
where
    XS: Clone + Sub<YS>,
    YS: DType,
{
    type Output = Tensor<<XS as Sub<YS>>::Output, SubBackwardVS<'g, XS>>;
    fn sub(self, rhs: YS) -> Self::Output {
        Tensor {
            data: self.data().clone() - rhs,
            grad_fn: SubBackwardVS {
                xgrad: &self.grad,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardVV<'g, XS, YS> {
    xgrad: &'g RefCell<XS>,
    ygrad: &'g RefCell<YS>,
}

impl<S, XS, YS> Backward<S> for SubBackwardVV<'_, XS, YS>
where
    S: Clone,
    XS: Default + Add<S, Output = XS>,
    YS: Default + Sub<S, Output = YS>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad.clone());
        self.ygrad.replace_take(|grad| grad - res_grad);
    }
}

impl<'g, XS, YS> Sub<&'g Variable<YS>> for &'g Variable<XS>
where
    XS: Clone + Sub<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as Sub<YS>>::Output, SubBackwardVV<'g, XS, YS>>;
    fn sub(self, rhs: &'g Variable<YS>) -> Self::Output {
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
pub struct SubBackwardVT<'g, XS, YF> {
    xgrad: &'g RefCell<XS>,
    ygrad_fn: YF,
}

impl<S, XS, YF> Backward<S> for SubBackwardVT<'_, XS, YF>
where
    S: Clone + Neg,
    XS: Default + Add<S, Output = XS>,
    YF: Backward<<S as Neg>::Output>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad.clone());
        self.ygrad_fn.backward(-res_grad);
    }
}

impl<'g, S, F> Sub<Tensor<S, F>> for &'g Variable<S>
where
    S: Clone + Sub<Output = S>,
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

impl<XS, YS, XF> Sub<YS> for Tensor<XS, XF>
where
    XS: Sub<YS>,
    YS: DType,
{
    type Output = Tensor<<XS as Sub<YS>>::Output, XF>;
    fn sub(self, rhs: YS) -> Self::Output {
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

impl<S, YS, XF> Backward<S> for SubBackwardTV<'_, YS, XF>
where
    S: Clone,
    YS: Default + Sub<S, Output = YS>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(res_grad.clone());
        self.ygrad.replace_take(|grad| grad - res_grad);
    }
}

impl<'g, XS, YS, F> Sub<&'g Variable<YS>> for Tensor<XS, F>
where
    XS: Clone + Sub<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as Sub<YS>>::Output, SubBackwardTV<'g, YS, F>>;
    fn sub(self, rhs: &'g Variable<YS>) -> Self::Output {
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
    S: Clone + Neg,
    XF: Backward<<S as Neg>::Output>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(-res_grad.clone());
        self.ygrad_fn.backward(res_grad);
    }
}

impl<XS, YS, XF, YF> Sub<Tensor<YS, YF>> for Tensor<XS, XF>
where
    XS: Sub<YS>,
{
    type Output = Tensor<<XS as Sub<YS>>::Output, SubBackwardTT<XF, YF>>;
    fn sub(self, rhs: Tensor<YS, YF>) -> Self::Output {
        Tensor {
            data: self.data - rhs.data,
            grad_fn: SubBackwardTT {
                xgrad_fn: self.grad_fn,
                ygrad_fn: rhs.grad_fn,
            },
        }
    }
}
