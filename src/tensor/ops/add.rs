use crate::{tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}, dtype::DType, accel::cpu};
use std::{cell::RefCell, ops::Add};
use duplicate::duplicate_item;

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardSV<'g, S2> {
    ygrad: &'g RefCell<S2>,
}

impl<S, S2> Backward<S> for AddBackwardSV<'_, S2>
where
    S2: Default + Add<S, Output = S2>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad + res_grad);
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];
    [cpu::Buffer<f32>]; [cpu::Buffer<f64>]; [cpu::Buffer<i32>]; [cpu::Buffer<i64>]; [cpu::Buffer<i128>];
    [cpu::Buffer<u8>]; [cpu::Buffer<u16>]; [cpu::Buffer<u32>]; [cpu::Buffer<u64>]; [cpu::Buffer<u128>]; [cpu::Buffer<bool>];)]
impl<'g, S> Add<&'g Variable<S>> for dtype
where
    Self: Add<S>,
    S: Clone + DType,
{
    type Output = Tensor<<Self as Add<S>>::Output, AddBackwardSV<'g, S>>;
    fn add(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self + rhs.data().clone(),
            grad_fn: AddBackwardSV {
                ygrad: &rhs.grad,
            }
        }
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];
    [cpu::Buffer<f32>]; [cpu::Buffer<f64>]; [cpu::Buffer<i32>]; [cpu::Buffer<i64>]; [cpu::Buffer<i128>];
    [cpu::Buffer<u8>]; [cpu::Buffer<u16>]; [cpu::Buffer<u32>]; [cpu::Buffer<u64>]; [cpu::Buffer<u128>]; [cpu::Buffer<bool>];)]
impl<S, F> Add<Tensor<S, F>> for dtype
where
    Self: Add<S>,
    S: DType,
{
    type Output = Tensor<<Self as Add<S>>::Output, F>;
    fn add(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self + rhs.data,
            grad_fn: rhs.grad_fn,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardVS<'g, S> {
    xgrad: &'g RefCell<S>,
}

impl<S, XS> Backward<S> for AddBackwardVS<'_, XS>
where
    XS: Default + Add<S, Output = XS>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad);
    }
}

impl<'g, XS, YS> Add<YS> for &'g Variable<XS>
where
    XS: Clone + Add<YS>,
    YS: DType,
{
    type Output = Tensor<<XS as Add<YS>>::Output, AddBackwardVS<'g, XS>>;
    fn add(self, rhs: YS) -> Self::Output {
        Tensor {
            data: self.data().clone() + rhs,
            grad_fn: AddBackwardVS {
                xgrad: &self.grad,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardVV<'g, XS, YS> {
    xgrad: &'g RefCell<XS>,
    ygrad: &'g RefCell<YS>,
}

impl<S, XS, YS> Backward<S> for AddBackwardVV<'_, XS, YS>
where
    S: Clone,
    XS: Default + Add<S, Output = XS>,
    YS: Default + Add<S, Output = YS>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad.clone());
        self.ygrad.replace_take(|grad| grad + res_grad);
    }
}

impl<'g, XS, YS> Add<&'g Variable<YS>> for &'g Variable<XS>
where
    XS: Clone + Add<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as Add<YS>>::Output, AddBackwardVV<'g, XS, YS>>;
    fn add(self, rhs: &'g Variable<YS>) -> Self::Output {
        Tensor {
            data: self.data().clone() + rhs.data().clone(),
            grad_fn: AddBackwardVV {
                xgrad: &self.grad,
                ygrad: &rhs.grad,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardVT<'g, XS, YF> {
    xgrad: &'g RefCell<XS>,
    ygrad_fn: YF,
}

impl<S, XS, YF> Backward<S> for AddBackwardVT<'_, XS, YF>
where
    S: Clone,
    XS: Default + Add<S, Output = XS>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad.clone());
        self.ygrad_fn.backward(res_grad);
    }
}

impl<'g, XS, YS, YF> Add<Tensor<YS, YF>> for &'g Variable<XS>
where
    XS: Clone + Add<YS>,
{
    type Output = Tensor<<XS as Add<YS>>::Output, AddBackwardVT<'g, XS, YF>>;
    fn add(self, rhs: Tensor<YS, YF>) -> Self::Output {
        Tensor {
            data: self.data().clone() + rhs.data,
            grad_fn: AddBackwardVT {
                xgrad: &self.grad,
                ygrad_fn: rhs.grad_fn,
            }
        }
    }
}

impl<XS, YS, F> Add<YS> for Tensor<XS, F>
where
    XS: Add<YS> + DType,
    YS: DType,
{
    type Output = Tensor<<XS as Add<YS>>::Output, F>;
    fn add(self, rhs: YS) -> Self::Output {
        Tensor {
            data: self.data + rhs,
            grad_fn: self.grad_fn,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardTV<'g, YS, XF> {
    xgrad_fn: XF,
    ygrad: &'g RefCell<YS>,
}

impl<S, YS, XF> Backward<S> for AddBackwardTV<'_, YS, XF>
where
    S: Clone,
    YS: Default + Add<S, Output = YS>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad + res_grad.clone());
        self.xgrad_fn.backward(res_grad);
    }
}

impl<'g, XS, YS, XF> Add<&'g Variable<YS>> for Tensor<XS, XF>
where
    XS: Add<YS> + DType,
    YS: Clone + DType,
{
    type Output = Tensor<<XS as Add<YS>>::Output, AddBackwardTV<'g, YS, XF>>;
    fn add(self, rhs: &'g Variable<YS>) -> Self::Output {
        Tensor {
            data: self.data + rhs.data().clone(),
            grad_fn: AddBackwardTV {
                xgrad_fn: self.grad_fn,
                ygrad: &rhs.grad,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardTT<XF, YF> {
    xgrad_fn: XF,
    ygrad_fn: YF,
}

impl<S, XF, YF> Backward<S> for AddBackwardTT<XF, YF>
where
    S: Clone,
    XF: Backward<S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(res_grad.clone());
        self.ygrad_fn.backward(res_grad);
    }
}

impl<XS, YS, XF, YF> Add<Tensor<YS, YF>> for Tensor<XS, XF>
where
    XS: Add<YS> + DType,
    YS: DType,
{
    type Output = Tensor<<XS as Add<YS>>::Output, AddBackwardTT<XF, YF>>;
    fn add(self, rhs: Tensor<YS, YF>) -> Self::Output {
        Tensor {
            data: self.data + rhs.data,
            grad_fn: AddBackwardTT {
                xgrad_fn: self.grad_fn,
                ygrad_fn: rhs.grad_fn,
            }
        }
    }
}
