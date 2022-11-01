use crate::{tensor::{Variable, Tensor, Backward, Gradient}, dtype::DType, accel::cpu, ops::ConvertFrom};
use std::{cell::RefCell, ops::Add};
use duplicate::duplicate_item;

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardSV<'g, YG> {
    ygrad: &'g Gradient<YG>,
}

impl<S> Backward<S> for AddBackwardSV<'_, S>
where
    S: Add<S, Output = S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.accumulate(res_grad);
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];
    [cpu::Buffer<f32>]; [cpu::Buffer<f64>]; [cpu::Buffer<i32>]; [cpu::Buffer<i64>]; [cpu::Buffer<i128>];
    [cpu::Buffer<u8>]; [cpu::Buffer<u16>]; [cpu::Buffer<u32>]; [cpu::Buffer<u64>]; [cpu::Buffer<u128>]; [cpu::Buffer<bool>];)]
impl<'g, YS, YG> Add<&'g Variable<YS, YG>> for dtype
where
    Self: Add<YS>,
    YS: Clone + DType,
{
    type Output = Tensor<<Self as Add<YS>>::Output, AddBackwardSV<'g, YG>>;
    fn add(self, rhs: &'g Variable<YS, YG>) -> Self::Output {
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
pub struct AddBackwardVS<'g, XG> {
    xgrad: &'g Gradient<XG>,
}

impl<S> Backward<S> for AddBackwardVS<'_, S>
where
    S: Add<S, Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(res_grad);
    }
}

impl<'g, XS, YS, XG> Add<YS> for &'g Variable<XS, XG>
where
    XS: Clone + Add<YS>,
    YS: DType,
{
    type Output = Tensor<<XS as Add<YS>>::Output, AddBackwardVS<'g, XG>>;
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
pub struct AddBackwardVV<'g, XG, YG> {
    xgrad: &'g Gradient<XG>,
    ygrad: &'g Gradient<YG>,
}

impl<S> Backward<S> for AddBackwardVV<'_, S, S>
where
    S: Clone + Add<S, Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(res_grad.clone());
        self.ygrad.accumulate(res_grad);
    }
}

impl<'g, XS, XG, YS, YG> Add<&'g Variable<YS, YG>> for &'g Variable<XS, XG>
where
    XS: Clone + Add<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as Add<YS>>::Output, AddBackwardVV<'g, XG, YG>>;
    fn add(self, rhs: &'g Variable<YS, YG>) -> Self::Output {
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
pub struct AddBackwardVT<'g, XG, YF> {
    xgrad: &'g Gradient<XG>,
    ygrad_fn: YF,
}

impl<S, YF> Backward<S> for AddBackwardVT<'_, S, YF>
where
    S: Clone + Add<S, Output = S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(res_grad.clone());
        self.ygrad_fn.backward(res_grad);
    }
}

impl<'g, XS, XG, YS, YF> Add<Tensor<YS, YF>> for &'g Variable<XS, XG>
where
    XS: Clone + Add<YS>,
{
    type Output = Tensor<<XS as Add<YS>>::Output, AddBackwardVT<'g, XG, YF>>;
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
pub struct AddBackwardTV<'g, YG, XF> {
    xgrad_fn: XF,
    ygrad: &'g Gradient<YG>,
}

impl<S, XF> Backward<S> for AddBackwardTV<'_, S, XF>
where
    S: Clone + Add<S, Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.accumulate(res_grad.clone());
        self.xgrad_fn.backward(res_grad);
    }
}

impl<'g, XS, YS, YG, XF> Add<&'g Variable<YS, YG>> for Tensor<XS, XF>
where
    XS: Add<YS> + DType,
    YS: Clone + DType,
{
    type Output = Tensor<<XS as Add<YS>>::Output, AddBackwardTV<'g, YG, XF>>;
    fn add(self, rhs: &'g Variable<YS, YG>) -> Self::Output {
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
