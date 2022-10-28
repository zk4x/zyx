use crate::{tensor::{Variable, Tensor, Backward, ops::RefCellReplaceTake}, dtype::DType, accel::cpu};
use std::{cell::RefCell, ops::Add};
use duplicate::duplicate_item;

// Naming scheme for backward function is FunctionName + Backward + letters of the tensor type:
// S - Storage = DType
// V - Variable
// T - Tensor
#[derive(Debug, Clone, Copy)]
pub struct AddBackwardSV<'g, S2> {
    ygrad: &'g RefCell<S2>,
}

impl<'g, S, S2> Backward<S> for AddBackwardSV<'g, S2>
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
    S: Clone,
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

impl<'g, S> Backward<S> for AddBackwardVS<'g, S>
where
    S: Default + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad);
    }
}

impl<'g, S, S2> Add<S2> for &'g Variable<S>
where
    S2: DType,
    S: Clone + Add<S2>,
{
    type Output = Tensor<<S as Add<S2>>::Output, AddBackwardVS<'g, S>>;
    fn add(self, rhs: S2) -> Self::Output {
        Tensor {
            data: self.data().clone() + rhs,
            grad_fn: AddBackwardVS {
                xgrad: &self.grad,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardVV<'g, S> {
    xgrad: &'g RefCell<S>,
    ygrad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for AddBackwardVV<'g, S>
where
    S: Default + Clone + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad.clone());
        self.ygrad.replace_take(|grad| grad + res_grad);
    }
}

impl<'g, S> Add<&'g Variable<S>> for &'g Variable<S>
where
    S: Clone + Add<Output = S>,
{
    type Output = Tensor<S, AddBackwardVV<'g, S>>;
    fn add(self, rhs: &'g Variable<S>) -> Self::Output {
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
pub struct AddBackwardVT<'g, S, YF> {
    xgrad: &'g RefCell<S>,
    ygrad_fn: YF,
}

impl<'g, S, YF> Backward<S> for AddBackwardVT<'g, S, YF>
where
    S: Default + Clone + Add<Output = S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad.clone());
        self.ygrad_fn.backward(res_grad);
    }
}

impl<'g, S, F> Add<Tensor<S, F>> for &'g Variable<S>
where
    S: Clone + Add<Output = S>,
{
    type Output = Tensor<S, AddBackwardVT<'g, S, F>>;
    fn add(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self.data().clone() + rhs.data,
            grad_fn: AddBackwardVT {
                xgrad: &self.grad,
                ygrad_fn: rhs.grad_fn,
            }
        }
    }
}

impl<S, F> Add<S> for Tensor<S, F>
where
    S: Add<Output = S>,
{
    type Output = Tensor<S, F>;
    fn add(self, rhs: S) -> Self::Output {
        Tensor {
            data: self.data + rhs,
            grad_fn: self.grad_fn,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardTV<'g, S, XF> {
    xgrad_fn: XF,
    ygrad: &'g RefCell<S>,
}

impl<'g, S, S2, XF> Backward<S> for AddBackwardTV<'g, S2, XF>
where
    S: Clone,
    S2: Default + Add<S, Output = S2>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad + res_grad.clone());
        self.xgrad_fn.backward(res_grad);
    }
}

impl<'g, S, XF> Add<&'g Variable<S>> for Tensor<S, XF>
where
    S: Clone + Add<Output = S>,
{
    type Output = Tensor<S, AddBackwardTV<'g, S, XF>>;
    fn add(self, rhs: &'g Variable<S>) -> Self::Output {
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
        // If res_grad is &S this is not a copy, but hey, advantages from passing S
        // by value and potentially doing operations in place are bigger than this copy
        // (at least hope so), if not, this will be changed, also, the buffer can implement
        // reference counting
        self.xgrad_fn.backward(res_grad.clone());
        self.ygrad_fn.backward(res_grad);
    }
}

impl<S, XF, YF> Add<Tensor<S, YF>> for Tensor<S, XF>
where
    S: Add<Output = S>,
{
    type Output = Tensor<S, AddBackwardTT<XF, YF>>;
    fn add(self, rhs: Tensor<S, YF>) -> Self::Output {
        Tensor {
            data: self.data + rhs.data,
            grad_fn: AddBackwardTT {
                xgrad_fn: self.grad_fn,
                ygrad_fn: rhs.grad_fn,
            }
        }
    }
}
