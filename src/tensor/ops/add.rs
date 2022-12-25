use crate::{tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, dtype::SType, accel::cpu, shape::Shape};
use core::ops::Add;
use duplicate::duplicate_item;

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardSV<'g, YG> {
    ygrad: GradientRef<'g, YG>,
}

impl<S, YG> Backward<S> for AddBackwardSV<'_, YG>
where
    S: Add,
    YG: GradAcc<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.accumulate(res_grad);
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<'g, YS> Add<&'g Variable<YS>> for dtype
where
    Self: Add<YS>,
    YS: Clone + SType,
{
    type Output = Tensor<<Self as Add<YS>>::Output, AddBackwardSV<'g, YS>>;
    fn add(self, rhs: &'g Variable<YS>) -> Self::Output {
        Tensor {
            data: self + rhs.data.clone(),
            grad_fn: AddBackwardSV {
                ygrad: GradientRef::new(&rhs.grad),
            }
        }
    }
}

impl<'g, YS, Sh, T> Add<&'g Variable<YS>> for cpu::Buffer<Sh, T>
where
    Sh: Shape,
    Self: Add<YS>,
    YS: Clone,
{
    type Output = Tensor<<Self as Add<YS>>::Output, AddBackwardSV<'g, YS>>;
    fn add(self, rhs: &'g Variable<YS>) -> Self::Output {
        Tensor {
            data: self + rhs.data.clone(),
            grad_fn: AddBackwardSV {
                ygrad: GradientRef::new(&rhs.grad),
            }
        }
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<S, F> Add<Tensor<S, F>> for dtype
where
    Self: Add<S>,
    S: SType,
{
    type Output = Tensor<<Self as Add<S>>::Output, F>;
    fn add(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self + rhs.data,
            grad_fn: rhs.grad_fn,
        }
    }
}

impl<S, F, T, Sh> Add<Tensor<S, F>> for cpu::Buffer<T, Sh>
where
    Sh: Shape,
    Self: Add<S>,
    S: SType,
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
    xgrad: GradientRef<'g, XG>,
}

impl<S, XG> Backward<S> for AddBackwardVS<'_, XG>
where
    S: Add,
    XG: GradAcc<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(res_grad);
    }
}

impl<'g, XS, YS> Add<YS> for &'g Variable<XS>
where
    XS: Clone + Add<YS>,
    YS: SType,
{
    type Output = Tensor<<XS as Add<YS>>::Output, AddBackwardVS<'g, XS>>;
    fn add(self, rhs: YS) -> Self::Output {
        Tensor {
            data: self.data.clone() + rhs,
            grad_fn: AddBackwardVS {
                xgrad: GradientRef::new(&self.grad),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardVV<'g, XG, YG> {
    xgrad: GradientRef<'g, XG>,
    ygrad: GradientRef<'g, YG>,
}

impl<S, XG, YG> Backward<S> for AddBackwardVV<'_, XG, YG>
where
    S: Clone + Add,
    XG: GradAcc<S>,
    YG: GradAcc<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(res_grad.clone());
        self.ygrad.accumulate(res_grad);
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
            data: self.data.clone() + rhs.data.clone(),
            grad_fn: AddBackwardVV {
                xgrad: GradientRef::new(&self.grad),
                ygrad: GradientRef::new(&rhs.grad),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardVT<'g, XG, YF> {
    xgrad: GradientRef<'g, XG>,
    ygrad_fn: YF,
}

impl<S, XG, YF> Backward<S> for AddBackwardVT<'_, XG, YF>
where
    S: Clone + Add,
    XG: GradAcc<S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(res_grad.clone());
        self.ygrad_fn.backward(res_grad);
    }
}

impl<'g, XS, YS, YF> Add<Tensor<YS, YF>> for &'g Variable<XS>
where
    XS: Clone + Add<YS>,
    YS: 'g,
{
    type Output = Tensor<<XS as Add<YS>>::Output, AddBackwardVT<'g, XS, YF>>;
    fn add(self, rhs: Tensor<YS, YF>) -> Self::Output {
        Tensor {
            data: self.data.clone() + rhs.data,
            grad_fn: AddBackwardVT {
                xgrad: GradientRef::new(&self.grad),
                ygrad_fn: rhs.grad_fn,
            }
        }
    }
}

impl<XS, YS, F> Add<YS> for Tensor<XS, F>
where
    XS: Add<YS> + SType,
    YS: SType,
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
    ygrad: GradientRef<'g, YG>,
}

impl<S, XF, YG> Backward<S> for AddBackwardTV<'_, YG, XF>
where
    S: Clone + Add,
    XF: Backward<S>,
    YG: GradAcc<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.accumulate(res_grad.clone());
        self.xgrad_fn.backward(res_grad);
    }
}

impl<'g, XS, YS, XF> Add<&'g Variable<YS>> for Tensor<XS, XF>
where
    XS: Add<YS> + SType,
    YS: Clone + SType,
{
    type Output = Tensor<<XS as Add<YS>>::Output, AddBackwardTV<'g, YS, XF>>;
    fn add(self, rhs: &'g Variable<YS>) -> Self::Output {
        Tensor {
            data: self.data + rhs.data.clone(),
            grad_fn: AddBackwardTV {
                xgrad_fn: self.grad_fn,
                ygrad: GradientRef::new(&rhs.grad),
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
    XS: Add<YS> + SType,
    YS: SType,
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
