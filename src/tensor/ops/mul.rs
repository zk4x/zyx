use crate::{tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, device::cpu, dtype::SType, shape::Shape};
use core::ops::Mul;
use duplicate::duplicate_item;

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardSV<'g, XS, YG> {
    xdata: XS,
    ygrad: GradientRef<'g, YG>,
}

impl<S, XS, YG> Backward<S> for MulBackwardSV<'_, XS, YG>
where
    XS: Mul<S>,
    YG: GradAcc<<XS as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.accumulate(self.xdata * res_grad);
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<'g, YS> Mul<&'g Variable<YS>> for dtype
where
    Self: Mul<YS>,
    YS: Clone,
{
    type Output = Tensor<<Self as Mul<YS>>::Output, MulBackwardSV<'g, Self, YS>>;
    fn mul(self, rhs: &'g Variable<YS>) -> Self::Output {
        Tensor {
            data: self * rhs.data.clone(),
            grad_fn: MulBackwardSV {
                xdata: self,
                ygrad: GradientRef::new(&rhs.grad),
            }
        }
    }
}

impl<'g, YS, Sh, T> Mul<&'g Variable<YS>> for cpu::Buffer<Sh, T>
where
    Sh: Shape,
    T: crate::dtype::DType,
    Self: Mul<YS>,
    YS: Clone,
{
    type Output = Tensor<<Self as Mul<YS>>::Output, MulBackwardSV<'g, Self, YS>>;
    fn mul(self, rhs: &'g Variable<YS>) -> Self::Output {
        Tensor {
            data: self.clone() * rhs.data.clone(),
            grad_fn: MulBackwardSV {
                xdata: self,
                ygrad: GradientRef::new(&rhs.grad),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardST<XS, YF> {
    xdata: XS,
    ygrad_fn: YF,
}

impl<S, XS, YF> Backward<S> for MulBackwardST<XS, YF>
where
    XS: Mul<S>,
    YF: Backward<<XS as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.ygrad_fn.backward(self.xdata * res_grad);
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool]; )]
impl<YS, YF> Mul<Tensor<YS, YF>> for dtype
where
    Self: Mul<YS>,
{
    type Output = Tensor<<Self as Mul<YS>>::Output, MulBackwardST<Self, YF>>;
    fn mul(self, rhs: Tensor<YS, YF>) -> Self::Output {
        Tensor {
            data: self * rhs.data,
            grad_fn: MulBackwardST {
                xdata: self,
                ygrad_fn: rhs.grad_fn,
            },
        }
    }
}

impl<S, F, Sh, T> Mul<Tensor<S, F>> for cpu::Buffer<Sh, T>
where
    Sh: Shape,
    T: crate::dtype::DType,
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
pub struct MulBackwardVS<'g, XG, YS> {
    xgrad: GradientRef<'g, XG>,
    ydata: YS,
}

impl<S, XG, YS> Backward<S> for MulBackwardVS<'_, XG, YS>
where
    YS: Mul<S>,
    XG: GradAcc<<YS as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(self.ydata * res_grad);
    }
}

impl<'g, XS, YS> Mul<YS> for &'g Variable<XS>
where
    YS: Clone + SType,
    XS: Clone + Mul<YS>,
{
    type Output = Tensor<<XS as Mul<YS>>::Output, MulBackwardVS<'g, XS, YS>>;
    fn mul(self, rhs: YS) -> Self::Output {
        Tensor {
            data: self.data.clone() * rhs.clone(),
            grad_fn: MulBackwardVS {
                xgrad: GradientRef::new(&self.grad),
                ydata: rhs,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardVV<'g, XS, XG, YS, YG> {
    xgrad: GradientRef<'g, XG>,
    xdata: XS,
    ygrad: GradientRef<'g, YG>,
    ydata: YS,
}

impl<S, XS, XG, YS, YG> Backward<S> for MulBackwardVV<'_, XS, XG, YS, YG>
where
    S: Clone,
    YS: Mul<S>,
    XS: Mul<S>,
    XG: GradAcc<<YS as Mul<S>>::Output>,
    YG: GradAcc<<XS as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(self.ydata * res_grad.clone());
        self.ygrad.accumulate(self.xdata * res_grad);
    }
}

impl<'g, XS, YS> Mul<&'g Variable<YS>> for &'g Variable<XS>
where
    XS: Clone + Mul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as Mul<YS>>::Output, MulBackwardVV<'g, XS, XS, YS, YS>>;
    fn mul(self, rhs: &'g Variable<YS>) -> Self::Output {
        Tensor {
            data: self.data.clone() * rhs.data.clone(),
            grad_fn: MulBackwardVV {
                xgrad: GradientRef::new(&self.grad),
                xdata: self.data.clone(),
                ygrad: GradientRef::new(&rhs.grad),
                ydata: rhs.data.clone(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulBackwardVT<'g, XS, XG, YS, YF> {
    xgrad: GradientRef<'g, XG>,
    xdata: XS,
    ygrad_fn: YF,
    ydata: YS,
}

impl<S, XS, XG, YS, YF> Backward<S> for MulBackwardVT<'_, XS, XG, YS, YF>
where
    S: Clone,
    YS: Mul<S>,
    XS: Mul<S>,
    XG: GradAcc<<YS as Mul<S>>::Output>,
    YF: Backward<<XS as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(self.ydata * res_grad.clone());
        self.ygrad_fn.backward(self.xdata * res_grad);
    }
}

impl<'g, XS, YS, F> Mul<Tensor<YS, F>> for &'g Variable<XS>
where
    XS: Clone + Mul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as Mul<YS>>::Output, MulBackwardVT<'g, XS, XS, YS, F>>;
    fn mul(self, rhs: Tensor<YS, F>) -> Self::Output {
        Tensor {
            data: self.data.clone() * rhs.data.clone(),
            grad_fn: MulBackwardVT {
                xgrad: GradientRef::new(&self.grad),
                xdata: self.data.clone(),
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
    YS: SType + Clone,
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
pub struct MulBackwardTV<'g, XS, YS, YG, XF> {
    xgrad_fn: XF,
    xdata: XS,
    ygrad: GradientRef<'g, YG>,
    ydata: YS,
}

impl<S, XS, YS, YG, XF> Backward<S> for MulBackwardTV<'_, XS, YS, YG, XF>
where
    S: Clone,
    XS: Mul<S>,
    YS: Mul<S>,
    XF: Backward<<YS as Mul<S>>::Output>,
    YG: GradAcc<<XS as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.accumulate(self.xdata * res_grad.clone());
        // this way it is tail recursive
        self.xgrad_fn.backward(self.ydata * res_grad);
    }
}

impl<'g, XS, YS, XF> Mul<&'g Variable<YS>> for Tensor<XS, XF>
where
    XS: Clone + Mul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as Mul<YS>>::Output, MulBackwardTV<'g, XS, YS, YS, XF>>;
    fn mul(self, rhs: &'g Variable<YS>) -> Self::Output {
        Tensor {
            data: self.data.clone() * rhs.data.clone(),
            grad_fn: MulBackwardTV {
                xgrad_fn: self.grad_fn,
                xdata: self.data,
                ygrad: GradientRef::new(&rhs.grad),
                ydata: rhs.data.clone(),
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
