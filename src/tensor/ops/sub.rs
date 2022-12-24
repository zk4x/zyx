use crate::{tensor::{Tensor, Variable, Backward, GradientRef, GradAcc}, accel::cpu, dtype::SType, shape::Shape};
use core::ops::{Sub, Neg};
use duplicate::duplicate_item;

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardSV<'g, YG> {
    ygrad: GradientRef<'g, YG>,
}

impl<S, YG> Backward<S> for SubBackwardSV<'_, YG>
where
    S: Neg,
    YG: GradAcc<<S as Neg>::Output>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.accumulate(-res_grad);
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<'g, YS> Sub<&'g Variable<YS>> for dtype
where
    Self: Sub<YS>,
    YS: Clone + SType,
{
    type Output = Tensor<<Self as Sub<YS>>::Output, SubBackwardSV<'g, YS>>;
    fn sub(self, rhs: &'g Variable<YS>) -> Self::Output {
        Tensor {
            data: self - rhs.data.clone(),
            grad_fn: SubBackwardSV {
                ygrad: GradientRef::new(&rhs.grad),
            },
        }
    }
}

impl<'g, YS, T, Sh> Sub<&'g Variable<YS>> for cpu::Buffer<T, Sh>
where
    Sh: Shape,
    Self: Sub<YS>,
    YS: Clone + SType,
{
    type Output = Tensor<<Self as Sub<YS>>::Output, SubBackwardSV<'g, YS>>;
    fn sub(self, rhs: &'g Variable<YS>) -> Self::Output {
        Tensor {
            data: self - rhs.data.clone(),
            grad_fn: SubBackwardSV {
                ygrad: GradientRef::new(&rhs.grad),
            },
        }
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<S, F> Sub<Tensor<S, F>> for dtype
where
    Self: Sub<S>,
    S: SType,
{
    type Output = Tensor<<Self as Sub<S>>::Output, F>;
    fn sub(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self - rhs.data,
            grad_fn: rhs.grad_fn,
        }
    }
}

impl<S, F, T, Sh> Sub<Tensor<S, F>> for cpu::Buffer<T, Sh>
where
    Sh: Shape,
    Self: Sub<S>,
    S: SType,
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
pub struct SubBackwardVS<'g, XG> {
    xgrad: GradientRef<'g, XG>,
}

impl<S, XG> Backward<S> for SubBackwardVS<'_, XG>
where
    XG: GradAcc<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(res_grad);
    }
}

impl<'g, XS, YS> Sub<YS> for &'g Variable<XS>
where
    XS: Clone + Sub<YS>,
    YS: SType,
{
    type Output = Tensor<<XS as Sub<YS>>::Output, SubBackwardVS<'g, XS>>;
    fn sub(self, rhs: YS) -> Self::Output {
        Tensor {
            data: self.data.clone() - rhs,
            grad_fn: SubBackwardVS {
                xgrad: GradientRef::new(&self.grad),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardVV<'g, XG, YG> {
    xgrad: GradientRef<'g, XG>,
    ygrad: GradientRef<'g, YG>,
}

impl<S, XG, YG> Backward<S> for SubBackwardVV<'_, XG, YG>
where
    S: Clone + Neg,
    XG: GradAcc<S>,
    YG: GradAcc<<S as Neg>::Output>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(res_grad.clone());
        self.ygrad.accumulate(-res_grad);
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
            data: self.data.clone() - rhs.data.clone(),
            grad_fn: SubBackwardVV {
                xgrad: GradientRef::new(&self.grad),
                ygrad: GradientRef::new(&rhs.grad),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardVT<'g, XG, YF> {
    xgrad: GradientRef<'g, XG>,
    ygrad_fn: YF,
}

impl<S, XG, YF> Backward<S> for SubBackwardVT<'_, XG, YF>
where
    S: Clone + Neg,
    XG: GradAcc<S>,
    YF: Backward<<S as Neg>::Output>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(res_grad.clone());
        self.ygrad_fn.backward(-res_grad);
    }
}

impl<'g, XS, YS, YF> Sub<Tensor<YS, YF>> for &'g Variable<XS>
where
    XS: Clone + Sub<YS>,
{
    type Output = Tensor<<XS as Sub<YS>>::Output, SubBackwardVT<'g, XS, YF>>;
    fn sub(self, rhs: Tensor<YS, YF>) -> Self::Output {
        Tensor {
            data: self.data.clone() - rhs.data,
            grad_fn: SubBackwardVT {
                xgrad: GradientRef::new(&self.grad),
                ygrad_fn: rhs.grad_fn,
            },
        }
    }
}

impl<XS, YS, XF> Sub<YS> for Tensor<XS, XF>
where
    XS: Sub<YS>,
    YS: SType,
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
pub struct SubBackwardTV<'g, YG, XF> {
    xgrad_fn: XF,
    ygrad: GradientRef<'g, YG>,
}

impl<S, YG, XF> Backward<S> for SubBackwardTV<'_, YG, XF>
where
    S: Clone + Neg,
    XF: Backward<S>,
    YG: GradAcc<<S as Neg>::Output>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(res_grad.clone());
        self.ygrad.accumulate(-res_grad);
    }
}

impl<'g, XS, YS, XF> Sub<&'g Variable<YS>> for Tensor<XS, XF>
where
    XS: Clone + Sub<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as Sub<YS>>::Output, SubBackwardTV<'g, YS, XF>>;
    fn sub(self, rhs: &'g Variable<YS>) -> Self::Output {
        Tensor {
            data: self.data - rhs.data.clone(),
            grad_fn: SubBackwardTV {
                xgrad_fn: self.grad_fn,
                ygrad: GradientRef::new(&rhs.grad),
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
