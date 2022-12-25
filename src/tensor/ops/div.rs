use crate::{tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, dtype::SType};
use core::ops::{Neg, Mul, Div};
//use duplicate::duplicate_item;

/*#[derive(Debug, Clone, Copy)]
pub struct DivBackwardSV<'g, S, S2> {
    res: S2,
    ygrad: GradientRef<'g, G>,
    ydata: S,
}

impl<S, S2> Backward<S> for DivBackwardSV<'_, S, S2>
where
    S: Default + Sub<<<S2 as Div<S>>::Output as Mul<S>>::Output, Output = S>,
    S2: Div<S>,
    <S2 as Div<S>>::Output: Mul<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.accumulate(grad - self.res / self.ydata * res_grad);
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];
    [cpu::Buffer<f32>]; [cpu::Buffer<f64>]; [cpu::Buffer<i32>]; [cpu::Buffer<i64>]; [cpu::Buffer<i128>];
    [cpu::Buffer<u8>]; [cpu::Buffer<u16>]; [cpu::Buffer<u32>]; [cpu::Buffer<u64>]; [cpu::Buffer<u128>]; [cpu::Buffer<bool>];)]
impl<'g, S> Div<&'g Variable<S>> for dtype
where
    S: Clone + SType,
    Self: Div<S>,
    <Self as Div<S>>::Output: Clone,
{
    type Output = Tensor<<Self as Div<S>>::Output, DivBackwardSV<'g, S, <Self as Div<S>>::Output>>;
    fn div(self, rhs: &'g Variable<S>) -> Self::Output {
        let res = self / rhs.data.clone();
        Tensor {
            data: res.clone(),
            grad_fn: DivBackwardSV {
                res,
                ygrad: &rhs.grad,
                ydata: rhs.data.clone(),
            }
        }
    }
}*/

/*#[derive(Debug, Clone, Copy)]
pub struct DivBackwardST<S, YS, YF> {
    res: S,
    ygrad_fn: YF,
    ydata: YS,
}

impl<S, YS, S3, YF> Backward<S3> for DivBackwardST<S, YS, YF>
where
    S: Neg,
    <S as Neg>::Output: Div<YS>,
    <<S as Neg>::Output as Div<YS>>::Output: Mul<S3, Output = YS>,
    YF: Backward<YS>,
{
    fn backward(self, res_grad: S3) {
        self.ygrad_fn.backward(-self.res / self.ydata * res_grad);
    }
}*/

/*#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];
    [cpu::Buffer<f32>]; [cpu::Buffer<f64>]; [cpu::Buffer<i32>]; [cpu::Buffer<i64>]; [cpu::Buffer<i128>];
    [cpu::Buffer<u8>]; [cpu::Buffer<u16>]; [cpu::Buffer<u32>]; [cpu::Buffer<u64>]; [cpu::Buffer<u128>]; [cpu::Buffer<bool>];)]
impl<YS, F> Div<Tensor<YS, F>> for dtype
where
    YS: Clone + SType,
    Self: Div<YS>,
    <Self as Div<YS>>::Output: Clone,
{
    type Output = Tensor<<Self as Div<YS>>::Output, DivBackwardST<<Self as Div<YS>>::Output, YS, F>>;
    fn div(self, rhs: Tensor<YS, F>) -> Self::Output {
        let res = self / rhs.data.clone();
        Tensor {
            data: res.clone(),
            grad_fn: DivBackwardST {
                res,
                ygrad_fn: rhs.grad_fn,
                ydata: rhs.data,
            },
        }
    }
}*/

// TODO: remove this and put it into the implementation above.
// And figure why the above implementation overflows during compilation.
/*impl<F, Sh> Div<Tensor<cpu::Buffer<T, Sh>, F>> for i32
where
    Sh: Shape,
{
    type Output = Tensor<<Self as Div<cpu::Buffer<f32, Sh>>>::Output, DivBackwardST<<Self as Div<cpu::Buffer<f32, Sh>>>::Output, cpu::Buffer<f32, Sh>, F>>;
    fn div(self, rhs: Tensor<cpu::Buffer<f32, Sh>, F>) -> Self::Output {
        let res = self / rhs.data.clone();
        Tensor {
            data: res.clone(),
            grad_fn: DivBackwardST {
                res,
                ygrad_fn: rhs.grad_fn,
                ydata: rhs.data,
            },
        }
    }
}*/

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardVS<'g, XG, YS> {
    xgrad: GradientRef<'g, XG>,
    ydata: YS,
}

impl<S, XG, YS> Backward<S> for DivBackwardVS<'_, XG, YS>
where
    S: Div<YS>,
    XG: GradAcc<<S as Div<YS>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(res_grad / self.ydata);
    }
}

impl<'g, XS, YS> Div<YS> for &'g Variable<XS>
where
    XS: Clone + Div<YS>,
    YS: Clone + SType,
{
    type Output = Tensor<<XS as Div<YS>>::Output, DivBackwardVS<'g, XS, YS>>;
    fn div(self, rhs: YS) -> Self::Output {
        Tensor {
            data: self.data.clone() / rhs.clone(),
            grad_fn: DivBackwardVS {
                xgrad: GradientRef::new(&self.grad),
                ydata: rhs,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardVV<'g, S, XG, YS, YG> {
    res: S,
    xgrad: GradientRef<'g, XG>,
    ygrad: GradientRef<'g, YG>,
    ydata: YS,
}

impl<S, S2, XG, YS, YG> Backward<S> for DivBackwardVV<'_, S2, XG, YS, YG>
where
    S: Div<YS>,
    <S as Div<YS>>::Output: Clone,
    S2: Mul<<S as Div<YS>>::Output>,
    <S2 as Mul<<S as Div<YS>>::Output>>::Output: Neg,

    XG: GradAcc<<S as Div<YS>>::Output>,
    YG: GradAcc<<<S2 as Mul<<S as Div<YS>>::Output>>::Output as Neg>::Output>,
{
    fn backward(self, res_grad: S) {
        let temp = res_grad / self.ydata;
        self.xgrad.accumulate(temp.clone());
        self.ygrad.accumulate(-(self.res * temp));
    }
}

impl<'g, XS, YS> Div<&'g Variable<YS>> for &'g Variable<XS>
where
    XS: Clone + Div<YS> + SType,
    YS: Clone + SType,
    <XS as Div<YS>>::Output: Clone,
{
    type Output = Tensor<<XS as Div<YS>>::Output, DivBackwardVV<'g, <XS as Div<YS>>::Output, XS, YS, YS>>;
    fn div(self, rhs: &'g Variable<YS>) -> Self::Output {
        let res = self.data.clone() / rhs.data.clone();
        Tensor {
            data: res.clone(),
            grad_fn: DivBackwardVV {
                xgrad: GradientRef::new(&self.grad),
                res,
                ygrad: GradientRef::new(&rhs.grad),
                ydata: rhs.data.clone(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardVT<'g, S, XG, YS, YF> {
    res: S,
    xgrad: GradientRef<'g, XG>,
    ygrad_fn: YF,
    ydata: YS,
}

impl<S, S2, XG, YS, YF> Backward<S> for DivBackwardVT<'_, S2, XG, YS, YF>
where
    S: Div<YS>,
    <S as Div<YS>>::Output: Clone,
    S2: Mul<<S as Div<YS>>::Output>,
    <S2 as Mul<<S as Div<YS>>::Output>>::Output: Neg,
    XG: GradAcc<<S as Div<YS>>::Output>,
    YF: Backward<<<S2 as Mul<<S as Div<YS>>::Output>>::Output as Neg>::Output>,
{
    fn backward(self, res_grad: S) {
        let temp = res_grad / self.ydata;
        self.xgrad.accumulate(temp.clone());
        self.ygrad_fn.backward(-(self.res * temp));
    }
}

impl<'g, XS, YS, YF> Div<Tensor<YS, YF>> for &'g Variable<XS>
where
    XS: Clone + Div<YS>,
    YS: Clone,
    <XS as Div<YS>>::Output: Clone,
{
    type Output = Tensor<<XS as Div<YS>>::Output, DivBackwardVT<'g, <XS as Div<YS>>::Output, XS, YS, YF>>;
    fn div(self, rhs: Tensor<YS, YF>) -> Self::Output {
        let res = self.data.clone() / rhs.data.clone();
        Tensor {
            data: res.clone(),
            grad_fn: DivBackwardVT {
                res,
                xgrad: GradientRef::new(&self.grad),
                ygrad_fn: rhs.grad_fn,
                ydata: rhs.data,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardTS<S, XF> {
    xgrad_fn: XF,
    ydata: S,
}

impl<S, S2, XF> Backward<S2> for DivBackwardTS<S, XF>
where
    S2: Div<S>,
    XF: Backward<<S2 as Div<S>>::Output>,
{
    fn backward(self, res_grad: S2) {
        self.xgrad_fn.backward(res_grad / self.ydata);
    }
}

impl<S, S2, F> Div<S2> for Tensor<S, F>
where
    S2: SType + Clone,
    S: Clone + Div<S2>,
{
    type Output = Tensor<<S as Div<S2>>::Output, DivBackwardTS<S2, F>>;
    fn div(self, rhs: S2) -> Self::Output {
        Tensor {
            data: self.data / rhs.clone(),
            grad_fn: DivBackwardTS {
                xgrad_fn: self.grad_fn,
                ydata: rhs,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardTV<'g, S, YS, YG, XF> {
    res: S,
    xgrad_fn: XF,
    ygrad: GradientRef<'g, YG>,
    ydata: YS,
}

impl<S, S2, YS, YG, XF> Backward<S> for DivBackwardTV<'_, S2, YS, YG, XF>
where
    S: Div<YS>,
    <S as Div<YS>>::Output: Clone,
    S2: Mul<<S as Div<YS>>::Output>,
    <S2 as Mul<<S as Div<YS>>::Output>>::Output: Neg,
    XF: Backward<<S as Div<YS>>::Output>,
    YG: GradAcc<<<S2 as Mul<<S as Div<YS>>::Output>>::Output as Neg>::Output>,
{
    fn backward(self, res_grad: S) {
        let temp = res_grad / self.ydata;
        self.ygrad.accumulate(-(self.res * temp.clone()));
        self.xgrad_fn.backward(temp);
    }
}

impl<'g, XS, YS, XF> Div<&'g Variable<YS>> for Tensor<XS, XF>
where
    XS: Div<YS> + SType,
    YS: Clone + SType,
    <XS as Div<YS>>::Output: Clone,
{
    type Output = Tensor<<XS as Div<YS>>::Output, DivBackwardTV<'g, <XS as Div<YS>>::Output, YS, YS, XF>>;
    fn div(self, rhs: &'g Variable<YS>) -> Self::Output {
        let res = self.data / rhs.data.clone();
        Tensor {
            data: res.clone(),
            grad_fn: DivBackwardTV {
                res,
                xgrad_fn: self.grad_fn,
                ygrad: GradientRef::new(&rhs.grad),
                ydata: rhs.data.clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DivBackwardTT<S, S2, XF, YF> {
    res: S2,
    xgrad_fn: XF,
    ygrad_fn: YF,
    ydata: S,
}

impl<S, S2, S3, XF, YF> Backward<S> for DivBackwardTT<S2, S3, XF, YF>
where
    S: Div<S2>,
    <S as Div<S2>>::Output: Clone,
    S3: Neg,
    <S3 as Neg>::Output: Mul<<S as Div<S2>>::Output>,
    XF: Backward<<S as Div<S2>>::Output>,
    YF: Backward<<<S3 as Neg>::Output as Mul<<S as Div<S2>>::Output>>::Output>,
{
    fn backward(self, res_grad: S) {
        let temp = res_grad / self.ydata;
        self.xgrad_fn.backward(temp.clone());
        self.ygrad_fn.backward(- self.res * temp);
    }
}

impl<XS, YS, XF, YF> Div<Tensor<YS, YF>> for Tensor<XS, XF>
where
    XS: Div<YS>,
    YS: Clone,
    <XS as Div<YS>>::Output: Clone,
{
    type Output = Tensor<<XS as Div<YS>>::Output, DivBackwardTT<YS, <XS as Div<YS>>::Output, XF, YF>>;
    fn div(self, rhs: Tensor<YS, YF>) -> Self::Output {
        let res = self.data / rhs.data.clone();
        Tensor {
            data: res.clone(),
            grad_fn: DivBackwardTT {
                res,
                xgrad_fn: self.grad_fn,
                ygrad_fn: rhs.grad_fn,
                ydata: rhs.data,
            }
        }
    }
}
