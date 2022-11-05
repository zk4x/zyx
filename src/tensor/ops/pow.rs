use crate::{ops::{Pow, Ln}, tensor::{Variable, Tensor, Backward, GradientRef}, dtype::DType};
use std::{ops::{Add, Mul, Div}};

/*#[derive(Debug, Clone, Copy)]
pub struct PowBackwardSV<'g, YG, YT> {
    ygrad: GradientRef<'g, YG>,
    ytemp: YT,
}

impl<S, YG, YT> Backward<S> for PowBackwardSV<'_, YG, YT>
where
    S: Mul<YT, Output = YG>,
    YG: Add<YG, Output = YG>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.accumulate(res_grad * self.ytemp);
    }
}

impl<'g, XS, YS, YG> Pow<&'g Variable<YS, YG>> for XS
where
    XS: Clone + Pow<YS> + Ln + DType,
    YS: Clone,
    <XS as Pow<YS>>::Output: Clone + Mul<<XS as Ln>::Output>,
{
    type Output = Tensor<<XS as Pow<YS>>::Output, PowBackwardSV<'g, YG, <<XS as Pow<YS>>::Output as Mul<<XS as Ln>::Output>>::Output>>;
    fn pow(self, rhs: &'g Variable<YS, YG>) -> Self::Output {
        let res = self.clone().pow(rhs.data().clone());
        Tensor {
            data: res.clone(),
            grad_fn: PowBackwardSV {
                ygrad: &rhs.grad,
                ytemp: res * self.ln(),
            }
        }
    }
}*/

#[derive(Debug, Clone, Copy)]
pub struct PowBackwardST<S, YF> {
    ygrad_fn: YF,
    ytemp: S,
}

impl<YF, S> Backward<S> for PowBackwardST<S, YF>
where
    S: Default + Add<Output = S> + Mul<Output = S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad_fn.backward(res_grad * self.ytemp);
    }
}

impl<S, F> Pow<Tensor<S, F>> for S
where
    S: Clone + Pow<Output = S> + Ln<Output = S> + Mul<Output = S>,
{
    type Output = Tensor<S, PowBackwardST<S, F>>;
    fn pow(self, rhs: Tensor<S, F>) -> Self::Output {
        let res = self.clone().pow(rhs.data);
        Tensor {
            data: res.clone(),
            grad_fn: PowBackwardST {
                ygrad_fn: rhs.grad_fn,
                ytemp: res * self.ln(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PowBackwardVS<'g, XG, XT> {
    xgrad: GradientRef<'g, XG>,
    xtemp: XT,
}

// TODO: rewrite everything like this
impl<S, XG, XT> Backward<S> for PowBackwardVS<'_, XG, XT>
where
    XT: Mul<S, Output = XG>,
    XG: Add<XG, Output = XG>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(self.xtemp * res_grad);
    }
}

impl<'g, XS, XG, YS> Pow<YS> for &'g Variable<XS, XG>
where
    XS: Clone + Pow<YS>,
    YS: Clone + DType + Mul<<XS as Pow<YS>>::Output>,
    <YS as Mul<<XS as Pow<YS>>::Output>>::Output: Div<XS>,
    <XS as Pow<YS>>::Output: Clone,
{
    type Output = Tensor<<XS as Pow<YS>>::Output, PowBackwardVS<'g, XG, <<YS as Mul<<XS as Pow<YS>>::Output>>::Output as Div<XS>>::Output>>;
    fn pow(self, rhs: YS) -> Self::Output {
        let res = self.data().clone().pow(rhs.clone());
        Tensor {
            data: res.clone(),
            grad_fn: PowBackwardVS {
                xgrad: GradientRef::new(&self.grad),
                xtemp: rhs * res/self.data().clone(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PowBackwardVV<'g, XG, XT, YG, YT> {
    xgrad: GradientRef<'g, XG>,
    xtemp: XT,
    ygrad: GradientRef<'g, YG>,
    ytemp: YT,
}

impl<S, XG, XT, YG, YT> Backward<S> for PowBackwardVV<'_, XG, XT, YG, YT>
where
    S: Clone,
    XT: Mul<S, Output = XG>,
    XG: Add<XG, Output = XG>,
    YT: Mul<S, Output = YG>,
    YG: Add<YG, Output = YG>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(self.xtemp * res_grad.clone());
        self.ygrad.accumulate(self.ytemp * res_grad);
    }
}

impl<'g, XS, XG, YS, YG> Pow<&'g Variable<YS, YG>> for &'g Variable<XS, XG>
where
    XS: Clone + Pow<YS> + Ln,
    YS: Clone + Mul<<XS as Pow<YS>>::Output>,
    <XS as Pow<YS>>::Output: Clone,
    <YS as Mul<<XS as Pow<YS>>::Output>>::Output: Div<XS>,
    <XS as Pow<YS>>::Output: Mul<<XS as Ln>::Output>,
{
    type Output = Tensor<<XS as Pow<YS>>::Output, PowBackwardVV<'g, XG, <<YS as Mul<<XS as Pow<YS>>::Output>>::Output as Div<XS>>::Output, YG, <<XS as Pow<YS>>::Output as Mul<<XS as Ln>::Output>>::Output>>;
    fn pow(self, rhs: &'g Variable<YS, YG>) -> Self::Output {
        let res = self.data().clone().pow(rhs.data().clone());
        Tensor {
            data: res.clone(),
            grad_fn: PowBackwardVV {
                xgrad: GradientRef::new(&self.grad),
                xtemp: rhs.data().clone() * res.clone()/self.data().clone(),
                ygrad: GradientRef::new(&rhs.grad),
                ytemp: res * self.data().clone().ln(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PowBackwardVT<'g, XG, XT, YT, YF> {
    xgrad: GradientRef<'g, XG>,
    xtemp: XT,
    ygrad_fn: YF,
    ytemp: YT,
}

impl<S, XG, XT, YT, YF> Backward<S> for PowBackwardVT<'_, XG, XT, YT, YF>
where
    S: Clone,
    XT: Mul<S, Output = XG>,
    XG: Add<XG, Output = XG>,
    YT: Mul<S>,
    YF: Backward<<YT as Mul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(self.xtemp * res_grad.clone());
        self.ygrad_fn.backward(self.ytemp * res_grad);
    }
}

impl<'g, XS, XG, YS, YF> Pow<Tensor<YS, YF>> for &'g Variable<XS, XG>
where
    XS: Clone + Pow<YS> + Ln,
    YS: Clone + Mul<<XS as Pow<YS>>::Output>,
    <XS as Pow<YS>>::Output: Clone,
    <YS as Mul<<XS as Pow<YS>>::Output>>::Output: Div<XS>,
    <XS as Pow<YS>>::Output: Mul<<XS as Ln>::Output>,
{
    type Output = Tensor<<XS as Pow<YS>>::Output, PowBackwardVT<'g, XG, <<YS as Mul<<XS as Pow<YS>>::Output>>::Output as Div<XS>>::Output, <<XS as Pow<YS>>::Output as Mul<<XS as Ln>::Output>>::Output, YF>>;
    fn pow(self, rhs: Tensor<YS, YF>) -> Self::Output {
        let res = self.data().clone().pow(rhs.data.clone());
        Tensor {
            data: res.clone(),
            grad_fn: PowBackwardVT {
                xgrad: GradientRef::new(&self.grad),
                xtemp: rhs.data * res.clone()/self.data().clone(),
                ygrad_fn: rhs.grad_fn,
                ytemp: res * self.data().clone().ln(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PowBackwardTS<S2, XF> {
    xgrad_fn: XF,
    xtemp: S2,
}

impl<S, S2, XF> Backward<S> for PowBackwardTS<S2, XF>
where
    S2: Default + Mul<S, Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(self.xtemp * res_grad);
    }
}

impl<S, S2, F> Pow<S2> for Tensor<S, F>
where
    S: Clone + Pow<S2>,
    S2: DType + Clone + Mul<<S as Pow<S2>>::Output>,
    <S as Pow<S2>>::Output: Clone,
    <S2 as Mul<<S as Pow<S2>>::Output>>::Output: Div<S>,
{
    type Output = Tensor<<S as Pow<S2>>::Output, PowBackwardTS<<<S2 as Mul<<S as Pow<S2>>::Output>>::Output as Div<S>>::Output, F>>;
    fn pow(self, rhs: S2) -> Self::Output {
        let res = self.data.clone().pow(rhs.clone());
        Tensor {
            data: res.clone(),
            grad_fn: PowBackwardTS {
                xgrad_fn: self.grad_fn,
                xtemp: rhs * res/self.data,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PowBackwardTV<'g, YG, XT, YT, XF> {
    xgrad_fn: XF,
    xtemp: XT,
    ygrad: GradientRef<'g, YG>,
    ytemp: YT,
}

impl<S, YG, XT, YT, XF> Backward<S> for PowBackwardTV<'_, YG, XT, YT, XF>
where
    S: Clone,
    YT: Mul<S, Output = YG>,
    XT: Mul<S>,
    XF: Backward<<XT as Mul<S>>::Output>,
    YG: Add<YG, Output = YG>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.accumulate(self.ytemp * res_grad.clone());
        self.xgrad_fn.backward(self.xtemp * res_grad);
    }
}

impl<'g, XS, YS, YG, XF> Pow<&'g Variable<YS, YG>> for Tensor<XS, XF>
where
    XS: Clone + Pow<YS> + Ln,
    YS: Clone + Mul<<XS as Pow<YS>>::Output>,
    <XS as Pow<YS>>::Output: Clone,
    <YS as Mul<<XS as Pow<YS>>::Output>>::Output: Div<XS>,
    <XS as Pow<YS>>::Output: Mul<<XS as Ln>::Output>,
{
    type Output = Tensor<<XS as Pow<YS>>::Output, PowBackwardTV<'g, YG, <<YS as Mul<<XS as Pow<YS>>::Output>>::Output as Div<XS>>::Output, <<XS as Pow<YS>>::Output as Mul<<XS as Ln>::Output>>::Output, XF>>;
    fn pow(self, rhs: &'g Variable<YS, YG>) -> Self::Output {
        let res = self.data.clone().pow(rhs.data().clone());
        Tensor {
            data: res.clone(),
            grad_fn: PowBackwardTV {
                xgrad_fn: self.grad_fn,
                xtemp: rhs.data().clone() * res.clone()/self.data.clone(),
                ygrad: GradientRef::new(&rhs.grad),
                ytemp: res * self.data.ln(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PowBackwardTT<S, XF, YF> {
    xgrad_fn: XF,
    xtemp: S,
    ygrad_fn: YF,
    ytemp: S,
}

impl<S, XF, YF> Backward<S> for PowBackwardTT<S, XF, YF>
where
    S: Clone + Mul<Output = S>,
    XF: Backward<S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad_fn.backward(self.ytemp * res_grad.clone());
        self.xgrad_fn.backward(self.xtemp * res_grad);
    }
}

impl<S, XF, YF> Pow<Tensor<S, YF>> for Tensor<S, XF>
where
    S: Clone + Mul<Output = S> + Div<Output = S> + Pow<Output = S> + Ln<Output = S>,
{
    type Output = Tensor<S, PowBackwardTT<S, XF, YF>>;
    fn pow(self, rhs: Tensor<S, YF>) -> Self::Output {
        let res = self.data.clone().pow(rhs.data.clone());
        Tensor {
            data: res.clone(),
            grad_fn: PowBackwardTT {
                xgrad_fn: self.grad_fn,
                xtemp: rhs.data().clone() * res.clone()/self.data.clone(),
                ygrad_fn: rhs.grad_fn,
                ytemp: res * self.data.ln(),
            }
        }
    }
}
