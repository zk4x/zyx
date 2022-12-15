use crate::{ops::{MatMul, Transpose}, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, dtype::DType};

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardSV<'g, XS, YG> {
    xdata: XS,
    ygrad: GradientRef<'g, YG>,
}

impl<S, XS, YG> Backward<S> for MatMulBackwardSV<'_, XS, YG>
where
    XS: Transpose,
    <XS as Transpose>::Output: MatMul<S>,
    YG: GradAcc<<<XS as Transpose>::Output as MatMul<S>>::Output>
{
    fn backward(self, res_grad: S) {
        self.ygrad.accumulate(self.xdata.transpose().matmul(res_grad));
    }
}

impl<'g, XS, YS> MatMul<&'g Variable<YS>> for XS
where
    XS: Clone + MatMul<YS> + DType,
    YS: Clone,
{
    type Output = Tensor<<XS as MatMul<YS>>::Output, MatMulBackwardSV<'g, XS, YS>>;
    fn matmul(self, rhs: &'g Variable<YS>) -> Self::Output {
        Tensor {
            data: self.clone().matmul(rhs.data.clone()),
            grad_fn: MatMulBackwardSV {
                xdata: self,
                ygrad: GradientRef::new(&rhs.grad),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardST<XS, YF> {
    xdata: XS,
    ygrad_fn: YF,
}

impl<S, XS, YF> Backward<S> for MatMulBackwardST<XS, YF>
where
    XS: Transpose,
    <XS as Transpose>::Output: MatMul<S>,
    YF: Backward<<<XS as Transpose>::Output as MatMul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.ygrad_fn.backward(self.xdata.transpose().matmul(res_grad));
    }
}

impl<XS, YS, YF> MatMul<Tensor<YS, YF>> for XS
where
    XS: Clone + MatMul<YS> + DType,
    YS: Clone,
{
    type Output = Tensor<<XS as MatMul<YS>>::Output, MatMulBackwardST<XS, YF>>;
    fn matmul(self, rhs: Tensor<YS, YF>) -> Self::Output {
        Tensor {
            data: self.clone().matmul(rhs.data),
            grad_fn: MatMulBackwardST {
                xdata: self,
                ygrad_fn: rhs.grad_fn,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardVS<'g, XG, YS> {
    xgrad: GradientRef<'g, XG>,
    ydata: YS,
}

impl<S, XG, YS> Backward<S> for MatMulBackwardVS<'_, XG, YS>
where
    YS: Transpose,
    S: MatMul<<YS as Transpose>::Output>,
    XG: GradAcc<<S as MatMul<<YS as Transpose>::Output>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.accumulate(res_grad.matmul(self.ydata.transpose()));
    }
}

impl<'g, XS, YS> MatMul<YS> for &'g Variable<XS>
where
    XS: Clone + MatMul<YS>,
    YS: Clone + DType,
{
    type Output = Tensor<<XS as MatMul<YS>>::Output, MatMulBackwardVS<'g, XS, YS>>;
    fn matmul(self, rhs: YS) -> Self::Output {
        Tensor {
            data: self.data.clone().matmul(rhs.clone()),
            grad_fn: MatMulBackwardVS {
                xgrad: GradientRef::new(&self.grad),
                ydata: rhs,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardVV<'g, XS, XG, YS, YG> {
    xgrad: GradientRef<'g, XG>,
    xdata: XS,
    ygrad: GradientRef<'g, YG>,
    ydata: YS,
}

impl<S, XS, XG, YS, YG> Backward<S> for MatMulBackwardVV<'_, XS, XG, YS, YG>
where
    S: Clone,
    XS: Transpose,
    <XS as Transpose>::Output: MatMul<S>,
    YS: Transpose,
    S: MatMul<<YS as Transpose>::Output>,

    XG: GradAcc<<S as MatMul<<YS as Transpose>::Output>>::Output>,
    YG: GradAcc<<<XS as Transpose>::Output as MatMul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.accumulate(self.xdata.transpose().matmul(res_grad.clone()));
        self.xgrad.accumulate(res_grad.matmul(self.ydata.transpose()));
    }
}

impl<'g, XS, YS> MatMul<&'g Variable<YS>> for &'g Variable<XS>
where
    XS: Clone + MatMul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as MatMul<YS>>::Output, MatMulBackwardVV<'g, XS, XS, YS, YS>>;
    fn matmul(self, rhs: &'g Variable<YS>) -> Self::Output {
        Tensor {
            data: self.data.clone().matmul(rhs.data.clone()),
            grad_fn: MatMulBackwardVV {
                xgrad: GradientRef::new(&self.grad),
                xdata: self.data.clone(),
                ygrad: GradientRef::new(&rhs.grad),
                ydata: rhs.data.clone(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardVT<'g, XG, XS, YS, YF> {
    xgrad: GradientRef<'g, XG>,
    xdata: XS,
    ygrad_fn: YF,
    ydata: YS,
}

impl<S, XG, XS, YS, YF> Backward<S> for MatMulBackwardVT<'_, XG, XS, YS, YF>
where
    S: Clone + Transpose,
    XS: Transpose,
    <XS as Transpose>::Output: MatMul<S>,
    <S as Transpose>::Output: MatMul<YS>,

    XG: GradAcc<<<S as Transpose>::Output as MatMul<YS>>::Output>,
    YF: Backward<<<XS as Transpose>::Output as MatMul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.ygrad_fn.backward(self.xdata.transpose().matmul(res_grad.clone()));
        self.xgrad.accumulate(res_grad.transpose().matmul(self.ydata));
    }
}

impl<'g, XS, YS, YF> MatMul<Tensor<YS, YF>> for &'g Variable<XS>
where
    XS: Clone + MatMul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as MatMul<YS>>::Output, MatMulBackwardVT<'g, XS, XS, YS, YF>>;
    fn matmul(self, rhs: Tensor<YS, YF>) -> Self::Output {
        Tensor {
            data: self.data.clone().matmul(rhs.data.clone()),
            grad_fn: MatMulBackwardVT {
                xgrad: GradientRef::new(&self.grad),
                xdata: self.data.clone(),
                ygrad_fn: rhs.grad_fn,
                ydata: rhs.data,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardTS<YS, XF> {
    xgrad_fn: XF,
    ydata: YS,
}

impl<S, YS, XF> Backward<S> for MatMulBackwardTS<YS, XF>
where
    YS: Transpose,
    <YS as Transpose>::Output: MatMul<S>,
    XF: Backward<<<YS as Transpose>::Output as MatMul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(self.ydata.transpose().matmul(res_grad));
    }
}

impl<XS, YS, XF> MatMul<YS> for Tensor<XS, XF>
where
    XS: MatMul<YS>,
    YS: Clone + DType,
{
    type Output = Tensor<<XS as MatMul<YS>>::Output, MatMulBackwardTS<YS, XF>>;
    fn matmul(self, rhs: YS) -> Self::Output {
        Tensor {
            data: self.data.matmul(rhs.clone()),
            grad_fn: MatMulBackwardTS {
                xgrad_fn: self.grad_fn,
                ydata: rhs,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardTV<'g, YG, XS, YS, XF> {
    xgrad_fn: XF,
    xdata: XS,
    ygrad: GradientRef<'g, YG>,
    ydata: YS,
}

impl<S, YG, XS, YS, XF> Backward<S> for MatMulBackwardTV<'_, YG, XS, YS, XF>
where
    YS: Transpose,
    S: Clone + MatMul<<YS as Transpose>::Output>,
    XS: Transpose,
    <XS as Transpose>::Output: MatMul<S>,

    XF: Backward<<S as MatMul<<YS as Transpose>::Output>>::Output>,
    YG: GradAcc<<<XS as Transpose>::Output as MatMul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(res_grad.clone().matmul(self.ydata.transpose()));
        self.ygrad.accumulate(self.xdata.transpose().matmul(res_grad));
    }
}

impl<'g, XS, XF, YS> MatMul<&'g Variable<YS>> for Tensor<XS, XF>
where
    XS: Clone + MatMul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as MatMul<YS>>::Output, MatMulBackwardTV<'g, YS, XS, YS, XF>>;
    fn matmul(self, rhs: &'g Variable<YS>) -> Self::Output {
        Tensor {
            data: self.data.clone().matmul(rhs.data.clone()),
            grad_fn: MatMulBackwardTV {
                xgrad_fn: self.grad_fn,
                xdata: self.data,
                ygrad: GradientRef::new(&rhs.grad),
                ydata: rhs.data.clone(),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulBackwardTT<XS, YS, XF, YF> {
    xgrad_fn: XF,
    xdata: XS,
    ygrad_fn: YF,
    ydata: YS,
}

impl<S, XS, YS, XF, YF> Backward<S> for MatMulBackwardTT<XS, YS, XF, YF>
where
    S: Clone + MatMul<<YS as Transpose>::Output>,
    YS: Transpose,
    XF: Backward<<S as MatMul<<YS as Transpose>::Output>>::Output>,
    XS: Transpose,
    <XS as Transpose>::Output: MatMul<S>,
    YF: Backward<<<XS as Transpose>::Output as MatMul<S>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.xgrad_fn.backward(res_grad.clone().matmul(self.ydata.transpose()));
        self.ygrad_fn.backward(self.xdata.transpose().matmul(res_grad));
    }
}

impl<XS, YS, XF, YF> MatMul<Tensor<YS, YF>> for Tensor<XS, XF>
where
    XS: Clone + MatMul<YS>,
    YS: Clone,
{
    type Output = Tensor<<XS as MatMul<YS>>::Output, MatMulBackwardTT<XS, YS, XF, YF>>;
    fn matmul(self, rhs: Tensor<YS, YF>) -> Self::Output {
        Tensor {
            data: self.data.clone().matmul(rhs.data.clone()),
            grad_fn: MatMulBackwardTT {
                xgrad_fn: self.grad_fn,
                xdata: self.data,
                ygrad_fn: rhs.grad_fn,
                ydata: rhs.data,
            }
        }
    }
}
