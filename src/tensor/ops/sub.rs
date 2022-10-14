use crate::tensor::{B, Tensor, Variable, Backward, ops::RefCellReplaceTake};
use std::{cell::RefCell, ops::{Sub, Neg, Add}};

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardSV<'g, S> {
    ygrad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for SubBackwardSV<'g, S>
where
    S: Default + Sub<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad - res_grad);
    }
}

impl<'g, S> Sub<&'g Variable<S>> for B<S>
where
    S: 'g + Clone + Sub<Output = S>,
{
    type Output = Tensor<S, SubBackwardSV<'g, S>>;
    fn sub(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.0 - rhs.data().clone(),
            func: SubBackwardSV {
                ygrad: &rhs.grad,
            },
        }
    }
}

impl<S, F> Sub<Tensor<S, F>> for B<S>
where
    S: Sub<Output = S>,
{
    type Output = Tensor<S, F>;
    fn sub(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self.0 - rhs.data,
            func: rhs.func,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardVS<'g, S> {
    xgrad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for SubBackwardVS<'g, S>
where
    S: Default + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad);
    }
}

impl<'g, S> Sub<S> for &'g Variable<S>
where
    S: 'g + Clone + Sub<Output = S>,
{
    type Output = Tensor<S, SubBackwardVS<'g, S>>;
    fn sub(self, rhs: S) -> Self::Output {
        Tensor {
            data: self.data().clone() - rhs,
            func: SubBackwardVS {
                xgrad: &self.grad,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardVV<'g, S> {
    xgrad: &'g RefCell<S>,
    ygrad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for SubBackwardVV<'g, S>
where
    S: Default + Clone + Add<Output = S> + Sub<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad.clone());
        self.ygrad.replace_take(|grad| grad - res_grad);
    }
}

impl<'g, S> Sub<&'g Variable<S>> for &'g Variable<S>
where
    S: 'g + Clone + Sub<Output = S>,
{
    type Output = Tensor<S, SubBackwardVV<'g, S>>;
    fn sub(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.data().clone() - rhs.data().clone(),
            func: SubBackwardVV {
                xgrad: &self.grad,
                ygrad: &rhs.grad,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardVT<'g, S, YF> {
    xgrad: &'g RefCell<S>,
    yfunc: YF,
}

impl<'g, S, YF> Backward<S> for SubBackwardVT<'g, S, YF>
where
    S: Default + Clone + Add<Output = S> + Neg<Output = S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad.clone());
        self.yfunc.backward(-res_grad);
    }
}

impl<'g, S, F> Sub<Tensor<S, F>> for &'g Variable<S>
where
    S: 'g + Clone + Sub<Output = S>,
{
    type Output = Tensor<S, SubBackwardVT<'g, S, F>>;
    fn sub(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self.data().clone() - rhs.data,
            func: SubBackwardVT {
                xgrad: &self.grad,
                yfunc: rhs.func,
            },
        }
    }
}

impl<S, XF> Sub<S> for Tensor<S, XF>
where
    S: Sub<Output = S>,
{
    type Output = Tensor<S, XF>;
    fn sub(self, rhs: S) -> Self::Output {
        Tensor {
            data: self.data - rhs,
            func: self.func,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardTV<'g, S, XF> {
    xfunc: XF,
    ygrad: &'g RefCell<S>,
}

impl<'g, S, XF> Backward<S> for SubBackwardTV<'g, S, XF>
where
    S: Default + Clone + Sub<Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xfunc.backward(res_grad.clone());
        self.ygrad.replace_take(|grad| grad - res_grad);
    }
}

impl<'g, S, F> Sub<&'g Variable<S>> for Tensor<S, F>
where
    S: 'g + Clone + Sub<Output = S>,
{
    type Output = Tensor<S, SubBackwardTV<'g, S, F>>;
    fn sub(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.data - rhs.data().clone(),
            func: SubBackwardTV {
                xfunc: self.func,
                ygrad: &rhs.grad,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardTT<XF, YF> {
    xfunc: XF,
    yfunc: YF,
}

impl<S, XF, YF> Backward<S> for SubBackwardTT<XF, YF>
where
    S: Clone + Neg<Output = S>,
    XF: Backward<S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xfunc.backward(-res_grad.clone());
        self.yfunc.backward(res_grad);
    }
}

impl<S, XF, YF> Sub<Tensor<S, YF>> for Tensor<S, XF>
where
    S: Sub<Output = S>,
{
    type Output = Tensor<S, SubBackwardTT<XF, YF>>;
    fn sub(self, rhs: Tensor<S, YF>) -> Self::Output {
        Tensor {
            data: self.data - rhs.data,
            func: SubBackwardTT {
                xfunc: self.func,
                yfunc: rhs.func,
            },
        }
    }
}
