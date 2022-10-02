use crate::{tensor::{Tensor, Backward}};
use std::ops::{Sub, Neg};

/*impl<'g, S> Sub<&'g Variable<S>> for Buffer<S>
where
    S: 'g,
    for<'a> &'a S: Sub<Output = S>,
{
    type Output = Tensor<S, impl FnOnce(S)>;
    fn sub(self, rhs: &'g Variable<S>) -> Self::Output {
        let rhs_grad = &rhs.grad;
        Tensor {
            data: self.data.as_ref() - rhs.data().as_ref()),
            func: move |res_grad| {
                rhs_grad.replace_take(|grad| &*grad - &res_grad);
            },
        }
    }
}

impl<S, F> Sub<Tensor<S, F>> for Buffer<S>
where
    F: FnOnce(S),
    for<'a> &'a S: Sub<Output = S>,
{
    type Output = Tensor<S, impl FnOnce(S)>;
    fn sub(self, rhs: Tensor<S, F>) -> Self::Output {
        let rhs_func = rhs.func;
        Tensor {
            data: self.data.as_ref() - rhs.data.as_ref()),
            func: move |res_grad| {
                rhs_func(res_grad);
            },
        }
    }
}

impl<'g, S> Sub<Buffer<S>> for &'g Variable<S>
where
    S: 'g,
    for<'a> &'a S: Sub<Output = S> + Add<Output = S>,
{
    type Output = Tensor<S, impl FnOnce(S)>;
    fn sub(self, rhs: Buffer<S>) -> Self::Output {
        let self_grad = &self.grad;
        Tensor {
            data: self.data().as_ref() - rhs.data.as_ref()),
            func: move |res_grad| { self_grad.replace_take(|grad| &*grad + &res_grad); },
        }
    }
}

impl<'g, S> Sub<&'g Variable<S>> for &'g Variable<S>
where
    S: 'g,
    for<'a> &'a S: Sub<Output = S> + Add<Output = S>,
{
    type Output = Tensor<S, impl FnOnce(S)>;
    fn sub(self, rhs: &'g Variable<S>) -> Self::Output {
        let self_grad = &self.grad;
        let rhs_grad = &rhs.grad;
        Tensor {
            data: self.data().as_ref() - rhs.data().as_ref()),
            func: move |res_grad| {
                self_grad.replace_take(|grad| &*grad + &res_grad);
                rhs_grad.replace_take(|grad| &*grad - &res_grad);
            },
        }
    }
}

impl<'g, S, F> Sub<Tensor<S, F>> for &'g Variable<S>
where
    S: 'g,
    F: FnOnce(S),
    for<'a> &'a S: Sub<Output = S> + Add<Output = S> + std::ops::Neg<Output = S>,
{
    type Output = Tensor<S, impl FnOnce(S)>;
    fn sub(self, rhs: Tensor<S, F>) -> Self::Output {
        let self_grad = &self.grad;
        let rhs_func = rhs.func;
        Tensor {
            data: self.data().as_ref() - rhs.data.as_ref()),
            func: move |res_grad| {
                self_grad.replace_take(|grad| &*grad + &res_grad);
                rhs_func(-&res_grad);
            },
        }
    }
}*/

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

/*impl<'g, S, F> Sub<&'g Variable<S>> for Tensor<S, F>
where
    S: 'g,
    for<'a> &'a S: Sub<Output = S>,
    F: FnOnce(S),
{
    type Output = Tensor<S, impl FnOnce(S)>;
    fn sub(self, rhs: &'g Variable<S>) -> Self::Output {
        let self_func = self.func;
        let rhs_grad = &rhs.grad;
        Tensor {
            data: self.data.as_ref() - rhs.data().as_ref()),
            func: move |res_grad| {
                rhs_grad.replace_take(|grad| &*grad - &res_grad);
                self_func(res_grad);
            },
        }
    }
}*/

#[derive(Debug, Clone, Copy)]
pub struct SubBackwardFF<XF, YF> {
    xfunc: XF,
    yfunc: YF,
}

impl<S, XF, YF> Backward<S> for SubBackwardFF<XF, YF>
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
    type Output = Tensor<S, SubBackwardFF<XF, YF>>;
    fn sub(self, rhs: Tensor<S, YF>) -> Self::Output {
        Tensor {
            data: self.data - rhs.data,
            func: SubBackwardFF {
                xfunc: self.func,
                yfunc: rhs.func,
            },
        }
    }
}
