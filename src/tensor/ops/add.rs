use crate::tensor::{B, Variable, Tensor, Backward, ops::RefCellReplaceTake};
use std::{cell::RefCell, ops::Add};

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardTG<'g, S> {
    ygrad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for AddBackwardTG<'g, S>
where
    S: Default + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad + res_grad);
    }
}

// If you wanted to add Variable or Tensor to S, you need to wrap it inside B(),
// but you can add S to Variable or Tensor
impl<'g, S> Add<&'g Variable<S>> for B<S>
where
    S: 'g + Clone + Add<Output = S>,
{
    type Output = Tensor<S, AddBackwardTG<'g, S>>;
    fn add(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.0 + rhs.data().clone(),
            func: AddBackwardTG {
                ygrad: &rhs.grad,
            }
        }
    }
}

impl<S, F> Add<Tensor<S, F>> for B<S>
where
    S: Add<Output = S>,
{
    type Output = Tensor<S, F>;
    fn add(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self.0 + rhs.data,
            func: rhs.func,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardGT<'g, S> {
    xgrad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for AddBackwardGT<'g, S>
where
    S: Default + Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad);
    }
}

impl<'g, S> Add<S> for &'g Variable<S>
where
    S: 'g + Clone + Add<Output = S>,
{
    type Output = Tensor<S, AddBackwardGT<'g, S>>;
    fn add(self, rhs: S) -> Self::Output {
        Tensor {
            data: self.data().clone() + rhs,
            func: AddBackwardGT {
                xgrad: &self.grad,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardGG<'g, S> {
    xgrad: &'g RefCell<S>,
    ygrad: &'g RefCell<S>,
}

impl<'g, S> Backward<S> for AddBackwardGG<'g, S>
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
    S: 'g + Clone + Add<Output = S>,
{
    type Output = Tensor<S, AddBackwardGG<'g, S>>;
    fn add(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.data().clone() + rhs.data().clone(),
            func: AddBackwardGG {
                xgrad: &self.grad,
                ygrad: &rhs.grad,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardGF<'g, S, YF> {
    xgrad: &'g RefCell<S>,
    yfunc: YF,
}

impl<'g, S, YF> Backward<S> for AddBackwardGF<'g, S, YF>
where
    S: Default + Clone + Add<Output = S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.xgrad.replace_take(|grad| grad + res_grad.clone());
        self.yfunc.backward(res_grad);
    }
}

impl<'g, S, F> Add<Tensor<S, F>> for &'g Variable<S>
where
    S: 'g + Clone + Add<Output = S>,
{
    type Output = Tensor<S, AddBackwardGF<'g, S, F>>;
    fn add(self, rhs: Tensor<S, F>) -> Self::Output {
        Tensor {
            data: self.data().clone() + rhs.data,
            func: AddBackwardGF {
                xgrad: &self.grad,
                yfunc: rhs.func,
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
            func: self.func,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardFG<'g, S, XF> {
    xfunc: XF,
    ygrad: &'g RefCell<S>,
}

impl<'g, S, XF> Backward<S> for AddBackwardFG<'g, S, XF>
where
    S: Default + Clone + Add<Output = S>,
    XF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.ygrad.replace_take(|grad| grad + res_grad.clone());
        self.xfunc.backward(res_grad);
    }
}

impl<'g, S, XF> Add<&'g Variable<S>> for Tensor<S, XF>
where
    S: 'g + Clone + Add<Output = S>,
{
    type Output = Tensor<S, AddBackwardFG<'g, S, XF>>;
    fn add(self, rhs: &'g Variable<S>) -> Self::Output {
        Tensor {
            data: self.data + (*rhs.data()).clone(),
            func: AddBackwardFG {
                xfunc: self.func,
                ygrad: &rhs.grad,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AddBackwardFF<XF, YF> {
    xfunc: XF,
    yfunc: YF,
}

impl<S, XF, YF> Backward<S> for AddBackwardFF<XF, YF>
where
    S: Clone + Add<Output = S>,
    XF: Backward<S>,
    YF: Backward<S>,
{
    fn backward(self, res_grad: S) {
        // If res_grad is &S this is not a copy, but hey, advantages from passing S
        // by value and potentially doing operations in place are bigger than this copy
        // (at least hope so), if not, this will be changed
        self.xfunc.backward(res_grad.clone());
        self.yfunc.backward(res_grad);
    }
}

impl<S, XF, YF> Add<Tensor<S, YF>> for Tensor<S, XF>
where
    S: Add<Output = S>,
{
    type Output = Tensor<S, AddBackwardFF<XF, YF>>;
    fn add(self, rhs: Tensor<S, YF>) -> Self::Output {
        Tensor {
            data: self.data + rhs.data,
            func: AddBackwardFF {
                xfunc: self.func,
                yfunc: rhs.func,
            }
        }
    }
}
