use crate::{ops::{MatMul, Transpose}, tensor::{Tensor, TensorGrad, TensorFunc}};
use std::{rc::Rc, ops::Add};

impl<S> MatMul<Tensor<S>> for Tensor<S>
where
    for<'a> &'a S: MatMul<Output = S>,
{
    type Output = Tensor<S>;
    fn matmul(self, rhs: Tensor<S>) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.matmul(&rhs.data)),
        }
    }
}

impl<'g, S> MatMul<&'g TensorGrad<S>> for Tensor<S>
where
    S: 'g,
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn matmul(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        let self_data = self.data;
        let rhs_grad = &rhs.grad;
        TensorFunc {
            data: Rc::new(self_data.matmul(&rhs.data())),
            func: move |res_grad: S| {
                rhs_grad.replace_with(|grad| &*grad + &self_data.transpose().matmul(&res_grad));
            },
        }
    }
}

impl<S, F> MatMul<TensorFunc<S, F>> for Tensor<S>
where
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn matmul(self, rhs: TensorFunc<S, F>) -> Self::Output {
        let self_data = self.data;
        let rhs_func = rhs.func;
        TensorFunc {
            data: Rc::new(self_data.matmul(&rhs.data)),
            func: move |res_grad| rhs_func(self_data.transpose().matmul(&res_grad)),
        }
    }
}

impl<'g, S> MatMul<Tensor<S>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: MatMul<&'a S, Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn matmul(self, rhs: Tensor<S>) -> Self::Output {
        let rhs_data = rhs.data;
        let self_grad = &self.grad;
        TensorFunc {
            data: Rc::new(self.data.borrow().matmul(&rhs_data)),
            func: move |res_grad: S| {
                self_grad.replace_with(|grad| &*grad + &res_grad.matmul(&rhs_data.transpose()));
            },
        }
    }
}

impl<'g, S> MatMul<&'g TensorGrad<S>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn matmul(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        let self_data = self.data.borrow();
        let rhs_data = rhs.data.borrow();
        let self_grad = &self.grad;
        let rhs_grad = &rhs.grad;
        TensorFunc {
            data: Rc::new(self_data.matmul(&rhs.data.borrow())),
            func: move |res_grad: S| {
                rhs_grad.replace_with(|grad| &*grad + &self_data.transpose().matmul(&res_grad));
                self_grad.replace_with(|grad| &*grad + &res_grad.matmul(&rhs_data.transpose()));
            },
        }
    }
}

impl<'g, S, F> MatMul<TensorFunc<S, F>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn matmul(self, rhs: TensorFunc<S, F>) -> Self::Output {
        let self_data = self.data.borrow();
        let rhs_data = rhs.data;
        let self_grad = &self.grad;
        let rhs_func = rhs.func;
        TensorFunc {
            data: Rc::new(self_data.matmul(&rhs_data)),
            func: move |res_grad| {
                rhs_func(self_data.transpose().matmul(&res_grad));
                self_grad.replace_with(|grad| &*grad + &res_grad.transpose().matmul(&rhs_data));
            },
        }
    }
}

impl<S, F> MatMul<Tensor<S>> for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn matmul(self, rhs: Tensor<S>) -> Self::Output {
        let rhs_data = rhs.data;
        let self_func = self.func;
        TensorFunc {
            data: Rc::new(self.data.matmul(&rhs_data)),
            func: move |res_grad: S| {
                self_func(rhs_data.transpose().matmul(&res_grad));
            },
        }
    }
}

impl<'g, S, F> MatMul<&'g TensorGrad<S>> for TensorFunc<S, F>
where
    S: 'g,
    F: FnOnce(S),
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn matmul(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        let self_data = self.data;
        let rhs_data = rhs.data.borrow();
        let self_func = self.func;
        let rhs_grad = &rhs.grad;
        TensorFunc {
            data: Rc::new(self_data.matmul(&rhs.data.borrow())),
            func: move |res_grad: S| {
                self_func(res_grad.matmul(&rhs_data.transpose()));
                rhs_grad.replace_with(|grad| &*grad + &self_data.transpose().matmul(&res_grad));
            },
        }
    }
}

impl<S, F2, F> MatMul<TensorFunc<S, F2>> for TensorFunc<S, F>
where
    F: FnOnce(S),
    F2: FnOnce(S),
    for<'a> &'a S: MatMul<Output = S> + Transpose<Output = S> + Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn matmul(self, rhs: TensorFunc<S, F2>) -> Self::Output {
        let self_data = self.data;
        let rhs_data = rhs.data;
        let self_func = self.func;
        let rhs_func = rhs.func;
        TensorFunc {
            data: Rc::new(self_data.matmul(&rhs_data)),
            func: move |res_grad: S| {
                self_func(res_grad.matmul(&rhs_data.transpose()));
                rhs_func(self_data.transpose().matmul(&res_grad));
            },
        }
    }
}