use crate::{
    ops::{self, Transpose, GetShape},
    tensor::{Tensor, TensorFunc, TensorGrad},
};
use std::{rc::Rc};

impl<S, T> ops::ToVec<T> for Tensor<S>
where
    S: ops::ToVec<T>,
{
    fn to_vec(&self) -> Vec<T> {
        self.data.to_vec()
    }
}

impl<S, T> ops::ToVec<T> for TensorGrad<S>
where
    S: ops::ToVec<T>,
{
    fn to_vec(&self) -> Vec<T> {
        self.data.borrow().to_vec()
    }
}

impl<S, F, T> ops::ToVec<T> for TensorFunc<S, F>
where
    S: ops::ToVec<T>,
{
    fn to_vec(&self) -> Vec<T> {
        self.data.to_vec()
    }
}

impl<S> ops::GetShape for &Tensor<S>
where
    for<'a> &'a S: GetShape,
{
    fn shape(self) -> Vec<usize> {
        self.data().shape()
    }
}

impl<S> ops::GetShape for &TensorGrad<S>
where
    for<'a> &'a S: GetShape,
{
    fn shape(self) -> Vec<usize> {
        self.data().shape()
    }
}

impl<S, F> ops::GetShape for &TensorFunc<S, F>
where
    for<'a> &'a S: GetShape,
{
    fn shape(self) -> Vec<usize> {
        self.data().shape()
    }
}

impl<S> ops::ReLU for Tensor<S>
where
    for<'a> &'a S: ops::ReLU<Output = S>,
{
    type Output = Tensor<S>;
    fn relu(self) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.relu()),
        }
    }
}

impl<'g, S> ops::ReLU for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: ops::ReLU<Output = S>
        + ops::DReLU<Output = S>
        + std::ops::Mul<Output = S>
        + std::ops::Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn relu(self) -> Self::Output {
        use ops::DReLU;
        let self_grad = &self.grad;
        let self_data = self.data();
        TensorFunc {
            data: Rc::new(self_data.relu()),
            func: move |res_grad: S| { self_grad.replace_with(|grad| &*grad + &(&res_grad * &self_data.drelu())); },
        }
    }
}

impl<S, F> ops::ReLU for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: ops::ReLU<Output = S> + ops::DReLU<Output = S> + std::ops::Mul<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn relu(self) -> Self::Output {
        use ops::DReLU;
        let self_func = self.func;
        let self_data = self.data.clone();
        TensorFunc {
            data: Rc::new(self.data.relu()),
            func: move |res_grad| self_func(&res_grad * &self_data.drelu()),
        }
    }
}

impl<S> ops::Exp for Tensor<S>
where
    for<'a> &'a S: ops::Exp<Output = S>,
{
    type Output = Tensor<S>;
    fn exp(self) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.exp()),
        }
    }
}

impl<'g, S> ops::Exp for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: ops::Exp<Output = S> + std::ops::Mul<Output = S> + std::ops::Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn exp(self) -> Self::Output {
        let self_grad = &self.grad;
        let data = Rc::new(self.data().exp());
        TensorFunc {
            data: data.clone(),
            func: move |res_grad| { self_grad.replace_with(|grad| &*grad + &(&res_grad * &data)); },
        }
    }
}

impl<S, F> ops::Exp for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: ops::Exp<Output = S> + std::ops::Mul<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn exp(self) -> Self::Output {
        let self_func = self.func;
        let data = Rc::new(self.data.exp());
        TensorFunc {
            data: data.clone(),
            func: move |res_grad| self_func(&res_grad * &data),
        }
    }
}

impl<S> ops::Ln for Tensor<S>
where
    for<'a> &'a S: ops::Ln<Output = S>,
{
    type Output = Tensor<S>;
    fn ln(self) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.ln()),
        }
    }
}

impl<'g, S> ops::Ln for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: ops::Ln<Output = S>
        + std::ops::Mul<Output = S>
        + std::ops::Add<Output = S>
        + ops::Pow<Output = S>
        + std::ops::Neg<Output = S>,
    S: ops::Ones,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn ln(self) -> Self::Output {
        let self_grad = &self.grad;
        let self_data = self.data();
        use crate::ops::Pow;
        TensorFunc {
            data: Rc::new(self_data.ln()),
            func: move |res_grad| { self_grad.replace_with(|grad| &*grad + &(&res_grad * &self_data.pow(&-&S::ones(&[1])))); },
        }
    }
}

impl<S, F> ops::Ln for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: ops::Ln<Output = S> + std::ops::Mul<Output = S> + ops::Pow<Output = S> + std::ops::Neg<Output = S>,
    S: ops::Ones,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn ln(self) -> Self::Output {
        let self_func = self.func;
        let self_data = self.data;
        use crate::ops::Pow;
        TensorFunc {
            data: Rc::new(self_data.ln()),
            func: move |res_grad| self_func(&res_grad * &self_data.pow(&-&S::ones(&[1]))),
        }
    }
}

impl<S> ops::Tanh for Tensor<S>
where
    for<'a> &'a S: ops::Tanh<Output = S>,
{
    type Output = Tensor<S>;
    fn tanh(self) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.tanh()),
        }
    }
}

impl<'g, S> ops::Tanh for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: ops::Tanh<Output = S>
        + std::ops::Mul<Output = S>
        + std::ops::Add<Output = S>
        + ops::Pow<Output = S>
        + std::ops::Neg<Output = S>,
    S: ops::Ones,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn tanh(self) -> Self::Output {
        let self_grad = &self.grad;
        let data = Rc::new(self.data().tanh());
        use crate::ops::Pow;
        TensorFunc {
            data: data.clone(),
            func: move |res_grad| {
                self_grad.replace_with(|grad| &*grad + &(&res_grad * &(&-&data.pow(&(&S::ones(&[1]) + &S::ones(&[1]))) + &S::ones(&[1]))));
            },
        }
    }
}

impl<S, F> ops::Tanh for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: ops::Tanh<Output = S>
        + std::ops::Mul<Output = S>
        + std::ops::Add<Output = S>
        + ops::Pow<Output = S>
        + std::ops::Neg<Output = S>,
    S: ops::Ones,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn tanh(self) -> Self::Output {
        let self_func = self.func;
        let data = Rc::new(self.data.tanh());
        use crate::ops::Pow;
        TensorFunc {
            data: data.clone(),
            func: move |res_grad| self_func(&res_grad * &(&-&data.pow(&(&S::ones(&[1]) + &S::ones(&[1]))) + &S::ones(&[1]))),
        }
    }
}

impl<S> std::ops::Neg for Tensor<S>
where
    for<'a> &'a S: std::ops::Neg<Output = S>,
{
    type Output = Tensor<S>;
    fn neg(self) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.neg()),
        }
    }
}

impl<'g, S> std::ops::Neg for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: std::ops::Mul<Output = S> + std::ops::Add<Output = S>,
    for<'b> &'b S: std::ops::Neg<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn neg(self) -> Self::Output {
        let self_grad = &self.grad;
        TensorFunc {
            data: Rc::new(self.data().neg()),
            func: move |res_grad: S| { self_grad.replace_with(|grad| &*grad + &(-&res_grad)); },
        }
    }
}

impl<S, F> std::ops::Neg for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: std::ops::Neg<Output = S> + std::ops::Mul<Output = S> + ops::Pow<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn neg(self) -> Self::Output {
        let self_func = self.func;
        TensorFunc {
            data: Rc::new(self.data.neg()),
            func: move |res_grad| self_func(-&res_grad),
        }
    }
}

impl<S> ops::Sum for Tensor<S>
where
    for<'a> &'a S: ops::Sum<Output = S>,
{
    type Output = Tensor<S>;
    fn sum(self, dims: &[i32]) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.sum(dims)),
        }
    }
}

// Why is compiler unable to distinguish between this function and function above?
/*impl<S> Tensor<S> {
    pub fn sum<const N: usize>(&self) -> Tensor<S>
    where
        S: GetShape<N>, // this is needed so that compiler can infer N
        for<'a> &'a S: ops::Sum<[i32; N], N, Output = S>,
    {
        use ops::Sum;
        Tensor {
            data: Rc::new(self.data.sum(std::array::from_fn(|i| i as i32))),
        }
    }
}*/

impl<'g, S> ops::Sum for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: ops::Sum<Output = S> + std::ops::Add<Output = S> + ops::Expand<Output = S> + ops::GetShape,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sum(self, dims: &[i32]) -> Self::Output {
        let self_grad = &self.grad;
        let self_shape = self.data.borrow().shape();
        use ops::Expand;
        TensorFunc {
            data: Rc::new(self.data.borrow().sum(dims)),
            func: move |res_grad: S| { self_grad.replace_with(|grad| &*grad + &res_grad.expand(&self_shape)); },
        }
    }
}

impl<S, F> ops::Sum for TensorFunc<S, F>
where
    for<'a> &'a S: ops::Sum<Output = S> + std::ops::Add<Output = S> + ops::Expand<Output = S> + GetShape,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sum(self, dims: &[i32]) -> Self::Output {
        let self_func = self.func;
        let self_shape = self.data.shape();
        use ops::Expand;
        TensorFunc {
            data: Rc::new(self.data.sum(dims)),
            func: move |res_grad: S| self_func(res_grad.expand(&self_shape)),
        }
    }
}

impl<S> ops::Max for Tensor<S>
where
    for<'a> &'a S: ops::Max<Output = S>,
{
    type Output = Tensor<S>;
    fn max(self, dims: &[i32]) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.max(dims)),
        }
    }
}

impl<'g, S> ops::Max for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: ops::Max<Output = S> + std::ops::Add<Output = S> + ops::Expand<Output = S> + GetShape,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn max(self, dims: &[i32]) -> Self::Output {
        let self_grad = &self.grad;
        let self_shape = self.data.borrow().shape();
        use ops::Expand;
        TensorFunc {
            data: Rc::new(self.data.borrow().max(dims)),
            func: move |res_grad: S| { self_grad.replace_with(|grad| &*grad + &res_grad.expand(&self_shape)); },
        }
    }
}

impl<S, F> ops::Max for TensorFunc<S, F>
where
    for<'a> &'a S: ops::Max<Output = S> + std::ops::Add<Output = S> + ops::Expand<Output = S> + GetShape,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn max(self, dims: &[i32]) -> Self::Output {
        let self_func = self.func;
        let self_shape = self.data.shape();
        use ops::Expand;
        TensorFunc {
            data: Rc::new(self.data.max(dims)),
            func: move |res_grad: S| self_func(res_grad.expand(&self_shape)),
        }
    }
}

impl<S> ops::Min for Tensor<S>
where
    for<'a> &'a S: ops::Min<Output = S>,
{
    type Output = Tensor<S>;
    fn min(self, dims: &[i32]) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.min(dims)),
        }
    }
}

impl<'g, S> ops::Min for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: ops::Min<Output = S> + std::ops::Add<Output = S> + ops::Expand<Output = S> + GetShape,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn min(self, dims: &[i32]) -> Self::Output {
        let self_grad = &self.grad;
        let self_shape = self.data.borrow().shape();
        use ops::Expand;
        TensorFunc {
            data: Rc::new(self.data.borrow().min(dims)),
            func: move |res_grad: S| { self_grad.replace_with(|grad| &*grad + &res_grad.expand(&self_shape)); },
        }
    }
}

impl<S, F> ops::Min for TensorFunc<S, F>
where
    for<'a> &'a S: ops::Min<Output = S> + std::ops::Add<Output = S> + ops::Expand<Output = S> + GetShape,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn min(self, dims: &[i32]) -> Self::Output {
        let self_func = self.func;
        let self_shape = self.data.shape();
        use ops::Expand;
        TensorFunc {
            data: Rc::new(self.data.min(dims)),
            func: move |res_grad: S| self_func(res_grad.expand(&self_shape)),
        }
    }
}

impl<S> ops::Reshape for Tensor<S>
where
    for<'a> &'a S: ops::Reshape<Output = S>,
{
    type Output = Tensor<S>;
    fn reshape(self, shape: &[usize]) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.reshape(shape)),
        }
    }
}

impl<'g, S> ops::Reshape for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: ops::Reshape<Output = S> + std::ops::Add<Output = S> + GetShape,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn reshape(self, shape: &[usize]) -> Self::Output {
        let self_grad = &self.grad;
        let self_shape = self.data.borrow().shape();
        TensorFunc {
            data: Rc::new(self.data().reshape(shape)),
            func: move |res_grad: S| { self_grad.replace_with(|grad| &*grad + &res_grad.reshape(&self_shape)); },
        }
    }
}

impl<S, F> ops::Reshape for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: ops::Reshape<Output = S> + GetShape,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn reshape(self, shape: &[usize]) -> Self::Output {
        let self_func = self.func;
        let self_shape = self.data.shape();
        TensorFunc {
            data: Rc::new(self.data.reshape(shape)),
            func: move |res_grad: S| self_func(res_grad.reshape(&self_shape)),
        }
    }
}

impl<S> ops::Expand for Tensor<S>
where
    for<'a> &'a S: ops::Expand<Output = S>,
{
    type Output = Tensor<S>;
    fn expand(self, shape: &[usize]) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.expand(shape)),
        }
    }
}

impl<'g, S> ops::Expand for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: ops::Expand<Output = S> + std::ops::Add<Output = S> + ops::Max<Output = S> + GetShape,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn expand(self, shape: &[usize]) -> Self::Output {
        // TODO: is max correct reduce for expand backward?
        let self_grad = &self.grad;
        let dims: Vec<i32> = self.data.borrow().shape().iter().zip(shape.iter()).enumerate().filter_map(|(i, (a, b))| if a != b { Some(i as i32) } else { None }).collect();
        use ops::Max;
        TensorFunc {
            data: Rc::new(self.data().expand(shape)),
            func: move |res_grad: S| { self_grad.replace_with(|grad| &*grad + &res_grad.max(&dims)); },
        }
    }
}

impl<S, F> ops::Expand for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: ops::Expand<Output = S> + ops::Max<Output = S> + GetShape,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn expand(self, shape: &[usize]) -> Self::Output {
        let self_func = self.func;
        let dims: Vec<i32> = self.data.shape().iter().zip(shape.iter()).enumerate().filter_map(|(i, (a, b))| if a != b { Some(i as i32) } else { None }).collect();
        use ops::Max;
        TensorFunc {
            data: Rc::new(self.data.expand(shape)),
            func: move |res_grad: S| self_func(res_grad.max(&dims)),
        }
    }
}

impl<S> ops::Permute for Tensor<S>
where
    for<'a> &'a S: ops::Permute<Output = S>,
{
    type Output = Tensor<S>;
    fn permute(self, dims: &[i32]) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.permute(dims)),
        }
    }
}

impl<'g, S> ops::Permute for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: ops::Permute<Output = S> + std::ops::Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn permute(self, dims: &[i32]) -> Self::Output {
        let self_grad = &self.grad;
        use crate::shape::Dims;
        let inv_dims = dims.argsort();
        TensorFunc {
            data: Rc::new(self.data().permute(dims)),
            func: move |res_grad: S| { self_grad.replace_with(|grad| &*grad + &res_grad.permute(&inv_dims)); },
        }
    }
}

impl<S, F> ops::Permute for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: ops::Permute<Output = S> + ops::Permute<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn permute(self, dims: &[i32]) -> Self::Output {
        let self_func = self.func;
        use crate::shape::Dims;
        let inv_dims = dims.argsort();
        TensorFunc {
            data: Rc::new(self.data.permute(dims)),
            func: move |res_grad: S| {
                self_func(res_grad.permute(&inv_dims));
            },
        }
    }
}

impl<S> std::ops::Add<Tensor<S>> for Tensor<S>
where
    for<'a> &'a S: std::ops::Add<Output = S>,
{
    type Output = Tensor<S>;
    fn add(self, rhs: Tensor<S>) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.as_ref() + rhs.data.as_ref()),
        }
    }
}

impl<'g, S> std::ops::Add<&'g TensorGrad<S>> for Tensor<S>
where
    S: 'g,
    for<'a> &'a S: std::ops::Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn add(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        let rhs_grad = &rhs.grad;
        TensorFunc {
            data: Rc::new(self.data.as_ref() + rhs.data().as_ref()),
            func: move |res_grad| { rhs_grad.replace_with(|grad| &*grad + &res_grad);
            },
        }
    }
}

impl<S, F> std::ops::Add<TensorFunc<S, F>> for Tensor<S>
where
    F: FnOnce(S),
    for<'a> &'a S: std::ops::Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn add(self, rhs: TensorFunc<S, F>) -> Self::Output {
        let rhs_func = rhs.func;
        TensorFunc {
            data: Rc::new(self.data.as_ref() + rhs.data.as_ref()),
            func: move |res_grad| {
                rhs_func(res_grad);
            },
        }
    }
}

impl<'g, S> std::ops::Add<Tensor<S>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: std::ops::Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn add(self, rhs: Tensor<S>) -> Self::Output {
        let self_grad = &self.grad;
        TensorFunc {
            data: Rc::new(self.data().as_ref() + rhs.data.as_ref()),
            func: move |res_grad| { self_grad.replace_with(|grad| &*grad + &res_grad); },
        }
    }
}

impl<'g, S> std::ops::Add<&'g TensorGrad<S>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: std::ops::Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn add(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        let self_grad = &self.grad;
        let rhs_grad = &rhs.grad;
        TensorFunc {
            data: Rc::new(self.data().as_ref() + rhs.data().as_ref()),
            func: move |res_grad| {
                self_grad.replace_with(|grad| &*grad + &res_grad);
                rhs_grad.replace_with(|grad| &*grad + &res_grad);
            },
        }
    }
}

impl<'g, S, F> std::ops::Add<TensorFunc<S, F>> for &'g TensorGrad<S>
where
    S: 'g,
    F: FnOnce(S),
    for<'a> &'a S: std::ops::Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn add(self, rhs: TensorFunc<S, F>) -> Self::Output {
        let self_grad = &self.grad;
        let rhs_func = rhs.func;
        TensorFunc {
            data: Rc::new(self.data().as_ref() + rhs.data.as_ref()),
            func: move |res_grad| {
                self_grad.replace_with(|grad| &*grad + &res_grad);
                rhs_func(res_grad);
            },
        }
    }
}

impl<S, F> std::ops::Add<Tensor<S>> for TensorFunc<S, F>
where
    for<'a> &'a S: std::ops::Add<Output = S>,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn add(self, rhs: Tensor<S>) -> Self::Output {
        let self_func = self.func;
        TensorFunc {
            data: Rc::new(self.data.as_ref() + rhs.data.as_ref()),
            func: self_func,
        }
    }
}

impl<'g, S, F> std::ops::Add<&'g TensorGrad<S>> for TensorFunc<S, F>
where
    S: 'g,
    for<'a> &'a S: std::ops::Add<Output = S>,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn add(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        let self_func = self.func;
        let rhs_grad = &rhs.grad;
        TensorFunc {
            data: Rc::new(self.data.as_ref() + rhs.data().as_ref()),
            func: move |res_grad| {
                rhs_grad.replace_with(|grad| &*grad + &res_grad);
                self_func(res_grad);
            },
        }
    }
}

impl<S, F1, F2> std::ops::Add<TensorFunc<S, F2>> for TensorFunc<S, F1>
where
    S: Clone,
    F1: FnOnce(S),
    F2: FnOnce(S),
    for<'a> &'a S: std::ops::Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn add(self, rhs: TensorFunc<S, F2>) -> Self::Output {
        let self_func = self.func;
        let rhs_func = rhs.func;
        TensorFunc {
            data: Rc::new(self.data.as_ref() + rhs.data.as_ref()),
            func: move |res_grad: S| {
                // With impl FnOnce(Rc<S>) this is not a copy, but hey, advantages from passing S
                // by value and potentially doing operations in place are bigger than this copy
                // (at least hope so)
                self_func(res_grad.clone());
                rhs_func(res_grad);
            },
        }
    }
}

impl<S> std::ops::Sub<Tensor<S>> for Tensor<S>
where
    for<'a> &'a S: std::ops::Sub<Output = S>,
{
    type Output = Tensor<S>;
    fn sub(self, rhs: Tensor<S>) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.as_ref() - rhs.data.as_ref()),
        }
    }
}

impl<'g, S> std::ops::Sub<&'g TensorGrad<S>> for Tensor<S>
where
    S: 'g,
    for<'a> &'a S: std::ops::Sub<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sub(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        let rhs_grad = &rhs.grad;
        TensorFunc {
            data: Rc::new(self.data.as_ref() - rhs.data().as_ref()),
            func: move |res_grad| {
                rhs_grad.replace_with(|grad| &*grad - &res_grad);
            },
        }
    }
}

impl<S, F> std::ops::Sub<TensorFunc<S, F>> for Tensor<S>
where
    F: FnOnce(S),
    for<'a> &'a S: std::ops::Sub<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sub(self, rhs: TensorFunc<S, F>) -> Self::Output {
        let rhs_func = rhs.func;
        TensorFunc {
            data: Rc::new(self.data.as_ref() - rhs.data.as_ref()),
            func: move |res_grad| {
                rhs_func(res_grad);
            },
        }
    }
}

impl<'g, S> std::ops::Sub<Tensor<S>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: std::ops::Sub<Output = S> + std::ops::Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sub(self, rhs: Tensor<S>) -> Self::Output {
        let self_grad = &self.grad;
        TensorFunc {
            data: Rc::new(self.data().as_ref() - rhs.data.as_ref()),
            func: move |res_grad| { self_grad.replace_with(|grad| &*grad + &res_grad); },
        }
    }
}

impl<'g, S> std::ops::Sub<&'g TensorGrad<S>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: std::ops::Sub<Output = S> + std::ops::Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sub(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        let self_grad = &self.grad;
        let rhs_grad = &rhs.grad;
        TensorFunc {
            data: Rc::new(self.data().as_ref() - rhs.data().as_ref()),
            func: move |res_grad| {
                self_grad.replace_with(|grad| &*grad + &res_grad);
                rhs_grad.replace_with(|grad| &*grad - &res_grad);
            },
        }
    }
}

impl<'g, S, F> std::ops::Sub<TensorFunc<S, F>> for &'g TensorGrad<S>
where
    S: 'g,
    F: FnOnce(S),
    for<'a> &'a S: std::ops::Sub<Output = S> + std::ops::Add<Output = S> + std::ops::Neg<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sub(self, rhs: TensorFunc<S, F>) -> Self::Output {
        let self_grad = &self.grad;
        let rhs_func = rhs.func;
        TensorFunc {
            data: Rc::new(self.data().as_ref() - rhs.data.as_ref()),
            func: move |res_grad| {
                self_grad.replace_with(|grad| &*grad + &res_grad);
                rhs_func(-&res_grad);
            },
        }
    }
}

impl<S, F> std::ops::Sub<Tensor<S>> for TensorFunc<S, F>
where
    for<'a> &'a S: std::ops::Sub<Output = S>,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sub(self, rhs: Tensor<S>) -> Self::Output {
        let self_func = self.func;
        TensorFunc {
            data: Rc::new(self.data.as_ref() - rhs.data.as_ref()),
            func: move |res_grad| {
                self_func(res_grad);
            },
        }
    }
}

impl<'g, S, F> std::ops::Sub<&'g TensorGrad<S>> for TensorFunc<S, F>
where
    S: 'g,
    for<'a> &'a S: std::ops::Sub<Output = S>,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sub(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        let self_func = self.func;
        let rhs_grad = &rhs.grad;
        TensorFunc {
            data: Rc::new(self.data.as_ref() - rhs.data().as_ref()),
            func: move |res_grad| {
                rhs_grad.replace_with(|grad| &*grad - &res_grad);
                self_func(res_grad);
            },
        }
    }
}

impl<S, F1, F2> std::ops::Sub<TensorFunc<S, F2>> for TensorFunc<S, F1>
where
    F1: FnOnce(S),
    F2: FnOnce(S),
    for<'a> &'a S: std::ops::Sub<Output = S>,
    for<'b> &'b S: std::ops::Neg<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn sub(self, rhs: TensorFunc<S, F2>) -> Self::Output {
        let self_func = self.func;
        let rhs_func = rhs.func;
        TensorFunc {
            data: Rc::new(self.data.as_ref() - rhs.data.as_ref()),
            func: move |res_grad: S| {
                rhs_func(-&res_grad);
                self_func(res_grad);
            },
        }
    }
}

/*impl<S> ops::Pow for Tensor<S>
where
    for<'a> &'a S: ops::Pow<Output = S>,
{
    type Output = Tensor<S>;
    fn pow(self, exponent: i32) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.pow(exponent)),
        }
    }
}

impl<S> ops::Pow for TensorGrad<S>
where
    for<'a> &'a S: ops::Pow<Output = S>
        + std::ops::Mul<Output = S>
        + std::ops::Add<Output = S>
        + ops::Pow<Output = S>
        + std::ops::Mul<i32, Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn pow(self, exponent: i32) -> Self::Output {
        let self_grad = Rc::downgrade(&self.grad);
        let self_data = self.data();
        TensorFunc {
            data: Rc::new(self_data.pow(exponent)),
            func: move |res_grad: | {
                
                    self_grad.replace_with(|grad| {
                        Rc::new(
                            grad.as_ref()
                                + &(res_grad.as_ref() * &(&self_data.pow(exponent - 1) * exponent)),
                        )
                    });
                }
            }))),
        }
    }
}

impl<S, F> ops::Pow for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: ops::Pow<Output = S>
        + std::ops::Mul<Output = S>
        + ops::Pow<Output = S>
        + std::ops::Mul<i32, Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn pow(self, exponent: i32) -> Self::Output {
        let self_func = Rc::downgrade(&self.func);
        let self_data = self.data.clone();
        TensorFunc {
            data: Rc::new(self.data.pow(exponent)),
            func: move |res_grad: | {
                if let Some(func) = self_func
                    .upgrade()
                    .unwrap_or_else(|| Rc::new(Cell::new(None)))
                    .take()
                {
                    func(Rc::new(
                        res_grad.as_ref() * &(&self_data.pow(exponent - 1) * exponent),
                    ));
                }
            }))),
        }
    }
}*/

impl<S> ops::MatMul<Tensor<S>> for Tensor<S>
where
    for<'a> &'a S: ops::MatMul<&'a S, Output = S>,
{
    type Output = Tensor<S>;
    fn matmul(self, rhs: Tensor<S>) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.matmul(&rhs.data)),
        }
    }
}

impl<'g, S> ops::MatMul<&'g TensorGrad<S>> for Tensor<S>
where
    S: 'g,
    for<'a> &'a S: ops::MatMul<Output = S> + ops::Transpose<Output = S> + std::ops::Add<Output = S>,
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

impl<S, F> ops::MatMul<TensorFunc<S, F>> for Tensor<S>
where
    for<'a> &'a S: ops::MatMul<Output = S> + ops::Transpose<Output = S> + std::ops::Add<Output = S>,
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

impl<'g, S> ops::MatMul<Tensor<S>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: ops::MatMul<&'a S, Output = S> + ops::Transpose<Output = S> + std::ops::Add<Output = S>,
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

impl<'g, S> ops::MatMul<&'g TensorGrad<S>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: ops::MatMul<Output = S> + ops::Transpose<Output = S> + std::ops::Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn matmul(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        let self_data = self.data.borrow().clone();
        let rhs_data = rhs.data.borrow().clone();
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

impl<'g, S, F> ops::MatMul<TensorFunc<S, F>> for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: ops::MatMul<Output = S> + ops::Transpose<Output = S> + std::ops::Add<Output = S>,
    F: FnOnce(S),
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn matmul(self, rhs: TensorFunc<S, F>) -> Self::Output {
        let self_data = self.data.borrow().clone();
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

impl<S, F> ops::MatMul<Tensor<S>> for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: ops::MatMul<&'a S, Output = S> + ops::Transpose<Output = S> + std::ops::Add<Output = S>,
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

impl<'g, S, F> ops::MatMul<&'g TensorGrad<S>> for TensorFunc<S, F>
where
    S: 'g,
    F: FnOnce(S),
    for<'a> &'a S: ops::MatMul<Output = S> + ops::Transpose<Output = S> + std::ops::Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn matmul(self, rhs: &'g TensorGrad<S>) -> Self::Output {
        let self_data = self.data;
        let rhs_data = rhs.data.borrow().clone();
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

impl<S, F2, F> ops::MatMul<TensorFunc<S, F2>> for TensorFunc<S, F>
where
    F: FnOnce(S),
    F2: FnOnce(S),
    for<'a> &'a S: ops::MatMul<Output = S> + ops::Transpose<Output = S> + std::ops::Add<Output = S>,
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
