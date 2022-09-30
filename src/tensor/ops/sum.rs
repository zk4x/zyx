use crate::{ops::{Sum, Expand, GetShape}, tensor::{Tensor, TensorGrad, TensorFunc, Backward}};
use std::{rc::Rc, ops::Add, cell::RefCell};

impl<S> Sum for Tensor<S>
where
    for<'a> &'a S: Sum<Output = S>,
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

#[derive(Debug)]
pub struct SumBackwardG<'g, S> {
    grad: &'g RefCell<S>,
    shape: Vec<usize>,
}

impl<'g, S> Backward<S> for SumBackwardG<'g, S>
where
    for<'a> &'a S: Add<Output = S> + Expand<Output = S> + GetShape,
{
    fn backward(self, res_grad: S) {
        self.grad.replace_with(|grad| &*grad + &res_grad.expand(&self.shape));
    }
}

impl<'g, S> Sum for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: Sum<Output = S> + GetShape,
{
    type Output = TensorFunc<S, SumBackwardG<'g, S>>;
    fn sum(self, dims: &[i32]) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.borrow().sum(dims)),
            func: SumBackwardG {
                grad: &self.grad,
                shape: self.data.borrow().shape(),
            }
        }
    }
}

#[derive(Debug)]
pub struct SumBackwardF<F> {
    func: F,
    shape: Vec<usize>,
}

impl<S, F> Backward<S> for SumBackwardF<F>
where
    for<'a> &'a S: Add<Output = S> + Expand<Output = S> + GetShape,
    F: Backward<S>,
{
    fn backward(self, res_grad: S) {
        self.func.backward(res_grad.expand(&self.shape));
    }
}

impl<S, F> Sum for TensorFunc<S, F>
where
    for<'a> &'a S: Sum<Output = S> + GetShape,
{
    type Output = TensorFunc<S, SumBackwardF<F>>;
    fn sum(self, dims: &[i32]) -> Self::Output {
        TensorFunc {
            data: Rc::new(self.data.sum(dims)),
            func: SumBackwardF {
                func: self.func,
                shape: self.data.shape(),
            }
        }
    }
}
