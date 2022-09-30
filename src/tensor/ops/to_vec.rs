use crate::{ops::ToVec, tensor::{Tensor, TensorGrad, TensorFunc}};

impl<S, T> ToVec<T> for Tensor<S>
where
    S: ToVec<T>,
{
    fn to_vec(&self) -> Vec<T> {
        self.data.to_vec()
    }
}

impl<S, T> ToVec<T> for TensorGrad<S>
where
    S: ToVec<T>,
{
    fn to_vec(&self) -> Vec<T> {
        self.data.borrow().to_vec()
    }
}

impl<S, F, T> ToVec<T> for TensorFunc<S, F>
where
    S: ToVec<T>,
{
    fn to_vec(&self) -> Vec<T> {
        self.data.to_vec()
    }
}