use crate::{ops::ToVec, tensor::{Variable, Tensor}};

impl<S, T> ToVec<T> for Variable<S>
where
    S: ToVec<T>,
{
    fn to_vec(&self) -> Vec<T> {
        self.data.borrow().to_vec()
    }
}

impl<S, F, T> ToVec<T> for Tensor<S, F>
where
    S: ToVec<T>,
{
    fn to_vec(&self) -> Vec<T> {
        self.data.to_vec()
    }
}