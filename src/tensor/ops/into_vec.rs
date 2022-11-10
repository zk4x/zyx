use crate::{ops::IntoVec, tensor::{Variable, Tensor}};

impl<S, T> IntoVec<T> for Variable<S>
where
    S: Clone + IntoVec<T>,
{
    fn to_vec(&self) -> Vec<T> {
        self.data.clone().to_vec()
    }
}

impl<S, F, T> IntoVec<T> for Tensor<S, F>
where
    S: IntoVec<T>,
{
    fn to_vec(&self) -> Vec<T> {
        self.data.to_vec()
    }
}