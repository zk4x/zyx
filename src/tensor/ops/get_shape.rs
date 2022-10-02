use crate::{ops::GetShape, tensor::{Variable, Tensor}};

impl<S> GetShape for Variable<S>
where
    S: GetShape,
{
    fn shape(&self) -> Vec<usize> {
        self.data().shape()
    }
}

impl<S, F> GetShape for Tensor<S, F>
where
    S: GetShape,
{
    fn shape(&self) -> Vec<usize> {
        self.data().shape()
    }
}