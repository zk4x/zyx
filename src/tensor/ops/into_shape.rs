use crate::{ops::IntoShape, tensor::{Variable, Tensor}};

impl<S> IntoShape for Variable<S>
where
    S: IntoShape,
{
    fn shape(&self) -> Vec<usize> {
        self.data().shape()
    }
}

impl<S, F> IntoShape for Tensor<S, F>
where
    S: IntoShape,
{
    fn shape(&self) -> Vec<usize> {
        self.data().shape()
    }
}