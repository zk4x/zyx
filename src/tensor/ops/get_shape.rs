use crate::{ops::GetShape, tensor::{Tensor, TensorGrad, TensorFunc}};

impl<S> GetShape for &Tensor<S>
where
    for<'a> &'a S: GetShape,
{
    fn shape(self) -> Vec<usize> {
        self.data().shape()
    }
}

impl<S> GetShape for &TensorGrad<S>
where
    for<'a> &'a S: GetShape,
{
    fn shape(self) -> Vec<usize> {
        self.data().shape()
    }
}

impl<S, F> GetShape for &TensorFunc<S, F>
where
    for<'a> &'a S: GetShape,
{
    fn shape(self) -> Vec<usize> {
        self.data().shape()
    }
}