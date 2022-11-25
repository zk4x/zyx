use crate::{ops::GetShape, tensor::{Variable, Tensor}};

impl<S> GetShape for Variable<S>
where
    S: GetShape,
{
    type Output = <S as GetShape>::Output;

    fn shape(&self) -> Self::Output {
        self.data().shape()
    }
}

impl<S, F> GetShape for Tensor<S, F>
where
    S: GetShape,
{
    type Output = <S as GetShape>::Output;

    fn shape(&self) -> Self::Output {
        self.data().shape()
    }
}
