use crate::{ops::HasDType, tensor::{Variable, Tensor}};

impl<S> HasDType for Variable<S>
where
    S: HasDType,
{
    type T = <S as HasDType>::T;
}

impl<S, F> HasDType for Tensor<S, F>
where
    S: HasDType,
{
    type T = <S as HasDType>::T;
}
