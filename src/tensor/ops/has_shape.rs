use crate::{
    ops::HasShape,
    tensor::{Tensor, Variable},
};

impl<S> HasShape for Variable<S>
where
    S: HasShape,
{
    type Sh = <S as HasShape>::Sh;
}

impl<S, F> HasShape for Tensor<S, F>
where
    S: HasShape,
{
    type Sh = <S as HasShape>::Sh;
}