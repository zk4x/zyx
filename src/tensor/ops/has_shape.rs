use crate::{
    ops::HasShape,
    tensor::{Tensor, Variable},
};

impl<B> HasShape for Variable<B>
where
    B: HasShape,
{
    type S = <B as HasShape>::S;
}

impl<B, F> HasShape for Tensor<B, F>
where
    B: HasShape,
{
    type S = <B as HasShape>::S;
}
