use crate::{
    ops::HasDevice,
    tensor::{Tensor, Variable},
};

impl<S> HasDevice for Variable<S>
where
    S: HasDevice,
{
    type Dev = <S as HasDevice>::Dev;
    fn device(&self) -> &Self::Dev {
        self.data.device()
    }
}

impl<S, F> HasDevice for Tensor<S, F>
where
    S: HasDevice,
{
    type Dev = <S as HasDevice>::Dev;
    fn device(&self) -> &Self::Dev {
        self.data.device()
    }
}
