use crate::{
    ops::HasDevice,
    tensor::{Tensor, Variable},
};

impl<B> HasDevice for Variable<B>
where
    B: HasDevice,
{
    type D = <B as HasDevice>::D;
    fn device(&self) -> &Self::D {
        self.data.device()
    }
}

impl<B, F> HasDevice for Tensor<B, F>
where
    B: HasDevice,
{
    type D = <B as HasDevice>::D;
    fn device(&self) -> &Self::D {
        self.data.device()
    }
}
