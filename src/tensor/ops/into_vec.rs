use crate::{
    ops::{HasDType, IntoVec},
    tensor::{Tensor, Variable},
};
extern crate alloc;
use alloc::vec::Vec;

/*impl<G, T> crate::ops::IntoVec<T> for Gradient<G>
where
    G: crate::ops::IntoVec<T>,
{
    fn to_vec(&self) -> Vec<T> {
        // This is safe, beacause it is read only access
        unsafe { &*self.0.get() }.to_vec()
    }
}*/

impl<S> IntoVec for Variable<S>
where
    S: IntoVec + HasDType,
{
    fn to_vec(&self) -> Vec<S::T> {
        self.data.to_vec()
    }
}

impl<S, F> IntoVec for Tensor<S, F>
where
    S: IntoVec + HasDType,
{
    fn to_vec(&self) -> Vec<S::T> {
        self.data.to_vec()
    }
}
