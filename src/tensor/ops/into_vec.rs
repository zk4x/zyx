use crate::{ops::IntoVec, tensor::{Variable, Tensor, Gradient}};
extern crate alloc;
use alloc::vec::Vec;

impl<G, T> crate::ops::IntoVec<T> for Gradient<G>
where
    G: crate::ops::IntoVec<T>,
{
    fn to_vec(&self) -> Vec<T> {
        // This is save, beacause it is read only access
        if let Some(grad) = unsafe { &*self.0.get() } {
            grad.to_vec()
        } else {
            Vec::new()
        }
    }
}

impl<S, T> IntoVec<T> for Variable<S>
where
    S: Clone + IntoVec<T>,
{
    fn to_vec(&self) -> Vec<T> {
        self.data.clone().to_vec()
    }
}

impl<S, F, T> IntoVec<T> for Tensor<S, F>
where
    S: IntoVec<T>,
{
    fn to_vec(&self) -> Vec<T> {
        self.data.to_vec()
    }
}