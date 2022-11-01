use crate::{ops::{FromVec, Zeros}, tensor::{Variable, Gradient}, shape::IntoShape};
use std::cell::RefCell;

/// Initialize tensor from vec and shape
impl<S, G, T> FromVec<T> for Variable<S, G>
where
    S: FromVec<T> + Zeros,
{
    fn from_vec(data: Vec<T>, shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        debug_assert_eq!(data.len(), shape.numel());
        Self {
            data: RefCell::new(S::from_vec(data, shape)),
            grad: Gradient::new(),
        }
    }
}
