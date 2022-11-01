use crate::{ops::Zeros, tensor::{Variable, Gradient}, shape::IntoShape};
use std::cell::RefCell;

/// Initialize tensor filled with zeros
impl<S, G> Zeros for Variable<S, G>
where
    S: Zeros,
{
    fn zeros(shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        Self {
            data: RefCell::new(S::zeros(shape)),
            grad: Gradient::new(),
        }
    }
}