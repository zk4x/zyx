use crate::{ops::{Ones, Zeros}, tensor::{Variable, Gradient}, shape::IntoShape};
use std::cell::RefCell;

/// Initialize tensor filled with zeros
impl<S, G> Ones for Variable<S, G>
where
    S: Ones + Zeros,
{
    fn ones(shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        Self {
            data: RefCell::new(S::ones(shape.clone())),
            grad: Gradient::new(),
        }
    }
}