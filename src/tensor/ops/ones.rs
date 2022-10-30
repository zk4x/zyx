use crate::{ops::{Ones, Zeros}, tensor::Variable, shape::IntoShape};
use std::cell::RefCell;

/// Initialize tensor filled with zeros
impl<S> Ones for Variable<S>
where
    S: Ones + Zeros,
{
    fn ones(shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        Self {
            data: RefCell::new(S::ones(shape.clone())),
            grad: RefCell::new(S::zeros(shape)),
        }
    }
}