use crate::{ops::Zeros, tensor::Variable, shape::IntoShape};
use std::cell::RefCell;

/// Initialize tensor filled with zeros
impl<S> Zeros for Variable<S>
where
    S: Zeros,
{
    fn zeros(shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        Self {
            data: RefCell::new(S::zeros(shape.clone())),
            grad: RefCell::new(S::zeros(shape)),
        }
    }
}