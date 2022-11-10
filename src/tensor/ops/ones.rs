use crate::{ops::{Ones, Zeros}, tensor::{Variable, Gradient}, shape::IntoShape};

/// Initialize tensor filled with zeros
impl<S> Ones for Variable<S>
where
    S: Ones + Zeros,
{
    fn ones(shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        Self {
            data: S::ones(shape),
            grad: Gradient::new(),
        }
    }
}