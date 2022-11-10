use crate::{ops::Zeros, tensor::{Variable, Gradient}, shape::IntoShape};

/// Initialize tensor filled with zeros
impl<S> Zeros for Variable<S>
where
    S: Zeros,
{
    fn zeros(shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        Self {
            data: S::zeros(shape),
            grad: Gradient::new(),
        }
    }
}