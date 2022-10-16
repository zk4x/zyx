//! Methods for tensor initialization
//! 

use crate::{tensor::Variable, ops::{self, FromVec}, shape::IntoShape};
use std::cell::RefCell;

// Different accelerators can be supported with either config flags similar to this, or with calls like Buffer::<opencl::Buffer<f32>>::convert_from(other_Buffer)
//#[cfg(Opencl)]
//type Storage<T> = opencl::Buffer<T>;

/// Initialize tensor from vec and shape
impl<S, T> ops::FromVec<T> for Variable<S>
where
    S: FromVec<T> + ops::Zeros,
{
    fn from_vec(data: Vec<T>, shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        debug_assert_eq!(data.len(), shape.numel());
        Self {
            data: RefCell::new(S::from_vec(data, shape.clone())),
            grad: RefCell::new(S::zeros(shape)),
        }
    }
}

/// Initialize tensor filled with zeros
impl<S> ops::Zeros for Variable<S>
where
    S: ops::Zeros,
{
    fn zeros(shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        Self {
            data: RefCell::new(S::zeros(shape.clone())),
            grad: RefCell::new(S::zeros(shape)),
        }
    }
}

/// Initialize tensor filled with ones
impl<S> ops::Ones for Variable<S>
where
    S: ops::Ones + ops::Zeros,
{
    fn ones(shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        Self {
            data: RefCell::new(S::ones(shape.clone())),
            grad: RefCell::new(S::zeros(shape)),
        }
    }
}
