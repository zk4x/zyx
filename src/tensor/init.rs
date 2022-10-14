//! Methods for Buffer initialization
//! 

use crate::{tensor::Variable, ops::{self, FromVec, ConvertFrom}, shape::IntoShape};
use std::cell::RefCell;

pub trait DType {}

impl DType for f32 {}
impl DType for f64 {}
impl DType for u8 {}
impl DType for i8 {}
impl DType for i16 {}
impl DType for i32 {}
impl DType for i64 {}
impl DType for i128 {}
impl DType for bool {}

// Different accelerators can be supported with either config flags similar to this, or with calls like Buffer::<opencl::Buffer<f32>>::convert_from(other_Buffer)
//#[cfg(Opencl)]
//type Storage<T> = opencl::Buffer<T>;

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

impl<S, F, T, Sh> ConvertFrom<(F, Sh)> for S
where
    T: Clone,
    S: FromVec<T>,
    F: FnMut() -> T,
    Sh: IntoShape,
{
    fn cfrom(mut f: (F, Sh)) -> Self {
        let shape = f.1.shape();
        S::from_vec(std::iter::repeat(0).take(shape.numel()).map(|_| f.0()).collect(), shape)
    }
}

impl<S, T, const D0: usize> ConvertFrom<[T; D0]> for S
where
    S: FromVec<T>,
    T: DType + Clone,
{
    fn cfrom(x: [T; D0]) -> Self {
        S::from_vec(x.to_vec(), [D0])
    }
}

impl<S, T, const D1: usize, const D0: usize> ConvertFrom<[[T; D0]; D1]> for S
where
    S: FromVec<T>,
    T: DType + Clone,
{
    fn cfrom(x: [[T; D0]; D1]) -> Self {
        S::from_vec(x.into_iter().flatten().collect(), [D1, D0])
    }
}

impl<S, T, const D2: usize, const D1: usize, const D0: usize> ConvertFrom<[[[T; D0]; D1]; D2]> for S
where
    S: FromVec<T>,
    T: DType + Clone
{
    fn cfrom(x: [[[T; D0]; D1]; D2]) -> Self {
        S::from_vec(x.into_iter().flatten().flatten().collect(), [D2, D1, D0])
    }
}

impl<S, T, const D3: usize, const D2: usize, const D1: usize, const D0: usize> ConvertFrom<[[[[T; D0]; D1]; D2]; D3]> for S
where
    S: FromVec<T>,
    T: DType + Clone
{
    fn cfrom(x: [[[[T; D0]; D1]; D2]; D3]) -> Self {
        S::from_vec(x.into_iter().flatten().flatten().flatten().collect(), [D3, D2, D1, D0])
    }
}

impl<S, T, const D4: usize, const D3: usize, const D2: usize, const D1: usize, const D0: usize> ConvertFrom<[[[[[T; D0]; D1]; D2]; D3]; D4]> for S
where
    S: FromVec<T>,
    T: DType + Clone
{
    fn cfrom(x: [[[[[T; D0]; D1]; D2]; D3]; D4]) -> Self {
        Self::from_vec(x.into_iter().flatten().flatten().flatten().flatten().collect(), [D4, D3, D2, D1, D0])
    }
}
