//! # Initialization methods for tensors
//! 
//! Includes eye, randn and uniform as well as array initialization.
//! These are implemented for all data structures that implement ops::FromVec trait.

use crate::{ops::{FromVec, ConvertFrom, Zeros, Ones}, shape::IntoShape};

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

/// ## Eye initialization
/// 
/// Initialize tensor with eye matrix
/// 
/// ### Example
/// 
/// ```
/// use zyx::prelude::*;
/// use zyx::accel::cpu::Buffer;
/// 
/// let x = Buffer::<i32>::eye(3);
/// assert_eq!(x.to_vec(), vec![1, 0, 0, 0, 1, 0, 0, 0, 1]);
/// assert_eq!(x.shape(), (3, 3));
/// println!("{}", x);
/// ```
/// 
/// ### Output
/// 
/// ```txt
/// [1 0 0
///  0 1 0
///  0 0 1]
/// ```
pub trait EyeInit<T> {
    fn eye(n: usize) -> Self;
}

impl<S, T> EyeInit<T> for S
where
    S: FromVec<T>,
    T: Clone + Zeros + Ones,
{
    fn eye(n: usize) -> Self {
        let mut data = vec![T::zeros(()); n*n];
        let mut i = 0;
        while i < n*n {
            data[i] = T::ones(());
            i += n + 1;
        }
        Self::from_vec(data, (n, n))
    }
}

/// ## Random initialization
/// 
/// Initialize tensor with random values
/// We are using rand::distributions::Standard.
/// 
/// ### Example
/// 
/// ```
/// use zyx::prelude::*;
/// use zyx::accel::cpu::Buffer;
/// 
/// let x = Buffer::<f32>::randn((3, 2, 3));
/// assert_eq!(x.shape(), (3, 2, 3));
/// println!("{}", x);
/// ```
pub trait RandInit<T> {
    fn randn(shape: impl IntoShape) -> Self;
}

impl<S, T> RandInit<T> for S
where
    S: FromVec<T>,
    rand::distributions::Standard: rand::prelude::Distribution<T>,
{
    fn randn(shape: impl IntoShape) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let shape = shape.shape();
        Self::from_vec(std::iter::repeat(0).take(shape.numel()).map(|_| rng.gen()).collect(), shape)
    }
}

/// ## Uniform initialization
/// 
/// Initialize tensor with value from uniform distribution
/// We are using rand::distributions::uniform.
/// 
/// ### Example
/// 
/// ```
/// use zyx::prelude::*;
/// use zyx::accel::cpu::Buffer;
/// 
/// let x = Buffer::uniform((3, 2, 3), -1., 1.);
/// assert_eq!(x.shape(), (3, 2, 3));
/// println!("{}", x);
/// ```
pub trait UniformInit<T> {
    fn uniform(shape: impl IntoShape, low: T, high: T) -> Self;
}

impl<S, T> UniformInit<T> for S
where
    S: FromVec<T>,
    T: rand::distributions::uniform::SampleUniform,
{
    fn uniform(shape: impl IntoShape, low: T, high: T) -> Self {
        use rand::distributions::Distribution;
        let mut rng = rand::thread_rng();
        let dist = rand::distributions::Uniform::new(low, high);
        let shape = shape.shape();
        Self::from_vec(std::iter::repeat(0).take(shape.numel()).map(|_| dist.sample(&mut rng)).collect(), shape)
    }
}

impl<S, F, T, Sh> ConvertFrom<(F, Sh)> for S
where
    T: Clone,
    S: FromVec<T>,
    Sh: IntoShape,
    F: FnMut() -> T,
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
