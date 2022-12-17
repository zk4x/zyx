//! # Initialization methods for tensors
//! 
//! Includes [eye](EyeInit), [randn](RandnInit) and [uniform](UniformInit) as well as array initialization.
//! These are implemented for all data structures that implement [FromVec](crate::ops::FromVec) trait.

use crate::{ops::{FromVec, ConvertFrom, Zeros, Ones}, shape::Shape, dtype::ScalarType};
extern crate alloc;
use alloc::vec;

/// # Eye initialization
/// 
/// Initialize tensor with eye matrix.
/// 
/// ## Example
/// 
/// ```
/// use zyx::prelude::*;
/// use zyx::accel::cpu::Buffer;
/// 
/// let x = Buffer::<i32, _>::eye(3);
/// assert_eq!(x.to_vec(), vec![1, 0, 0, 0, 1, 0, 0, 0, 1]);
/// assert_eq!(x.shape(), (3, 3));
/// println!("{}", x);
/// ```
/// 
/// ## Output
/// 
/// ```txt
/// [1 0 0
///  0 1 0
///  0 0 1]
/// ```
pub trait EyeInit {
    /// dtype of values stored in eye matrix
    type T;
    /// Initialize tensor with eye matrix
    fn eye(n: usize) -> Self;
}

impl<S, T> EyeInit for S
where
    S: FromVec<T = T, Sh = (usize, usize)>,
    T: Clone + Zeros<Sh = ()> + Ones<Sh = ()>,
{
    type T = T;

    fn eye(n: usize) -> Self {
        let mut data = vec![T::zeros(()); n*n];
        let mut i = 0;
        while i < n*n {
            data[i] = T::ones(());
            i += n + 1;
        }
        Self::from_vec(&data, (n, n))
    }
}

/// # Random initialization
/// 
/// Initialize tensor with random values
/// We are using rand::distributions::Standard.
/// 
/// ## Example
/// 
/// ```
/// use zyx::prelude::*;
/// use zyx::accel::cpu::Buffer;
/// 
/// let x = Buffer::<f32, _>::randn((3usize, 2, 3));
/// assert_eq!(x.shape(), (3, 2, 3));
/// println!("{}", x);
/// ```
pub trait RandnInit {
    /// dtype of values stored in random matrix
    type T;
    /// [Shape] of random matrix
    type Sh: Shape<D = usize>;
    /// Initialize tensor with random numbers
    fn randn(shape: Self::Sh) -> Self;
}

impl<S, T, Sh> RandnInit for S
where
    S: FromVec<T = T, Sh = Sh>,
    rand::distributions::Standard: rand::prelude::Distribution<T>,
    Sh: Shape<D = usize>,
{
    type T = T;
    type Sh = Sh;

    fn randn(shape: Sh) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Self::from_vec(&core::iter::repeat(0).take(shape.numel()).map(|_| rng.gen()).collect::<alloc::vec::Vec<T>>(), shape)
    }
}

/// # Uniform initialization
/// 
/// Initialize tensor with value from uniform distribution
/// We are using rand::distributions::uniform.
/// 
/// ## Example
/// 
/// ```
/// use zyx::prelude::*;
/// use zyx::accel::cpu::Buffer;
/// 
/// let x = Buffer::uniform((3usize, 2, 3), -1., 1.);
/// assert_eq!(x.shape(), (3, 2, 3));
/// println!("{}", x);
/// ```
pub trait UniformInit {
    /// dtype of values stored in random matrix
    type T;
    /// [Shape] of matrix initialized from uniform distribution
    type Sh: Shape<D = usize>;
    /// Initialize tensor with random numbers from uniform distribution
    fn uniform(shape: Self::Sh, low: Self::T, high: Self::T) -> Self;
}

impl<S, T, Sh> UniformInit for S
where
    S: FromVec<T = T, Sh = Sh>,
    T: rand::distributions::uniform::SampleUniform,
    Sh: Shape<D = usize>,
{
    type T = T;
    type Sh = Sh;

    fn uniform(shape: Sh, low: T, high: T) -> Self {
        use rand::distributions::Distribution;
        let mut rng = rand::thread_rng();
        let dist = rand::distributions::Uniform::new(low, high);
        Self::from_vec(&core::iter::repeat(0).take(shape.numel()).map(|_| dist.sample(&mut rng)).collect::<alloc::vec::Vec<T>>(), shape)
    }
}

impl<S, F, T, Sh> ConvertFrom<(F, Sh)> for S
where
    T: Clone,
    S: FromVec<T = T, Sh = Sh>,
    Sh: Shape<D = usize>,
    F: FnMut() -> T,
{
    fn cfrom(mut f: (F, Sh)) -> Self {
        S::from_vec(&core::iter::repeat(0).take(f.1.numel()).map(|_| f.0()).collect::<alloc::vec::Vec<T>>(), f.1)
    }
}

// TODO: Figure wheter this is usefull. But currently we don't think it is usefull enough.
/*impl<S, T> ConvertFrom<T> for S
where
    S: FromVec<dtype>,
    T: ScalarType,
{
    fn cfrom(x: T) -> Self {
        S::from_vec(vec![x], [1])
    }
}*/

impl<S, T, const D0: usize> ConvertFrom<[T; D0]> for S
where
    S: FromVec<T = T, Sh = usize>,
    T: ScalarType + Clone,
{
    fn cfrom(x: [T; D0]) -> Self {
        S::from_vec(&x, D0)
    }
}

impl<S, T, const D1: usize, const D0: usize> ConvertFrom<[[T; D0]; D1]> for S
where
    S: FromVec<T = T, Sh = (usize, usize)>,
    T: ScalarType + Clone,
{
    fn cfrom(x: [[T; D0]; D1]) -> Self {
        S::from_vec(&x.into_iter().flatten().collect::<alloc::vec::Vec<T>>(), (D1, D0))
    }
}

impl<S, T, const D2: usize, const D1: usize, const D0: usize> ConvertFrom<[[[T; D0]; D1]; D2]> for S
where
    S: FromVec<T = T, Sh = (usize, usize, usize)>,
    T: ScalarType + Clone
{
    fn cfrom(x: [[[T; D0]; D1]; D2]) -> Self {
        S::from_vec(&x.into_iter().flatten().flatten().collect::<alloc::vec::Vec<T>>(), (D2, D1, D0))
    }
}

impl<S, T, const D3: usize, const D2: usize, const D1: usize, const D0: usize> ConvertFrom<[[[[T; D0]; D1]; D2]; D3]> for S
where
    S: FromVec<T = T, Sh = (usize, usize, usize, usize)>,
    T: ScalarType + Clone
{
    fn cfrom(x: [[[[T; D0]; D1]; D2]; D3]) -> Self {
        S::from_vec(&x.into_iter().flatten().flatten().flatten().collect::<alloc::vec::Vec<T>>(), (D3, D2, D1, D0))
    }
}

impl<S, T, const D4: usize, const D3: usize, const D2: usize, const D1: usize, const D0: usize> ConvertFrom<[[[[[T; D0]; D1]; D2]; D3]; D4]> for S
where
    S: FromVec<T = T, Sh = (usize, usize, usize, usize, usize)>,
    T: ScalarType + Clone
{
    fn cfrom(x: [[[[[T; D0]; D1]; D2]; D3]; D4]) -> Self {
        Self::from_vec(&x.into_iter().flatten().flatten().flatten().flatten().collect::<alloc::vec::Vec<T>>(), (D4, D3, D2, D1, D0))
    }
}
