//! # Initialization methods for tensors
//! 
//! Includes [eye](EyeInit), [randn](RandnInit) and [uniform](UniforMinimizableit) as well as array initialization.
//! These are implemented for all data structures that implement [FromSlice](crate::ops::FromSlice) trait.

use crate::{ops::{FromSlice, HasShape, ConvertFrom, Zeros, Ones, HasDType}, shape::{Shape, Sh1, Sh2, Sh3, Sh4, Sh5}, dtype::DType};
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
    /// Initialize tensor with eye matrix
    fn eye() -> Self;
}

impl<S, const N: usize> EyeInit for S
where
    S: FromSlice + HasShape<Sh = Sh2<N, N>>,
    S::T: Zeros + Ones,
{
    fn eye() -> Self {
        let mut data = vec![S::T::zeros(); N*N];
        let mut i = 0;
        while i < N*N {
            data[i] = S::T::ones();
            i += N + 1;
        }
        Self::from_slice(&data)
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
    /// Initialize tensor with random numbers
    fn randn() -> Self;
}

impl<S> RandnInit for S
where
    S: FromSlice + HasShape, // TODO maybe change this HasShape requirement to HasLen requirement
    rand::distributions::Standard: rand::prelude::Distribution<S::T>,
{
    fn randn() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Self::from_slice(&core::iter::repeat(0).take(S::Sh::numel()).map(|_| rng.gen()).collect::<alloc::vec::Vec<S::T>>())
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
pub trait UniformInit: HasDType {
    /// Initialize tensor with random numbers from uniform distribution
    fn uniform(low: Self::T, high: Self::T) -> Self;
}

impl<S> UniformInit for S
where
    S: FromSlice + HasShape,
    S::T: rand::distributions::uniform::SampleUniform,
{
    fn uniform(low: Self::T, high: Self::T) -> Self {
        use rand::distributions::Distribution;
        let mut rng = rand::thread_rng();
        let dist = rand::distributions::Uniform::new(low, high);
        Self::from_slice(&core::iter::repeat(0).take(S::Sh::numel()).map(|_| dist.sample(&mut rng)).collect::<alloc::vec::Vec<S::T>>())
    }
}

// TODO: Figure wheter this is usefull. But currently we don't think it is usefull enough.
/*impl<S, F, T, Sh> ConvertFrom<(F, Sh)> for S
where
    T: Clone,
    S: FromSlice<T = T, Sh = Sh>,
    Sh: Shape,
    F: FnMut() -> T,
{
    fn cfrom(mut f: (F, Sh)) -> Self {
        S::from_slice(&core::iter::repeat(0).take(f.1.numel()).map(|_| f.0()).collect::<alloc::vec::Vec<T>>(), f.1)
    }
}

impl<S, T> ConvertFrom<T> for S
where
    S: FromSlice<dtype>,
    T: DType,
{
    fn cfrom(x: T) -> Self {
        S::from_slice(vec![x], [1])
    }
}*/

impl<S, T, const D0: usize> ConvertFrom<[T; D0]> for S
where
    S: FromSlice<T = T> + HasShape<Sh = Sh1<D0>>,
    T: DType + Clone,
{
    fn cfrom(x: [T; D0]) -> Self {
        S::from_slice(&x)
    }
}

impl<S, T, const D1: usize, const D0: usize> ConvertFrom<[[T; D0]; D1]> for S
where
    S: FromSlice<T = T> + HasShape<Sh = Sh2<D1, D0>>,
    T: DType + Clone,
{
    fn cfrom(x: [[T; D0]; D1]) -> Self {
        S::from_slice(&x.into_iter().flatten().collect::<alloc::vec::Vec<T>>())
    }
}

impl<S, T, const D2: usize, const D1: usize, const D0: usize> ConvertFrom<[[[T; D0]; D1]; D2]> for S
where
    S: FromSlice<T = T> + HasShape<Sh = Sh3<D2, D1, D0>>,
    T: DType + Clone
{
    fn cfrom(x: [[[T; D0]; D1]; D2]) -> Self {
        S::from_slice(&x.into_iter().flatten().flatten().collect::<alloc::vec::Vec<T>>())
    }
}

impl<S, T, const D3: usize, const D2: usize, const D1: usize, const D0: usize> ConvertFrom<[[[[T; D0]; D1]; D2]; D3]> for S
where
    S: FromSlice<T = T> + HasShape<Sh = Sh4<D3, D2, D1, D0>>,
    T: DType + Clone
{
    fn cfrom(x: [[[[T; D0]; D1]; D2]; D3]) -> Self {
        S::from_slice(&x.into_iter().flatten().flatten().flatten().collect::<alloc::vec::Vec<T>>())
    }
}

impl<S, const D4: usize, const D3: usize, const D2: usize, const D1: usize, const D0: usize> ConvertFrom<[[[[[S::T; D0]; D1]; D2]; D3]; D4]> for S
where
    S: FromSlice + HasShape<Sh = Sh5<D4, D3, D2, D1, D0>>,
    S::T: DType,
{
    fn cfrom(x: [[[[[S::T; D0]; D1]; D2]; D3]; D4]) -> Self {
        Self::from_slice(&x.into_iter().flatten().flatten().flatten().flatten().collect::<alloc::vec::Vec<S::T>>())
    }
}
