//! Various implementations of deviceerators.
//! The default is [CPU Buffer][cpu::Buffer].
//! 

// Every device can implement following traits in order to be fully compatible with tensors:
// ```txt
// - Clone
// - Device
// - BufferFromSlice
// - core::ops::{Neg, Add, Sub, Mul, Div}
// - core::ops::Mul<f32> // for SGD optimizer
// - zyx::ops::*
// ```
// 
// Some functors in [nn module][crate::nn] also require the buffer to implement binary operations with anything that implements [DType](crate::dtype::DType).
// 
// The [ops module](crate::ops) documents how these operations should work.
// 
// All operations take buffer by value. Cloning can be implemented as shallow copying,
// but you will need to do the necessary reference counting.
// 

pub mod cpu;
//#[cfg(features = "opencl")]
pub mod opencl;
//#[cfg(feature = "ndarray")]
//pub mod ndarray;

pub trait Device {}

use crate::{ops::{HasDType, HasShape}, shape::{Shape, Sh1, Sh2, Sh3, Sh4, Sh5}, dtype::DType};

pub trait BufferFromSlice<Sh, T = f32> {
    type Buffer: HasDType<T = T> + HasShape<Sh = Sh>;
    fn _slice(&mut self, slice: &[T]) -> Self::Buffer;
}

pub trait BufferInit {
    fn slice<Sh, T>(&mut self, slice: &[T]) -> <Self as BufferFromSlice<Sh, T>>::Buffer
    where
        Self: BufferFromSlice<Sh, T>,
    {
        self._slice(slice)
    }
    
    fn randn<Sh, T>(&mut self) -> <Self as BufferFromSlice<Sh, T>>::Buffer
    where
        Sh: Shape,
        Self: BufferFromSlice<Sh, T>,
        rand::distributions::Standard: rand::prelude::Distribution<T>,
    {
        use rand::Rng;
        extern crate alloc;
        let mut rng = rand::thread_rng();
        self._slice(&core::iter::repeat(0).take(Sh::numel()).map(|_| rng.gen()).collect::<alloc::vec::Vec<T>>())
    }
    
    fn uniform<Sh, T>(&mut self, low: T, high: T) -> <Self as BufferFromSlice<Sh, T>>::Buffer
    where
        Sh: Shape,
        Self: BufferFromSlice<Sh, T>,
        T: rand::distributions::uniform::SampleUniform,
    {
        use rand::distributions::Distribution;
        extern crate alloc;
        let mut rng = rand::thread_rng();
        let dist = rand::distributions::Uniform::new(low, high);
        self._slice(&core::iter::repeat(0).take(Sh::numel()).map(|_| dist.sample(&mut rng)).collect::<alloc::vec::Vec<T>>())
    }

    fn eye<const N: usize, T>(&mut self) -> <Self as BufferFromSlice<Sh2<N, N>, T>>::Buffer
    where
        Self: BufferFromSlice<Sh2<N, N>, T>,
        T: Clone + crate::ops::Zero + crate::ops::One,
    {
        extern crate alloc;
        let mut data = alloc::vec![T::zero(); N*N];
        let mut i = 0;
        while i < N*N {
            data[i] = T::one();
            i += N + 1;
        }
        self._slice(&data)
    }
    
    fn zeros<Sh, T>(&mut self) -> <Self as BufferFromSlice<Sh, T>>::Buffer
    where
        Sh: Shape,
        Self: BufferFromSlice<Sh, T>,
        T: Clone + crate::ops::Zero,
    {
        extern crate alloc;
        self._slice(&alloc::vec![T::zero(); Sh::numel()])
    }
    
    fn ones<Sh, T>(&mut self) -> <Self as BufferFromSlice<Sh, T>>::Buffer
    where
        Sh: Shape,
        Self: BufferFromSlice<Sh, T>,
        T: Clone + crate::ops::One,
    {
        extern crate alloc;
        self._slice(&alloc::vec![T::one(); Sh::numel()])
    }
}

impl<S> BufferInit for S {}

pub trait ShapedBufferInit<Input, Sh, T>: BufferFromSlice<Sh, T> {
    fn buffer(&mut self, x: Input) -> Self::Buffer;
}

impl<S, T, const D0: usize> ShapedBufferInit<[T; D0], Sh1<D0>, T> for S
where
    S: BufferFromSlice<Sh1<D0>, T>,
    T: DType,
{
    fn buffer(&mut self, x: [T; D0]) -> Self::Buffer {
        self._slice(&x)
    }
}

impl<S, T, const D1: usize, const D0: usize> ShapedBufferInit<[[T; D0]; D1], Sh2<D1, D0>, T> for S
where
    S: BufferFromSlice<Sh2<D1, D0>, T>,
    T: DType,
{
    fn buffer(&mut self, x: [[T; D0]; D1]) -> Self::Buffer {
        extern crate alloc;
        self._slice(&x.into_iter().flatten().collect::<alloc::vec::Vec<T>>())
    }
}

impl<S, T, const D2: usize, const D1: usize, const D0: usize> ShapedBufferInit<[[[T; D0]; D1]; D2], Sh3<D2, D1, D0>, T> for S
where
    S: BufferFromSlice<Sh3<D2, D1, D0>, T>,
    T: DType,
{
    fn buffer(&mut self, x: [[[T; D0]; D1]; D2]) -> Self::Buffer {
        extern crate alloc;
        self._slice(&x.into_iter().flatten().flatten().collect::<alloc::vec::Vec<T>>())
    }
}

impl<S, T, const D3: usize, const D2: usize, const D1: usize, const D0: usize> ShapedBufferInit<[[[[T; D0]; D1]; D2]; D3], Sh4<D3, D2, D1, D0>, T> for S
where
    S: BufferFromSlice<Sh4<D3, D2, D1, D0>, T>,
    T: DType,
{
    fn buffer(&mut self, x: [[[[T; D0]; D1]; D2]; D3]) -> Self::Buffer {
        extern crate alloc;
        self._slice(&x.into_iter().flatten().flatten().flatten().collect::<alloc::vec::Vec<T>>())
    }
}

impl<S, T, const D4: usize, const D3: usize, const D2: usize, const D1: usize, const D0: usize> ShapedBufferInit<[[[[[T; D0]; D1]; D2]; D3]; D4], Sh5<D4, D3, D2, D1, D0>, T> for S
where
    S: BufferFromSlice<Sh5<D4, D3, D2, D1, D0>, T>,
    T: DType,
{
    fn buffer(&mut self, x: [[[[[T; D0]; D1]; D2]; D3]; D4]) -> Self::Buffer {
        extern crate alloc;
        self._slice(&x.into_iter().flatten().flatten().flatten().flatten().collect::<alloc::vec::Vec<T>>())
    }
}
