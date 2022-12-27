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

/// This trait must be implemented for all devices
pub trait Device {}

use crate::{ops::{HasDType, HasShape}, shape::{Shape, Sh1, Sh2, Sh3, Sh4, Sh5, HasLastDim}};

/// Trait that allows us to create new buffer from given slice.
/// This is the only trait that must be implemented for all devices,
/// all other initialization methods are automatically implemented.
pub trait BufferFromSlice<'d, Buf: 'd + HasDType + HasShape> {
    fn slice(&'d mut self, slice: &[Buf::T]) -> Buf;
}

/// Various methods to create a new buffer
pub trait BufferInit<'d, Buf>: BufferFromSlice<'d, Buf>
where
    Buf: 'd + HasDType + HasShape,
{
    /// Create new buffer filled with random values
    fn randn(&'d mut self) -> Buf
    where
        rand::distributions::Standard: rand::prelude::Distribution<Buf::T>,
    {
        use rand::Rng;
        extern crate alloc;
        let mut rng = rand::thread_rng();
        self.slice(&core::iter::repeat(0).take(Buf::Sh::numel()).map(|_| rng.gen()).collect::<alloc::vec::Vec<Buf::T>>())
    }
    
    /// Create new buffer filled with values from uniform distribution
    fn uniform(&'d mut self, low: Buf::T, high: Buf::T) -> Buf
    where
        Buf::T: rand::distributions::uniform::SampleUniform,
    {
        use rand::distributions::Distribution;
        extern crate alloc;
        let mut rng = rand::thread_rng();
        let dist = rand::distributions::Uniform::new(low, high);
        self.slice(&core::iter::repeat(0).take(Buf::Sh::numel()).map(|_| dist.sample(&mut rng)).collect::<alloc::vec::Vec<Buf::T>>())
    }
    
    /// Create new buffer filled with zeros
    fn zeros(&'d mut self) -> Buf
    where
        Buf::T: Clone + crate::ops::Zero,
    {
        extern crate alloc;
        use crate::ops::Zero;
        self.slice(&alloc::vec![Buf::T::zero(); Buf::Sh::numel()])
    }
    
    /// Create new buffer filled with ones
    fn ones(&'d mut self) -> Buf
    where
        Buf::T: Clone + crate::ops::One,
    {
        extern crate alloc;
        use crate::ops::One;
        self.slice(&alloc::vec![Buf::T::one(); Buf::Sh::numel()])
    }
}

impl<'d, Buf, Dev> BufferInit<'d, Buf> for Dev where Dev: BufferFromSlice<'d, Buf>, Buf: 'd + HasDType + HasShape {}

pub trait ShapedBufferInit<'d, Input, Buf, Sh>: BufferFromSlice<'d, Buf>
where
    Buf: 'd + HasDType + HasShape<Sh = Sh>,
{
    fn buffer(&'d mut self, x: Input) -> Buf;
}

impl<'d, Dev, Buf, const D0: usize> ShapedBufferInit<'d, [Buf::T; D0], Buf, Sh1<D0>> for Dev
where
    Dev: BufferFromSlice<'d, Buf>,
    Buf: 'd + HasDType + HasShape<Sh = Sh1<D0>>,
{
    fn buffer(&'d mut self, x: [Buf::T; D0]) -> Buf {
        self.slice(&x)
    }
}

impl<'d, Dev, Buf, const D1: usize, const D0: usize> ShapedBufferInit<'d, [[Buf::T; D0]; D1], Buf, Sh2<D1, D0>> for Dev
where
    Dev: BufferFromSlice<'d, Buf>,
    Buf: 'd + HasDType + HasShape<Sh = Sh2<D1, D0>>,
{
    fn buffer(&'d mut self, x: [[Buf::T; D0]; D1]) -> Buf {
        extern crate alloc;
        self.slice(&x.into_iter().flatten().collect::<alloc::vec::Vec<Buf::T>>())
    }
}

impl<'d, Dev, Buf, const D2: usize, const D1: usize, const D0: usize> ShapedBufferInit<'d, [[[Buf::T; D0]; D1]; D2], Buf, Sh3<D2, D1, D0>> for Dev
where
    Dev: BufferFromSlice<'d, Buf>,
    Buf: 'd + HasDType + HasShape<Sh = Sh3<D2, D1, D0>>,
{
    fn buffer(&'d mut self, x: [[[Buf::T; D0]; D1]; D2]) -> Buf {
        extern crate alloc;
        self.slice(&x.into_iter().flatten().flatten().collect::<alloc::vec::Vec<Buf::T>>())
    }
}

impl<'d, Dev, Buf, const D3: usize, const D2: usize, const D1: usize, const D0: usize> ShapedBufferInit<'d, [[[[Buf::T; D0]; D1]; D2]; D3], Buf, Sh4<D3, D2, D1, D0>> for Dev
where
    Dev: BufferFromSlice<'d, Buf>,
    Buf: 'd + HasDType + HasShape<Sh = Sh4<D3, D2, D1, D0>>,
{
    fn buffer(&'d mut self, x: [[[[Buf::T; D0]; D1]; D2]; D3]) -> Buf {
        extern crate alloc;
        self.slice(&x.into_iter().flatten().flatten().flatten().collect::<alloc::vec::Vec<Buf::T>>())
    }
}

impl<'d, Dev, Buf, const D4: usize, const D3: usize, const D2: usize, const D1: usize, const D0: usize> ShapedBufferInit<'d, [[[[[Buf::T; D0]; D1]; D2]; D3]; D4], Buf, Sh5<D4, D3, D2, D1, D0>> for Dev
where
    Dev: BufferFromSlice<'d, Buf>,
    Buf: 'd + HasDType + HasShape<Sh = Sh5<D4, D3, D2, D1, D0>>,
{
    fn buffer(&'d mut self, x: [[[[[Buf::T; D0]; D1]; D2]; D3]; D4]) -> Buf {
        extern crate alloc;
        self.slice(&x.into_iter().flatten().flatten().flatten().flatten().collect::<alloc::vec::Vec<Buf::T>>())
    }
}

/*
    fn eye<const N: usize>(&mut self) -> Buf
    where
        Buf::Sh: Sh2<N>,
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
*/

extern crate alloc;
trait NDBufferToString: HasDType + HasShape + crate::ops::IntoVec<Self::T>
where
    Self::Sh: HasLastDim,
    Self::T: core::fmt::Display,
{
    fn buffer_to_string(&self) -> alloc::string::String {
        extern crate alloc;
        use alloc::string::String;
        let mut res = String::new();
        let data = self.to_vec();
        if data.is_empty() { return res + "[]"; }
        let n = Self::Sh::numel();
        let ndim = Self::Sh::RANK;
        //const PRECISION: usize = 3;
        // get maximal width of single value
        let mut w = 0;
        for x in data.iter() {
            let l = alloc::format!("{x:w$}").len();
            if l > w { w = l; }
        }
        let d0 = Self::Sh::LAST_DIM;
        for i in 0..n {
            {
                let mut var = 1;
                let mut r = ndim;
                while r > 0 {
                    if i % (n/var) == 0 {
                        res += &(" ".repeat(ndim - r)+&"[".repeat(r - 1));
                        break
                    }
                    var *= Self::Sh::at(ndim - r);
                    r -= 1;
                }
            }
            use core::fmt::Write;
            let _ = write!(res, "{0:>1$}", data[i], w);
            if (i + 1) % d0 != 0usize { res += " "; }
            {
                let mut var = 1;
                let mut r = ndim;
                while r > 0 {
                    if (i + 1) % (n/var) == 0 {
                        res += &"]".repeat(r-1);
                        break
                    }
                    var *= Self::Sh::at(ndim - r);
                    r -= 1;
                }
            }
            if (i + 1) % d0 == 0usize && i != n - 1 { res += "\n"; }
        }
        res
    }
}

impl<Buf> NDBufferToString for Buf
where
    Buf: HasDType + HasShape + crate::ops::IntoVec<Buf::T>,
    Buf::T: core::fmt::Display,
    Buf::Sh: HasLastDim,
{}
