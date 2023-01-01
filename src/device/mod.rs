//! Various implementations of devices.
//! The default is [CPU Buffer][cpu::Buffer].
//!
//! Every device can implement following traits in order to be fully usable:
//! ```txt
//! - Clone
//! - Device
//! - SType
//! - BufferFromSlice
//! - core::ops::{Neg, Add, Sub, Mul, Div}
//! - core::ops::Mul<f32> // for SGD optimizer
//! - zyx::ops::*
//! ```
//!
//! Some functors in [nn][crate::nn] also require the buffer to implement binary operations with anything that implements [DType](crate::device::DType).
//!
//! The [ops module](crate::ops) documents how these operations should work.
//!
//! ## If you want to provide your own device
//! 
//! All operations take buffer by value. Cloning can be implemented as shallow copying,
//! but you will need to do the necessary reference counting or use your [Device] as a storage and buffers can be just references.
//!

mod dtype;
pub mod cpu;
//#[cfg(features = "opencl")]
pub mod opencl;
//#[cfg(feature = "ndarray")]
//pub mod ndarray;

pub use dtype::{DType, SType};

/// All devices have implemented this trait.
/// Devices store information about the underlaying hardward.
/// 
/// Devices can also store actual tensor data.
/// 
/// Creating devices is simple:
/// ```
/// use zyx::device::{cpu, opencl};
/// 
/// let cpu_device = cpu::Device::default(); // creates device on the GPU
/// let gpu_device = opencl::Device::default(); // creates device on the GPU
/// ```
/// After creating a device, you use it to create actual [buffers](crate::device::cpu::Buffer).
/// Following example creates gpu buffer with values 2, 3 and 4 stored in it.
/// ```
/// # use zyx::prelude::*;
/// # use zyx::device::opencl;
/// # let gpu_device = opencl::Device::default();
/// let gpu_buffer = gpu_device.buffer([2, 3, 4]);
/// ```
/// Buffers implemet [SType](crate::device::SType). That means they can have gradients attached:
/// ```
/// # use zyx::prelude::*;
/// # use zyx::device::opencl;
/// # let gpu_device = opencl::Device::default();
/// # let gpu_buffer = gpu_device.buffer([2, 3, 4]);
/// let gpu_buffer = gpu_buffer.with_grad();
/// ```
/// NOTE that [buffers](crate::device::cpu::Buffer) are immutable! Any operation on buffer results in creation of a new [buffer](crate::device::cpu::Buffer)
/// or as in this case [Variable](crate::tensor::Variable).
/// This however doesn't hurt performace, because buffers use reference counting and possibly other techniques to avoid copying.
pub trait Device {}

use crate::{
    ops::{HasDType, HasShape},
    shape::{HasLastDim, Sh1, Sh2, Sh3, Sh4, Sh5, Shape},
};

/// Trait that allows us to create new buffer from given slice.
/// This is the only trait that must be implemented for all devices,
/// all other initialization methods are automatically implemented.
pub trait BufferFromSlice<'d, Buf: 'd + HasDType + HasShape> {
    /// Create new buffer from given slice
    fn slice(&'d self, slice: &[Buf::T]) -> Buf;
}

/// Various methods to create a new buffer.
pub trait BufferInit<'d, Buf>: BufferFromSlice<'d, Buf>
where
    Buf: 'd + HasDType + HasShape,
{
    /// Create new buffer filled with random values
    /// ```
    /// use zyx::prelude::*;
    /// use zyx::shape::Sh5;
    /// use zyx::device::opencl;
    /// let dev = opencl::Device::default();
    /// let randn_buffer: opencl::Buffer<'_, Sh5<2, 4, 1, 5, 2>> = dev.randn();
    /// ```
    fn randn(&'d self) -> Buf
    where
        rand::distributions::Standard: rand::prelude::Distribution<Buf::T>,
    {
        use rand::Rng;
        extern crate alloc;
        let mut rng = rand::thread_rng();
        self.slice(
            &core::iter::repeat(0)
                .take(Buf::Sh::NUMEL)
                .map(|_| rng.gen())
                .collect::<alloc::vec::Vec<Buf::T>>(),
        )
    }

    /// Create new buffer filled with values from uniform distribution
    /// ```
    /// use zyx::prelude::*;
    /// use zyx::shape::Sh5;
    /// use zyx::device::opencl;
    /// let dev = opencl::Device::default();
    /// // Creates buffer with values ranging from -1.0 to 5.0
    /// let uniform_buffer: opencl::Buffer<'_, Sh5<2, 4, 1, 5, 2>> = dev.uniform(-1., 5.);
    /// ```
    fn uniform(&'d self, low: Buf::T, high: Buf::T) -> Buf
    where
        Buf::T: rand::distributions::uniform::SampleUniform,
    {
        use rand::distributions::Distribution;
        extern crate alloc;
        let mut rng = rand::thread_rng();
        let dist = rand::distributions::Uniform::new(low, high);
        self.slice(
            &core::iter::repeat(0)
                .take(Buf::Sh::NUMEL)
                .map(|_| dist.sample(&mut rng))
                .collect::<alloc::vec::Vec<Buf::T>>(),
        )
    }

    /// Create new buffer filled with zeros
    /// ```
    /// use zyx::prelude::*;
    /// use zyx::shape::Sh5;
    /// use zyx::device::opencl;
    /// let dev = opencl::Device::default();
    /// let zeros_buffer: opencl::Buffer<'_, Sh5<2, 4, 1, 5, 2>> = dev.zeros();
    /// ```
    fn zeros(&'d self) -> Buf
    where
        Buf::T: Clone + num_traits::Zero,
    {
        extern crate alloc;
        use num_traits::Zero;
        self.slice(&alloc::vec![Buf::T::zero(); Buf::Sh::NUMEL])
    }

    /// Create new buffer filled with ones
    /// ```
    /// use zyx::prelude::*;
    /// use zyx::shape::Sh5;
    /// use zyx::device::opencl;
    /// let dev = opencl::Device::default();
    /// let ones_buffer: opencl::Buffer<'_, Sh5<2, 4, 1, 5, 2>> = dev.ones();
    /// ```
    fn ones(&'d self) -> Buf
    where
        Buf::T: Clone + num_traits::One,
    {
        extern crate alloc;
        use num_traits::One;
        self.slice(&alloc::vec![Buf::T::one(); Buf::Sh::NUMEL])
    }
}

impl<'d, Buf, Dev> BufferInit<'d, Buf> for Dev
where
    Dev: BufferFromSlice<'d, Buf>,
    Buf: 'd + HasDType + HasShape,
{
}

/// Create new buffer with shape automatically inferred
/// This trait has only one function.
/// You pass an array or an array of arrays and rust compiler
/// automatically infers the required [Shape](crate::shape::Shape).
pub trait ShapedBufferInit<'d, Input, Buf, Sh>: BufferFromSlice<'d, Buf>
where
    Buf: 'd + HasDType + HasShape<Sh = Sh>,
{
    /// Create new buffer with shape automatically inferred
    /// ```
    /// use zyx::prelude::*;
    /// use zyx::device::cpu;
    /// let dev = cpu::Device::default();
    /// let buffer = dev.buffer([4, 2, 5]); // also type is automatically inferred as i32 here
    /// ```
    fn buffer(&'d self, x: Input) -> Buf;
}

impl<'d, Dev, Buf, const D0: usize> ShapedBufferInit<'d, [Buf::T; D0], Buf, Sh1<D0>> for Dev
where
    Dev: BufferFromSlice<'d, Buf>,
    Buf: 'd + HasDType + HasShape<Sh = Sh1<D0>>,
{
    fn buffer(&'d self, x: [Buf::T; D0]) -> Buf {
        self.slice(&x)
    }
}

impl<'d, Dev, Buf, const D1: usize, const D0: usize>
    ShapedBufferInit<'d, [[Buf::T; D0]; D1], Buf, Sh2<D1, D0>> for Dev
where
    Dev: BufferFromSlice<'d, Buf>,
    Buf: 'd + HasDType + HasShape<Sh = Sh2<D1, D0>>,
{
    fn buffer(&'d self, x: [[Buf::T; D0]; D1]) -> Buf {
        extern crate alloc;
        self.slice(&x.into_iter().flatten().collect::<alloc::vec::Vec<Buf::T>>())
    }
}

impl<'d, Dev, Buf, const D2: usize, const D1: usize, const D0: usize>
    ShapedBufferInit<'d, [[[Buf::T; D0]; D1]; D2], Buf, Sh3<D2, D1, D0>> for Dev
where
    Dev: BufferFromSlice<'d, Buf>,
    Buf: 'd + HasDType + HasShape<Sh = Sh3<D2, D1, D0>>,
{
    fn buffer(&'d self, x: [[[Buf::T; D0]; D1]; D2]) -> Buf {
        extern crate alloc;
        self.slice(
            &x.into_iter()
                .flatten()
                .flatten()
                .collect::<alloc::vec::Vec<Buf::T>>(),
        )
    }
}

impl<'d, Dev, Buf, const D3: usize, const D2: usize, const D1: usize, const D0: usize>
    ShapedBufferInit<'d, [[[[Buf::T; D0]; D1]; D2]; D3], Buf, Sh4<D3, D2, D1, D0>> for Dev
where
    Dev: BufferFromSlice<'d, Buf>,
    Buf: 'd + HasDType + HasShape<Sh = Sh4<D3, D2, D1, D0>>,
{
    fn buffer(&'d self, x: [[[[Buf::T; D0]; D1]; D2]; D3]) -> Buf {
        extern crate alloc;
        self.slice(
            &x.into_iter()
                .flatten()
                .flatten()
                .flatten()
                .collect::<alloc::vec::Vec<Buf::T>>(),
        )
    }
}

impl<
        'd,
        Dev,
        Buf,
        const D4: usize,
        const D3: usize,
        const D2: usize,
        const D1: usize,
        const D0: usize,
    > ShapedBufferInit<'d, [[[[[Buf::T; D0]; D1]; D2]; D3]; D4], Buf, Sh5<D4, D3, D2, D1, D0>>
    for Dev
where
    Dev: BufferFromSlice<'d, Buf>,
    Buf: 'd + HasDType + HasShape<Sh = Sh5<D4, D3, D2, D1, D0>>,
{
    fn buffer(&'d self, x: [[[[[Buf::T; D0]; D1]; D2]; D3]; D4]) -> Buf {
        extern crate alloc;
        self.slice(
            &x.into_iter()
                .flatten()
                .flatten()
                .flatten()
                .flatten()
                .collect::<alloc::vec::Vec<Buf::T>>(),
        )
    }
}

/*
    fn eye<const N: usize>(&mut self) -> Buf
    where
        Buf::Sh: Sh2<N>,
        T: Clone + crate::num_traits::Zero + crate::ops::One,
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
// Use this trait in devices to easily implement display
trait NDBufferToString: HasDType + HasShape + crate::ops::IntoVec
where
    Self::Sh: HasLastDim,
    Self::T: core::fmt::Display,
{
    fn buffer_to_string(&self) -> alloc::string::String {
        extern crate alloc;
        use alloc::string::String;
        let mut res = String::new();
        let data = self.to_vec();
        if data.is_empty() {
            return res + "[]";
        }
        let n = Self::Sh::NUMEL;
        let ndim = Self::Sh::RANK;
        //const PRECISION: usize = 3;
        // get maximal width of single value
        let mut w = 0;
        for x in data.iter() {
            let l = alloc::format!("{x:w$}").len();
            if l > w {
                w = l;
            }
        }
        let d0 = Self::Sh::LAST_DIM;
        for (i, x) in data.iter().enumerate() {
            {
                let mut var = 1;
                let mut r = ndim;
                while r > 0 {
                    if i % (n / var) == 0 {
                        res += &(" ".repeat(ndim - r) + &"[".repeat(r - 1));
                        break;
                    }
                    var *= Self::Sh::at(ndim - r);
                    r -= 1;
                }
            }
            use core::fmt::Write;
            let _ = write!(res, "{0:>1$}", x, w);
            if (i + 1) % d0 != 0usize {
                res += " ";
            }
            {
                let mut var = 1;
                let mut r = ndim;
                while r > 0 {
                    if (i + 1) % (n / var) == 0 {
                        res += &"]".repeat(r - 1);
                        break;
                    }
                    var *= Self::Sh::at(ndim - r);
                    r -= 1;
                }
            }
            if (i + 1) % d0 == 0usize && i != n - 1 {
                res += "\n";
            }
        }
        res
    }
}

impl<Buf> NDBufferToString for Buf
where
    Buf: HasDType + HasShape + crate::ops::IntoVec,
    Buf::T: core::fmt::Display,
    Buf::Sh: HasLastDim,
{
}
