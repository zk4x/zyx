//! CPU Module
//! 
//! This module contains implementation of opencl device and buffer
//! OpenCL [device](super::Device) is a struct that holds information about your GPU.
//! Optionally, this can also use your CPU, with appropriate runtime, such as [pocl](http://portablecl.org/).
//! 
//! [Buffer] is multidimensional storage type using gpu for the calculations.
//! 
//! Create [opencl device](Device):
//! ```
//! use zyx::device::opencl;
//! let device = opencl::Device::default();
//! ```
//! Create a [Buffer] on this [Device]:
//! ```
//! # use zyx::prelude::*;
//! # use zyx::device::opencl;
//! # let device = opencl::Device::default();
//! let buffer = device.buffer([[4, 5, 2], [5, 2, 1]]);
//! ```

use crate::{device::DType, ops, shape::Shape};
use core::marker::PhantomData;

use ocl::{self, OclPrm};

extern crate alloc;

use super::BufferFromSlice;

/*static UNARY_KERNEL: &str = "
    __kernel void NAME_DTYPE(__global DTYPE* data, __global DTYPE* res) {
        int id = get_global_id(0);
        res[id] = OP(data[id]);
    }
";*/

/// OpenCL device
///
/// This device provides acces to OpenCL devices on your computer.
/// Buffers created using this device are stored in GPU memory and computations are done using the GPU.
/// 
/// Create [opencl device](Device):
/// ```
/// use zyx::device::opencl;
/// let device = opencl::Device::default();
/// ```
#[derive(Debug, Clone)]
pub struct Device {
    context: ocl::Context,
    device: ocl::Device, // TODO support multiple devices
}

impl crate::device::Device for Device {}

impl Default for Device {
    fn default() -> Self {
        let platform = ocl::Platform::list()[0];
        //std::println!("Using platform {}", platform);
        let context = ocl::Context::builder()
            .platform(platform)
            .devices(ocl::builders::DeviceSpecifier::First)
            .build()
            .expect("Couldn't create context");
        let device = context.devices()[0];
        //std::println!("{}", context);
        Self { context, device }
    }
}

impl<'d, Sh, T> BufferFromSlice<'d, Buffer<'d, Sh, T>> for Device
where
    T: OclPrm + DType,
    Sh: 'd + Shape,
{
    fn slice(&'d self, slice: &[T]) -> Buffer<'d, Sh, T> {
        Buffer::<'d, Sh, T> {
            device: self,
            data: ocl::Buffer::builder()
                .queue(
                    ocl::Queue::new(&self.context, self.device, None)
                        .expect("Couldn't create queue"),
                )
                .copy_host_slice(slice)
                .len(Sh::NUMEL)
                .flags(ocl::flags::MEM_READ_ONLY)
                .build()
                .expect("Unable to create buffer"),
            shape: PhantomData,
        }
    }
}

#[test]
fn opencl_device() {
    use crate::device::opencl;
    use crate::prelude::*;
    use crate::shape::Sh3;

    let device = opencl::Device::default();

    let _x = device.buffer([3, 4, 2]);

    let x: opencl::Buffer<'_, Sh3<2, 4, 3>> = device.uniform(0., 4.);

    std::println!("{}", x);

    //panic!();
}

/// OpenCL buffer
///
/// Each buffer has a shape and data.
/// Data is stored in row major order.
/// 
/// By default buffers use f32 [DType](crate::device::DType).
/// 
/// Create opencl [Buffer]:
/// ```
/// use zyx::prelude::*;
/// use zyx::device::opencl;
/// let device = opencl::Device::default();
/// let buffer = device.buffer([4, 1, 6]);
/// ```
#[derive(Debug, Clone)]
pub struct Buffer<'d, Sh, T = f32>
where
    T: OclPrm + DType,
    Sh: Shape,
{
    data: ocl::Buffer<T>,
    device: &'d Device,
    shape: PhantomData<Sh>,
}

impl<Sh, T> core::fmt::Display for Buffer<'_, Sh, T>
where
    Sh: Shape + crate::shape::HasLastDim,
    T: OclPrm + DType + core::fmt::Display + ops::Zero,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use super::NDBufferToString;
        f.write_str(&self.buffer_to_string())
    }
}

impl<Sh, T> ops::HasDevice for Buffer<'_, Sh, T>
where
    T: OclPrm + DType,
    Sh: Shape,
{
    type Dev = Device;
    fn device(&self) -> &Self::Dev {
        self.device
    }
}

impl<Sh, T> ops::HasDType for Buffer<'_, Sh, T>
where
    T: OclPrm + DType,
    Sh: Shape,
{
    type T = T;
}

impl<Sh, T> ops::HasShape for Buffer<'_, Sh, T>
where
    T: OclPrm + DType,
    Sh: Shape,
{
    type Sh = Sh;
}

impl<Sh, T> ops::ZerosLike for Buffer<'_, Sh, T>
where
    for<'a> Sh: Shape + 'a,
    T: DType + ocl::OclPrm + ops::Zero,
{
    fn zeros_like(&self) -> Self {
        use super::BufferInit;
        self.device.zeros()
    }
}

impl<Sh, T> ops::IntoVec<T> for Buffer<'_, Sh, T>
where
    T: OclPrm + DType + ops::Zero,
    Sh: Shape,
{
    fn to_vec(&self) -> alloc::vec::Vec<T> {
        extern crate alloc;
        let mut res = alloc::vec![T::zero(); Sh::NUMEL];
        self.data
            .read(&mut res)
            .enq()
            .expect("Couldn't read buffer");
        res
    }
}
