//! This module contains implementation of opencl device and buffer

use core::marker::PhantomData;
use crate::{dtype::DType, shape::Shape, ops};

use ocl::{self, OclPrm};

use super::BufferFromSlice;

static UNARY_KERNEL: &str = "
    __kernel void NAME(__global float* data, __global float* res) {
        int id = get_global_id(0);
        res[id] = OP(data[id]);
    }
";

/// OpenCL device
/// 
/// This device provides acces to OpenCL devices on your computer.
/// Buffers created using this device are stored in GPU memory and computations are done using the GPU.
#[derive(Debug, Clone)]
pub struct Device {
    pro_que: ocl::ProQue,
}

impl super::Device for Device {}

impl Default for Device {
    fn default() -> Self {
        Self {
            pro_que: ocl::ProQue::builder()
                .src(UNARY_KERNEL.replace("NAME", "exp_kernel").replace("OP", "exp"))
                .dims(1 << 20)
                .build().expect("Couldn't create opencl pro_que"),
        }
    }
}

impl<'d, Sh, T> BufferFromSlice<'d, Buffer<'d, Sh, T>> for Device
where
    T: OclPrm + DType,
    Sh: 'd + Shape,
{
    fn slice(&'d mut self, slice: &[T]) -> Buffer<'d, Sh, T> {
        Buffer::<'d, Sh, T> {
            pro_que: &self.pro_que,
            data: self.pro_que.buffer_builder().copy_host_slice(slice).len(Sh::numel()).flags(ocl::flags::MEM_READ_ONLY).build().expect("Unable to create buffer"),
            shape: PhantomData,
        }
    }
}

#[test]
fn opencl_device() {
    use crate::prelude::*;
    use crate::device::opencl;

    let mut device = opencl::Device::default();

    let x = device.buffer([3, 4, 2]);

    std::println!("{}", x);

    panic!();
}

/// OpenCL buffer
#[derive(Debug)]
pub struct Buffer<'d, Sh, T>
where
    T: OclPrm + DType,
    Sh: Shape,
{
    data: ocl::Buffer<T>,
    pro_que: &'d ocl::ProQue,
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
    fn device(&self) -> Self::Dev {
        Device {
            // TODO is this ok?
            pro_que: self.pro_que.clone(),
        }
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

extern crate alloc;
impl<Sh, T> ops::IntoVec<T> for Buffer<'_, Sh, T>
where
    T: OclPrm + DType + ops::Zero,
    Sh: Shape,
{
    fn to_vec(&self) -> alloc::vec::Vec<T> {
        extern crate alloc;
        let mut res = alloc::vec![T::zero(); Sh::numel()];
        self.data.read(&mut res).enq().expect("Couldn't read buffer");
        res
    }
}
