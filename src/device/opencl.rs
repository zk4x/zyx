use super::{dtype::DType, Device};

struct OpenCLDevice {
    context: ocl::Context,
    device: ocl::Device, // TODO support multiple devices
}

impl Device for OpenCLDevice {
    type Buffer<T> = OpenCLBuffer<T>
    where
        T: ocl::OclPrm + DType;

    fn slice<T: super::DType + ocl::OclPrm>(&self, slice: &[T]) -> Self::Buffer<T> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct OpenCLBuffer<T = f32>
where
    T: ocl::OclPrm,
{
    data: ocl::Buffer<T>,
}
