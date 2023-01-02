extern crate alloc;

#[derive(Default, Clone)]
pub struct Device {}

impl super::Device for Device {
    type Buffer<T: super::DType + ocl::OclPrm> = CPUBuffer<T>;

    fn slice<T: super::DType>(&self, slice: &[T]) -> Self::Buffer<T> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct CPUBuffer<T>(alloc::vec::Vec<T>);
