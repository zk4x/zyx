use core::ops::RangeBounds;
use crate::device::Device;
use crate::dtype::DType;
use crate::RT;
use crate::scalar::Scalar;
use crate::shape::{IntoAxes, IntoShape};
use alloc::vec::Vec;

pub struct Tensor {
    id: u32,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        RT.lock().release(self.clone());
    }
}

impl Tensor {
    pub(crate) fn new(id: usize) -> Tensor {
        Tensor {
            id: id as u32,
        }
    }

    pub(crate) fn id(&self) -> usize {
        self.id as usize
    }
}

impl Tensor {
    /// Get default device used for new tensors.
    pub fn default_device() -> Device {
        RT.lock().default_device
    }

    /// Set default device used for new tensors.
    /// Returns true if the device initialized successfully.
    /// Returns false if the device failed to initialize.
    pub fn set_default_device(device: Device) -> bool {
        let mut g = RT.lock();
        g.default_device = device;
        g.initialize_device(device)
    }

    /// Tries to initialize all devices and set the first
    /// successfully initialized device as the default_device in this order:
    /// 1. CUDA
    /// 2. OpenCL
    /// 3. WGPU
    /// If they all fail to initialize, then default_device
    /// is set to CPU.
    pub fn set_default_device_best() {
        RT.lock().set_default_device_best();
    }

    pub fn shape(&self) -> Vec<usize> {
        RT.lock().shape(self.clone())
    }

    pub fn dtype(&self) -> DType {
        RT.lock().dtype(self.clone())
    }

    pub fn device(&self) -> Device {
        RT.lock().device(self.clone())
    }

    pub fn to(self, device: Device) -> Tensor {
        todo!()
    }

    pub fn randn(shape: impl IntoShape, dtype: DType) -> Tensor {
        todo!()
    }

    pub fn uniform<T: Scalar>(shape: impl IntoShape, range: impl RangeBounds<T>) -> Tensor {
        todo!()
    }

    pub fn kaiming_uniform<T: Scalar>(shape: impl IntoShape, range: impl RangeBounds<T>) -> Tensor {
        todo!()
    }

    pub fn zeros(shape: impl IntoShape, dtype: DType) -> Tensor {
        todo!()
    }

    pub fn ones(shape: impl IntoShape, dtype: DType) -> Tensor {
        todo!()
    }

    pub fn eye(n: usize, dtype: DType) -> Tensor {
        todo!()
    }

    pub fn exp(&self) -> Tensor {
        RT.lock().exp(self.clone())
    }

    pub fn tanh(&self) -> Tensor {
        RT.lock().tanh(self.clone())
    }

    pub fn cos(&self) -> Tensor {
        RT.lock().cos(self.clone())
    }

    pub fn sin(&self) -> Tensor {
        RT.lock().sin(self.clone())
    }

    pub fn reshape(&self, shape: impl IntoShape) -> Tensor {
        todo!()
    }

    pub fn permute(&self, axes: impl IntoAxes) -> Tensor {
        todo!()
    }

    pub fn sum(&self, axes: impl IntoAxes) -> Tensor {
        todo!()
    }

    pub fn max(&self, axes: impl IntoAxes) -> Tensor {
        todo!()
    }

    pub fn transpose(&self) -> Tensor {
        todo!()
    }
}
