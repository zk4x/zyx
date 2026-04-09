use crate::{Tensor, cache::KernelId, kernel::Kernel, tensor::TensorId};

/// Custom kernel, for now custom kernels can have only 1 output
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub struct CustomKernel {
    params: Vec<TensorId>,
    kernel: KernelId,
}

impl Kernel {
    pub fn into_tensor_op(self) -> fn(&Tensor) -> Tensor {
        todo!()
    }

    #[allow(unused)]
    pub fn into_backward_op(self) -> fn(&Tensor) -> Tensor {
        todo!()
    }
}
