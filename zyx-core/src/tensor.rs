use crate::backend::Backend;

#[derive(Clone, Copy)]
pub struct Id(usize);

pub struct Tensor<B: Backend> {
    data: Id,
    backend: B,
}

pub fn tensor<B: Backend>(backend: B, data: Id) -> Tensor<B> {
    Tensor {
        backend,
        data,
    }
}

impl<B: Backend> Tensor<B> {
    pub fn exp(&self) -> Tensor<B> {
        Tensor {
            data: self.backend.clone().exp(self.data),
            backend: self.backend.clone(),
        }
    }
}
