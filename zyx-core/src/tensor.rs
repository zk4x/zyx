extern crate alloc;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ops::SubAssign;
use crate::{backend::Backend, node::Node};
use crate::dtype::DType;
use crate::shape::Shape;

#[derive(Clone, Copy, PartialOrd, PartialEq, Ord, Eq, Debug)]
pub struct Id(usize);

pub fn id(id: usize) -> Id {
    Id(id)
}

impl Id {
    pub fn i(self) -> usize {
        self.0
    }
}

impl SubAssign<usize> for Id {
    fn sub_assign(&mut self, rhs: usize) {
        self.0 -= rhs;
    }
}

pub struct Tensor<B: Backend> {
    id: Id,
    backend: B,
}

impl<B: Backend> Clone for Tensor<B> {
    fn clone(&self) -> Self {
        self.backend.retain(self.id);
        tensor(self.id, self.backend)
    }
}

impl<B: Backend> Drop for Tensor<B> {
    fn drop(&mut self) {
        self.backend.release(self.id);
    }
}

pub fn tensor<B: Backend>(id: Id, backend: B) -> Tensor<B> {
    Tensor {
        id,
        backend,
    }
}

impl<B: Backend> Tensor<B> {
    #[must_use]
    pub fn shape(&self) -> Shape {
        self.backend.shape(self.id)
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.backend.dtype(self.id)
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    #[must_use]
    pub fn backend(&self) -> B {
        self.backend
    }

    #[must_use]
    pub fn backward<'a>(&'a self, sources: impl IntoIterator<Item = &'a Tensor<B>>) -> impl Iterator<Item = Option<Tensor<B>>> + 'a
    where
        B: 'a
    {
        let sources: Vec<&Tensor<B>> = sources.into_iter().collect();
        let grads = self.backend.backward(self.id, &sources.iter().map(|t| t.id).collect());
        sources.into_iter().map(move |x: &Tensor<B>| grads.get(&x.id).cloned()).map(move |x| x.map(|x| tensor(x, self.backend)))
    }

    #[must_use]
    pub fn exp(&self) -> Tensor<B> {
        tensor(self.backend.push(Node::Exp(self.id)), self.backend)
    }

    /// Tensors should be treated as immutable data structures.
    /// However optimizers need to update model's parameters with new values.
    /// This is the only function that mutates tensor.
    pub fn set(&mut self, x: impl IntoTensor<B>) {
        self.backend.release(self.id);
        let x = x.into_tensor(self.backend);
        assert_eq!(self.shape(), x.shape());
        assert_eq!(self.dtype(), x.dtype());
        self.backend.retain(x.id);
        self.id = x.id;
        self.backend.set_leaf(self.id);
    }
}

pub trait IntoTensor<B: Backend> {
    fn into_tensor(self, backend: B) -> Tensor<B>;
}

impl<B: Backend> IntoTensor<B> for Tensor<B> {
    fn into_tensor(self, _backend: B) -> Tensor<B> {
        // TODO assert self.backend == backend
        self
    }
}

impl<B: Backend> IntoTensor<B> for &Tensor<B> {
    fn into_tensor(self, _backend: B) -> Tensor<B> {
        // TODO assert self.backend == backend
        self.clone()
    }
}

impl<B: Backend, const D0: usize> IntoTensor<B> for [f32; D0] {
    fn into_tensor(self, backend: B) -> Tensor<B> {
        tensor(backend.push(Node::IterF32(Box::new(self.into_iter()), D0.into())), backend)
    }
}

impl<B: Backend, IT: IntoTensor<B>> core::ops::Add<IT> for &Tensor<B> {
    type Output = Tensor<B>;
    fn add(self, rhs: IT) -> Self::Output {
        let rhs = rhs.into_tensor(self.backend);
        tensor(self.backend.push(Node::Add(self.id, rhs.id)), self.backend)
    }
}

impl<B: Backend, IT: IntoTensor<B>> core::ops::Add<IT> for Tensor<B> {
    type Output = Tensor<B>;
    fn add(self, rhs: IT) -> Self::Output {
        let rhs = rhs.into_tensor(self.backend);
        tensor(self.backend.push(Node::Add(self.id, rhs.id)), self.backend)
    }
}
