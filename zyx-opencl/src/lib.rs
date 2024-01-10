#![no_std]

mod inner;
mod eval;

extern crate alloc;
use core::cell::RefCell;
use alloc::collections::{BTreeMap, BTreeSet};
use cl3::error_codes::ClError;
use zyx_core::{backend::Backend, node::Node, shape::Shape, tensor::Id};
use zyx_core::backend::BufferView;
use zyx_core::scalar::Scalar;
use zyx_core::tensor::{IntoTensor, tensor};

pub use zyx_core::tensor::Tensor;
pub use zyx_core::dtype::DType;
use zyx_core::common::Autograd;

pub struct OpenCL(RefCell<inner::Inner>);

pub fn default() -> Result<OpenCL, ClError> {
    Ok(OpenCL(RefCell::new(inner::Inner::new()?)))
}

impl OpenCL {
    pub fn tensor<'a>(&'a self, data: impl IntoTensor<&'a Self>) -> Tensor<&'a Self> {
        data.into_tensor(self)
    }

    pub fn randn(&self, shape: impl Into<Shape>, dtype: DType) -> Tensor<&Self> {
        tensor(self.0.borrow_mut().randn(shape.into(), dtype), self)
    }

    pub fn uniform<T: Scalar>(&self, shape: impl Into<Shape>, low: T, high: T) -> Tensor<&Self> {
        tensor(self.0.borrow_mut().uniform(shape.into(), low, high), self)
    }

    pub fn full<T: Scalar>(&self, shape: impl Into<Shape>, value: T) -> Tensor<&Self> {
        tensor(self.0.borrow_mut().full(shape.into(), value), self)
    }

    pub fn zeros(&self, shape: impl Into<Shape>, dtype: DType) -> Tensor<&Self> {
        match dtype {
            DType::F32 => tensor(self.0.borrow_mut().full(shape.into(), 0.), self),
            DType::I32 => tensor(self.0.borrow_mut().full(shape.into(), 0), self),
        }
    }

    pub fn ones(&self, shape: impl Into<Shape>, dtype: DType) -> Tensor<&Self> {
        match dtype {
            DType::F32 => tensor(self.0.borrow_mut().full(shape.into(), 1.), self),
            DType::I32 => tensor(self.0.borrow_mut().full(shape.into(), 1), self),
        }
    }

    pub fn eye(&self, n: usize, dtype: DType) -> Tensor<&Self> {
        tensor(self.0.borrow_mut().eye(n, dtype), self)
    }
}

impl Backend for &OpenCL {
    fn shape(self, x: Id) -> Shape { self.0.borrow().shape(x).clone() }

    fn dtype(self, x: Id) -> DType { self.0.borrow().dtype(x) }

    fn backward(self, x: Id, sources: &BTreeSet<Id>) -> BTreeMap<Id, Id> {
        self.0.borrow_mut().backward(x, sources)
    }

    fn load(self, x: Id) -> BufferView { self.0.borrow_mut().load(x) }

    fn push(self, node: Node) -> Id { self.0.borrow_mut().push(node) }

    fn set_leaf(self, x: Id) { self.0.borrow_mut().set_leaf(x); }

    fn release(self, x: Id) { self.0.borrow_mut().release(x); }

    fn retain(self, x: Id) { self.0.borrow_mut().retain(x); }
}

#[test]
fn t0() ->Result<(), ClError> {
    let dev = crate::default()?;
    let x = dev.randn([2, 3], DType::F32);
    let y = dev.randn([2, 3], DType::F32);
    let z = (&x + &y).exp() + &x;
    let _grads = z.backward([&y]);
    Ok(())
}
