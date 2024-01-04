#![no_std]

mod inner;

extern crate alloc;
use core::cell::RefCell;
use alloc::collections::{BTreeMap, BTreeSet};
use cl3::error_codes::ClError;
use zyx_core::{backend::{Backend}, node::Node, shape::Shape, tensor::Id};
use zyx_core::backend::BufferView;
use zyx_core::dtype::DType;
use zyx_core::tensor::{IntoTensor, Tensor, tensor};

pub struct OpenCL(RefCell<inner::Inner>);

impl OpenCL {
    pub fn new() -> Result<OpenCL, ClError> {
        Ok(OpenCL(RefCell::new(inner::Inner::new()?)))
    }

    pub fn tensor<'a>(&'a self, data: impl IntoTensor<&'a Self>) -> Tensor<&'a Self> {
        data.into_tensor(self)
    }

    /// Returns f32 tensor. User is expected to cast if he needs different type.
    pub fn randn(&self, shape: impl Into<Shape>) -> Tensor<&Self> {
        tensor(self.0.borrow_mut().randn(shape.into()), self)
    }

    /// Returns f32 tensor. User is expected to cast if he needs different type.
    pub fn ones(&self, shape: impl Into<Shape>) -> Tensor<&Self> {
        tensor(self.0.borrow_mut().ones(shape.into()), self)
    }

    /// Returns f32 tensor. User is expected to cast if he needs different type.
    pub fn full(&self, value: f32, shape: impl Into<Shape>) -> Tensor<&Self> {
        tensor(self.0.borrow_mut().full(value, shape.into()), self)
    }
}

impl Backend for &OpenCL {
    fn shape(self, x: Id) -> Shape { self.0.borrow().shape(x) }

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
fn t0() {
    let dev = OpenCL::new().unwrap();
    let x = dev.randn([2, 3]);
    let y = dev.randn([2, 3]);
    let z = (&x + &y).exp() + &x;
    let _grads = z.backward([&y]);
}
