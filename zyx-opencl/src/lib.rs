#![no_std]

#[cfg(feature = "debug1")]
extern crate std;

mod runtime;

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use cl3::error_codes::ClError;
use zyx_core::scalar::Scalar;
use zyx_core::tensor::{tensor, IntoTensor};
use zyx_core::{backend::Backend, node::Node, shape::Shape, tensor::Id};
use zyx_core::autograd::Autograd;
use zyx_core::compiled_backend::CompiledBackend;
pub use zyx_core::dtype::DType;
pub use zyx_core::tensor::Tensor;

// This works OK, it gets rid of the RefCell overhead,
// but it's only safe if used in single threaded environment.
// Can moving things around in memory invalidate these adresses?
// In that case there could be memory unsafety.
// But it should not be possible to move inner while reading or updating,
// is it possible?.
struct MCell<T> {
    inner: core::cell::UnsafeCell<T>,
}

impl<T> MCell<T> {
    fn new(inner: T) -> MCell<T> {
        MCell {
            inner: core::cell::UnsafeCell::new(inner),
        }
    }

    fn read<R>(&self, func: impl FnOnce(&T) -> R) -> R {
        func(unsafe { &*self.inner.get() })
    }

    fn update<R>(&self, func: impl FnOnce(&mut T) -> R) -> R {
        func(unsafe { &mut *self.inner.get() })
    }
}

pub struct OpenCL(MCell<CompiledBackend<runtime::Runtime>>);

pub fn device() -> Result<OpenCL, ClError> {
    Ok(OpenCL(MCell::new(CompiledBackend::new(
        runtime::Runtime::new()?,
    ))))
}

impl OpenCL {
    pub fn tensor<'a>(&'a self, data: impl IntoTensor<&'a Self>) -> Tensor<&'a Self> {
        data.into_tensor(self)
    }

    pub fn randn(&self, shape: impl Into<Shape>, dtype: DType) -> Tensor<&Self> {
        tensor(self.0.update(|mut b| b.randn(shape.into(), dtype)), self)
    }

    pub fn uniform<T: Scalar>(&self, shape: impl Into<Shape>, low: T, high: T) -> Tensor<&Self> {
        tensor(self.0.update(|mut b| b.uniform(shape.into(), low, high)), self)
    }

    pub fn full<T: Scalar>(&self, shape: impl Into<Shape>, value: T) -> Tensor<&Self> {
        tensor(self.0.update(|mut b| b.full(shape.into(), value)), self)
    }

    pub fn zeros(&self, shape: impl Into<Shape>, dtype: DType) -> Tensor<&Self> {
        match dtype {
            DType::F32 => tensor(self.0.update(|mut b| b.full(shape.into(), 0.)), self),
            DType::I32 => tensor(self.0.update(|mut b| b.full(shape.into(), 0)), self),
        }
    }

    pub fn ones(&self, shape: impl Into<Shape>, dtype: DType) -> Tensor<&Self> {
        match dtype {
            DType::F32 => tensor(self.0.update(|mut b| b.full(shape.into(), 1.)), self),
            DType::I32 => tensor(self.0.update(|mut b| b.full(shape.into(), 1)), self),
        }
    }

    pub fn eye(&self, n: usize, dtype: DType) -> Tensor<&Self> {
        tensor(self.0.update(|mut b| b.eye(n, dtype)), self)
    }
}

impl Backend for &OpenCL {
    fn shape(self, x: Id) -> Shape {
        self.0.read(|b| b.shape(x).clone())
    }

    fn dtype(self, x: Id) -> DType {
        self.0.read(|b| b.dtype(x))
    }

    fn backward(self, x: Id, sources: &BTreeSet<Id>) -> BTreeMap<Id, Id> {
        self.0.update(|mut b| b.backward(x, sources))
    }

    fn load<T: Scalar>(self, x: Id) -> Vec<T> {
        self.0.update(|mut b| b.load(x))
    }

    fn push(self, node: Node) -> Id {
        self.0.update(|mut b| b.push(node))
    }

    fn set_leaf(self, x: Id) {
        self.0.update(|mut b| b.set_leaf(x));
    }

    fn release(self, x: Id) {
        self.0.update(|mut b| { b.release(x) });
    }

    fn retain(self, x: Id) {
        self.0.update(|mut b| b.retain(x));
    }
}

#[test]
fn t0() -> Result<(), ClError> {
    let dev = device()?;
    let x = dev.randn([2, 3], DType::F32);
    let y = dev.randn([2, 3], DType::F32);
    let z = (&x + &y).exp() + &x;
    let _grads = z.backward([&y]);
    Ok(())
}

#[test]
fn test_layer_norm() -> Result<(), ClError> {
    let dev = device()?;
    let x = dev.randn([2, 3], DType::F32);
    let _n = x.shape()[-1];

    //let z = (x - (x.sum(-1)/n).expand())/(((x - (x.sum(-1)/n).expand()).sum(-1)/n + 0.00001.expand()).sqrt()).expand();

    //let x = x.dot(w);
    //let x = a * (x - x.mean(-1))/(x.var(-1) + 0.00001).sqrt() + b;
    //let x = x.tanh();
    //let x = x.dropout(0.3);

    Ok(())
}
