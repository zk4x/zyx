#![no_std]

#[cfg(feature = "debug1")]
extern crate std;

mod compiler;

extern crate alloc;
use alloc::{collections::{BTreeMap, BTreeSet}, vec::Vec};
use core::ops::Range;
use cl3::error_codes::ClError;
use zyx_core::{
    scalar::Scalar,
    tensor::{tensor, IntoTensor},
    backend::Backend, node::Node, shape::Shape, tensor::Id,
    runtime::Runtime,
};
pub use zyx_core::{dtype::DType, tensor::Tensor, error::ZyxError};
use zyx_core::compiler::CompiledBackend;
use crate::compiler::Compiler;

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
    const fn new(inner: T) -> MCell<T> {
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

pub struct OpenCL(MCell<Runtime<CompiledBackend<Compiler>>>);

pub fn device() -> Result<OpenCL, ZyxError<ClError>> {
    Ok(OpenCL(MCell::new(Runtime::new(
        CompiledBackend::new(Compiler::new().map_err(|err| ZyxError::BackendError(err))?),
    ))))
}

impl OpenCL {
    #[must_use]
    pub fn tensor<'a>(&'a self, data: impl IntoTensor<&'a Self>) -> Tensor<&'a Self> {
        data.into_tensor(self)
    }

    #[must_use]
    pub fn randn(&self, shape: impl Into<Shape>, dtype: DType) -> Tensor<&Self> {
        tensor(self.0.update(|b| b.randn(shape.into(), dtype)), self)
    }

    #[must_use]
    pub fn uniform(&self, shape: impl Into<Shape>, range: Range<impl Scalar>) -> Tensor<&Self> {
        tensor(self.0.update(|b| b.uniform(shape.into(), range.start, range.end)), self)
    }

    #[must_use]
    pub fn full(&self, shape: impl Into<Shape>, value: impl Scalar) -> Tensor<&Self> {
        tensor(self.0.update(|b| b.full(shape.into(), value)), self)
    }

    #[must_use]
    pub fn zeros(&self, shape: impl Into<Shape>, dtype: DType) -> Tensor<&Self> {
        match dtype {
            DType::F32 => tensor(self.0.update(|b| b.full(shape.into(), 0.)), self),
            DType::I32 => tensor(self.0.update(|b| b.full(shape.into(), 0)), self),
        }
    }

    #[must_use]
    pub fn ones(&self, shape: impl Into<Shape>, dtype: DType) -> Tensor<&Self> {
        match dtype {
            DType::F32 => tensor(self.0.update(|b| b.full(shape.into(), 1.)), self),
            DType::I32 => tensor(self.0.update(|b| b.full(shape.into(), 1)), self),
        }
    }

    #[must_use]
    pub fn eye(&self, n: usize, dtype: DType) -> Tensor<&Self> {
        tensor(self.0.update(|b| b.eye(n, dtype)), self)
    }
}

impl Backend for &OpenCL {
    type Error = ClError;
    
    fn _uniform(self, shape: Shape, dtype: DType) -> Id {
        self.0.update(|b| b._uniform(shape.into(), dtype))
    }

    fn shape(self, x: Id) -> Shape {
        self.0.read(|b| b.shape(x).clone())
    }

    fn dtype(self, x: Id) -> DType {
        self.0.read(|b| b.dtype(x))
    }

    fn backward(self, x: Id, sources: &BTreeSet<Id>) -> Result<BTreeMap<Id, Id>, Self::Error> {
        self.0.update(|b| b.backward(x, sources))
    }

    fn load<T: Scalar>(self, x: Id) -> Result<Vec<T>, Self::Error> {
        self.0.update(|b| b.load(x))
    }

    fn push(self, node: Node) -> Result<Id, Self::Error> {
        self.0.update(|b| b.push(node))
    }

    fn set_leaf(self, x: Id) {
        self.0.update(|b| b.set_leaf(x));
    }

    fn release(self, x: Id) -> Result<(), Self::Error> {
        self.0.update(|b| { b.release(x) })
    }

    fn retain(self, x: Id) {
        self.0.update(|b| b.retain(x));
    }
}

#[test]
fn t0() -> Result<(), ZyxError<ClError>> {
    let dev = device()?;
    let x = dev.randn([2, 3], DType::F32);
    let y = dev.randn([2, 3], DType::F32);
    let z = (&x + &y).exp() + &x;
    let _grads = z.backward([&y]);
    Ok(())
}

#[test]
fn test_layer_norm() -> Result<(), ZyxError<ClError>> {
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
