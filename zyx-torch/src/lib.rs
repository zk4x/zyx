//! Libtorch backend for zyx
//!
//! zyx-opencl is used as any other Zyx backend.
//!
//! Initialize backend.
//!
//! Create tensors.
//!

#![no_std]
#![forbid(rustdoc::broken_intra_doc_links)]
#![forbid(rustdoc::private_intra_doc_links)]
#![forbid(missing_docs)]
#![forbid(rustdoc::missing_crate_level_docs)]
//#![forbid(rustdoc::missing_doc_code_examples)]
#![forbid(rustdoc::private_doc_tests)]
#![forbid(rustdoc::invalid_codeblock_attributes)]
#![forbid(rustdoc::invalid_html_tags)]
#![forbid(rustdoc::invalid_rust_codeblocks)]
#![forbid(rustdoc::bare_urls)]
#![forbid(rustdoc::unescaped_backticks)]
#![forbid(rustdoc::redundant_explicit_links)]

#[cfg(feature = "std")]
extern crate std;

mod interpreter;
use crate::interpreter::Interpreter;

extern crate alloc;
use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use core::ops::Range;
#[cfg(feature = "std")]
pub use zyx_core::io::{load, save};
use zyx_core::{
    backend::Backend,
    node::Node,
    runtime::Runtime,
    scalar::Scalar,
    shape::Shape,
    tensor::Id,
    tensor::{tensor, IntoTensor},
};
pub use zyx_core::{dtype::DType, error::ZyxError, tensor::Tensor};

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

/// Libtorch backend
pub struct Torch(MCell<Runtime<Interpreter>>);

/// Create new Torch backend
pub fn device() -> Result<Torch, ZyxError> {
    Ok(Torch(MCell::new(Runtime::new(Interpreter::new()))))
}

impl Torch {
    /// Create new tensor
    #[must_use]
    pub fn tensor<'a>(&'a self, data: impl IntoTensor<&'a Self>) -> Tensor<&'a Self> {
        <&Self as Backend>::tensor(self, data)
    }

    /// Create new tensor using values from standard normal distribution
    #[must_use]
    pub fn randn(&self, shape: impl Into<Shape>, dtype: DType) -> Tensor<&Self> {
        <&Self as Backend>::randn(self, shape, dtype)
    }

    /// Create new tensor using values from uniform distribution
    #[must_use]
    pub fn uniform(&self, shape: impl Into<Shape>, range: Range<impl Scalar>) -> Tensor<&Self> {
        <&Self as Backend>::uniform(self, shape, range)
    }

    /// Create new tensor by repeating single value
    #[must_use]
    pub fn full(&self, shape: impl Into<Shape>, value: impl Scalar) -> Tensor<&Self> {
        <&Self as Backend>::full(self, shape, value)
    }

    /// Create new tensor by repeating zeroes
    #[must_use]
    pub fn zeros(&self, shape: impl Into<Shape>, dtype: DType) -> Tensor<&Self> {
        <&Self as Backend>::zeros(self, shape, dtype)
    }

    /// Create new tensor by repeating ones
    #[must_use]
    pub fn ones(&self, shape: impl Into<Shape>, dtype: DType) -> Tensor<&Self> {
        <&Self as Backend>::ones(self, shape, dtype)
    }

    /// Create eye tensor
    #[must_use]
    pub fn eye(&self, n: usize, dtype: DType) -> Tensor<&Self> {
        <&Self as Backend>::eye(self, n, dtype)
    }

    /// Load tensors from disk.
    #[cfg(feature = "std")]
    pub fn load(&self, path: impl AsRef<std::path::Path>) -> Result<Vec<Tensor<&Torch>>, ZyxError> {
        zyx_core::io::load(self, path)
    }

    /// Create graph of operations between tensors in dot format for visualization
    #[must_use]
    pub fn plot_graph<'a, B: Backend + 'a>(&self, tensors: impl IntoIterator<Item = &'a Tensor<B>>) -> alloc::string::String {
        <&Self as Backend>::plot_graph(self, tensors)
    }
}

impl Backend for &Torch {
    fn plot_graph<'a, B: Backend + 'a>(self, tensors: impl IntoIterator<Item = &'a Tensor<B>>) -> alloc::string::String {
        let ids: Vec<Id> = tensors.into_iter().map(|t| t.id()).collect();
        self.0.read(|b| b.plot_graph_dot(&ids))
    }

    fn randn(self, shape: impl Into<Shape>, dtype: DType) -> Tensor<Self> {
        tensor(self.0.update(|b| b.randn(shape.into(), dtype)), self)
    }

    fn uniform(self, shape: impl Into<Shape>, range: Range<impl Scalar>) -> Tensor<Self> {
        tensor(self.0.update(|b| b.uniform(shape.into(), range)), self)
    }

    fn shape(self, x: Id) -> Shape {
        self.0.read(|b| b.shape(x).clone())
    }

    fn dtype(self, x: Id) -> DType {
        self.0.read(|b| b.dtype(x))
    }

    fn backward(self, x: Id, sources: &BTreeSet<Id>) -> Result<BTreeMap<Id, Id>, ZyxError> {
        self.0.update(|b| b.backward(x, sources))
    }

    fn load<T: Scalar>(self, x: Id) -> Result<Vec<T>, ZyxError> {
        self.0.update(|b| b.load(x))
    }

    fn push(self, node: Node) -> Result<Id, ZyxError> {
        self.0.update(|b| b.push(node))
    }

    fn set_leaf(self, x: Id) {
        self.0.update(|b| b.set_leaf(x));
    }

    fn release(self, x: Id) -> Result<(), ZyxError> {
        self.0.update(|b| b.release(x))
    }

    fn retain(self, x: Id) {
        self.0.update(|b| b.retain(x));
    }
}

/*#[test]
fn t0() -> Result<(), ZyxError> {
    let dev = crate::device()?;
    //let x = dev.tensor([[3, 2, 4], [4, 2, 3]]);
    //crate::save([&x], "../x.safetensors")?;
    let x = crate::load(&dev, "../x.safetensors")?.next().unwrap();
    std::println!("{x}");
    Ok(())
}*/

/*#[test]
fn t0() -> Result<(), ZyxError> {
    let dev = device()?;
    let x = dev.randn([2, 3], DType::F32);
    let y = dev.randn([2, 3], DType::F32);
    let z = (&x + &y).exp() + &x;
    let _grads = z.backward([&y]);
    Ok(())
}*/

/*#[test]
fn test_layer_norm() -> Result<(), ZyxError> {
    let dev = device()?;
    let x = dev.randn([2, 3], DType::F32);
    let _n = x.shape()[-1];

    //let z = (x - (x.sum(-1)/n).expand())/(((x - (x.sum(-1)/n).expand()).sum(-1)/n + 0.00001.expand()).sqrt()).expand();

    //let x = x.dot(w);
    //let x = a * (x - x.mean(-1))/(x.var(-1) + 0.00001).sqrt() + b;
    //let x = x.tanh();
    //let x = x.dropout(0.3);

    Ok(())
}*/
