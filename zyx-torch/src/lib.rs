//! Libtorch backend for zyx
//!
//! Initialize backend.
//! ```rust
//! let dev = zyx_torch::device()?;
//! # Ok::<(), zyx_torch::ZyxError>(())
//! ```
//!
//! For README, quick tutorial and source code, please visit [https://www.github.com/zk4x/zyx].
//!
//! For more details, there is a [book](https://www.github.com/zk4x/zyx/tree/main/zyx-book).

#![no_std]
//#![forbid(unsafe_code)]
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
use core::{ops::Range, cell::RefCell};
#[cfg(feature = "std")]
pub use zyx_core::io::save;
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
/// Libtorch backend
pub struct Torch(RefCell<Runtime<Interpreter>>);

/// Create new Torch backend
pub fn device() -> Result<Torch, ZyxError> {
    Ok(Torch(RefCell::new(Runtime::new(Interpreter::new()))))
}

impl Torch {
    /// Create new tensor
    #[must_use]
    pub fn tensor<'a>(&'a self, data: impl IntoTensor<&'a Self>) -> Tensor<&'a Self> {
        <&Self as Backend>::tensor(self, data).unwrap()
    }

    /// Create new tensor using values from standard normal distribution
    #[must_use]
    pub fn randn(&self, shape: impl Into<Shape>, dtype: DType) -> Tensor<&Self> {
        <&Self as Backend>::randn(self, shape, dtype).unwrap()
    }

    /// Create new tensor using values from uniform distribution
    #[must_use]
    pub fn uniform(&self, shape: impl Into<Shape>, range: Range<impl Scalar>) -> Tensor<&Self> {
        <&Self as Backend>::uniform(self, shape, range).unwrap()
    }

    /// Create new tensor by repeating single value
    #[must_use]
    pub fn full(&self, shape: impl Into<Shape>, value: impl Scalar) -> Tensor<&Self> {
        <&Self as Backend>::full(self, shape, value).unwrap()
    }

    /// Create new tensor by repeating zeroes
    #[must_use]
    pub fn zeros(&self, shape: impl Into<Shape>, dtype: DType) -> Tensor<&Self> {
        <&Self as Backend>::zeros(self, shape, dtype).unwrap()
    }

    /// Create new tensor by repeating ones
    #[must_use]
    pub fn ones(&self, shape: impl Into<Shape>, dtype: DType) -> Tensor<&Self> {
        <&Self as Backend>::ones(self, shape, dtype).unwrap()
    }

    /// Create eye tensor
    #[must_use]
    pub fn eye(&self, n: usize, dtype: DType) -> Tensor<&Self> {
        <&Self as Backend>::eye(self, n, dtype).unwrap()
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
        self.0.borrow().plot_graph_dot(&ids)
    }

    fn randn(self, shape: impl Into<Shape>, dtype: DType) -> Result<Tensor<Self>, ZyxError> {
        Ok(tensor(self.0.borrow_mut().randn(shape.into(), dtype)?, self))
    }

    fn uniform(self, shape: impl Into<Shape>, range: Range<impl Scalar>) -> Result<Tensor<Self>, ZyxError> {
        Ok(tensor(self.0.borrow_mut().uniform(shape.into(), range)?, self))
    }

    fn shape(self, x: Id) -> Shape {
        self.0.borrow().shape(x).clone()
    }

    fn dtype(self, x: Id) -> DType {
        self.0.borrow().dtype(x)
    }

    fn backward(self, x: Id, sources: &BTreeSet<Id>) -> Result<BTreeMap<Id, Id>, ZyxError> {
        self.0.borrow_mut().backward(x, sources)
    }

    fn load<T: Scalar>(self, x: Id) -> Result<Vec<T>, ZyxError> {
        self.0.borrow_mut().load(x)
    }

    fn store<T: Scalar, IT>(self, iter: IT) -> Result<Id, ZyxError>
    where
        IT: IntoIterator<Item=T>,
        IT::IntoIter: ExactSizeIterator,
    {
        self.0.borrow_mut().store(iter)
    }

    fn push(self, node: Node) -> Result<Id, ZyxError> {
        self.0.borrow_mut().push(node)
    }

    fn release(self, x: Id) -> Result<(), ZyxError> {
        self.0.borrow_mut().release(x)
    }

    fn retain(self, x: Id) {
        self.0.borrow_mut().retain(x);
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

/*#[test]
fn t5() -> Result<(), ZyxError> {
    let dev = crate::device()?;
    let mut x = dev.randn([2, 2], DType::F32);
    //let y = dev.randn([1024, 1024], DType::F32);

    for i in 0..100000000 {
        if i % 1000 == 0 {
            std::println!("Iter: {}", i/1000);
        }
        x = x + 1f32;
        //x = x - 2;
    }

    std::println!("{x}");
    panic!();
    Ok(())
}*/
