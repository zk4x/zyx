//! OpenCL backend for zyx
//!
//! Initialize backend. You can use builder if you want to change settings of the backend.
//! ```rust
//! let dev = zyx_opencl::device()?;
//! let dev = zyx_opencl::device_builder().build()?;
//! # Ok::<(), zyx_opencl::ZyxError>(())
//! ```
//!
//! For README, quick tutorial and source code, please visit [https://www.github.com/zk4x/zyx].
//!
//! For more details, there is a [book](https://www.github.com/zk4x/zyx/tree/main/zyx-book).

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

#[cfg(any(feature = "debug1", feature = "std"))]
extern crate std;

mod compiler;

extern crate alloc;
use crate::compiler::Compiler;
use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use core::cell::RefCell;
use core::ops::Range;
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

/// OpenCL backend
pub struct OpenCL(RefCell<Runtime<zyx_compiler::CompiledBackend<Compiler>>>);

/// Create new OpenCL backend using first OpenCL platform
/// and all hardware devices in that platform.
pub fn device() -> Result<OpenCL, ZyxError> {
    Ok(OpenCL(RefCell::new(Runtime::new(
        zyx_compiler::CompiledBackend::new(Compiler::new(0, 8)?),
    ))))
}

/// Create new OpenCL backend using builder.
pub fn device_builder() -> OpenCLBuilder {
    OpenCLBuilder {
        platform_id: 0,
        queues_per_device: 8,
    }
}

/// OpenCL backend builder
#[derive(Clone, Debug)]
pub struct OpenCLBuilder {
    platform_id: usize,
    queues_per_device: usize,
}

impl OpenCLBuilder {
    /// Choose OpenCL platform by id
    pub fn platform_id(&mut self, platform_id: usize) -> Self {
        self.platform_id = platform_id;
        self.clone()
    }

    /// Choose number of queues per each platform device
    pub fn queues_per_device(&mut self, queues_per_device: usize) -> Self {
        self.queues_per_device = queues_per_device;
        self.clone()
    }

    /// Build
    pub fn build(self) -> Result<OpenCL, ZyxError> {
        Ok(OpenCL(RefCell::new(Runtime::new(
            zyx_compiler::CompiledBackend::new(Compiler::new(
                self.platform_id,
                self.queues_per_device,
            )?),
        ))))
    }
}

impl OpenCL {
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
    pub fn load(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Vec<Tensor<&OpenCL>>, ZyxError> {
        zyx_core::io::load(self, path)
    }

    /// Create graph of operations between tensors in dot format for visualization
    #[must_use]
    pub fn plot_graph<'a, B: Backend + 'a>(
        &self,
        tensors: impl IntoIterator<Item = &'a Tensor<B>>,
    ) -> alloc::string::String {
        <&Self as Backend>::plot_graph(self, tensors)
    }

    #[allow(dead_code)]
    fn debug_graph(&self) {
        self.0.borrow_mut().debug_graph();
    }
}

impl Backend for &OpenCL {
    fn plot_graph<'a, B: Backend + 'a>(
        self,
        tensors: impl IntoIterator<Item = &'a Tensor<B>>,
    ) -> alloc::string::String {
        let ids: Vec<Id> = tensors.into_iter().map(|t| t.id()).collect();
        self.0.borrow().plot_graph_dot(&ids)
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
        IT: IntoIterator<Item = T>,
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
    let dev = device()?;
    let x = dev.randn([2, 3], DType::F32);
    let y = dev.randn([2, 3], DType::F32);
    let z = (&x + &y).exp() + &x;
    let _grads = z.backward([&y]);
    let y_grad = _grads.into_iter().next().unwrap().unwrap();
    //std::fs::write("graph.dot", dev.plot_graph([&y, &z])).unwrap();
    //y_grad.to_vec::<f32>()?;
    //panic!();
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
fn t0() -> Result<(), ZyxError> {
    let dev = device()?;
    /*let x = dev.tensor(0..150).reshape([10, 15]);
    let x = x.transpose().reshape([10, 15]);

    let x = dev.tensor(0..150).reshape([10, 15]).transpose();
    let x = x.reshape([15, 1, 10]).expand([15, 2, 10]);*/

    let x = dev.tensor([2, 3]);
    let y = &x + &x;
    let g = y.backward([&x]).pop().unwrap().unwrap();
    assert_eq!(g, [2, 2]);
    let z = y.detach();
    let g = z.backward([&z]).pop().unwrap().unwrap();
    assert_eq!(g, [1, 1]);
    let g = z.backward([&x]).pop().unwrap();
    assert_eq!(g, None);

    //panic!();
    Ok(())
}*/

#[test]
fn t0() -> Result<(), ZyxError> {
    let dev = device()?;
    let x = dev.uniform([2, 1, 1], 0f32..1f32);
    let z = &x + x.exp();
    //let x = x.expand([2, 1, 5]);
    //let z = x.expand([2, 3, 5]) + y.reshape([2, 1, 1]);
    std::println!("{z}");
    panic!();
    Ok(())
}

/*#[test]
fn t5() -> Result<(), ZyxError> {
    let dev = device()?;
    //let mut x = dev.randn([1024, 1024], DType::F32);
    let mut x = dev.tensor(3);
    //let y = dev.randn([1024, 1024], DType::F32);
    /*let x = x + 1;
    let x = x + 1;
    let x = x + 1;
    let x = x + 1;*/
    for _ in 0..4 {
        x = x + 1;
    }

    //std::println!("{x}");
    //panic!();
    Ok(())
}*/

/*#[test]
fn t0() {
    let n = 6;
    let dev = device_builder().platform_id(0).build().unwrap();
    let x = dev.tensor(0..(n*n) as i32).reshape([n, n]);
    //std::println!("{x}");
    let z = x.dot(&x);
    let _: Vec<i32> = z.to_vec().unwrap();
    std::println!("{z}");
    panic!();
}

#[test]
fn dot_test() -> Result<(), ZyxError> {
    let dev = device_builder().platform_id(1).build()?;
    let n = 1024;
    let x = dev.randn([n, n], DType::F32);
    //let y = dev.randn([n, n], DType::F32);
    let z = (x.reshape([1, n, 1, n]) * x.reshape([1, 1, n, n])).sum(-1);
    //let z = x.dot(&y).tanh() + x;
    let _: Vec<f32> = z.to_vec()?;
    panic!();
    Ok(())
}*/

/*#[test]
fn dot_test2() -> Result<(), ZyxError> {
    let dev = device_builder().platform_id(0).build()?;
    let mut x = dev.randn([1024, 1024], DType::F32);
    //let begin = std::time::Instant::now();
    for _ in 0..10 {
        x = x.dot(&x);
    }
    let _ = x.to_vec::<f32>();
    //let elapsed = begin.elapsed().as_millis();
    //std::println!("{elapsed}ms");
    //panic!();
    Ok(())
}*/

/*#[test]
fn t6() {
    let dev = device_builder().platform_id(0).build().unwrap();
    let x = dev.randn([1024, 1024], DType::F32);
    let z = x.sum(..);
    let _: Vec<f32> = z.to_vec().unwrap();
    //panic!()
}*/
