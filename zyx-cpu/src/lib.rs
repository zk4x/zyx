//! CPU only, pure rust backend for zyx
//!
//! Initialize backend.
//! ```rust
//! let dev = zyx_cpu::device()?;
//! # Ok::<(), zyx_cpu::ZyxError>(())
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
use core::ops::Range;
use std::cell::RefCell;
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

/// CPU backend
pub struct CPU(RefCell<Runtime<Interpreter>>);

/// Create new CPU backend
pub fn device() -> Result<CPU, ZyxError> {
    Ok(CPU(RefCell::new(Runtime::new(Interpreter::new()))))
}

impl Backend for &CPU {
    fn plot_graph<'a>(
        self,
        tensors: impl IntoIterator<Item = &'a Tensor<Self>>,
    ) -> alloc::string::String
    where
        Self: 'a,
    {
        let ids: Vec<Id> = tensors.into_iter().map(|t| t.id()).collect();
        self.0.borrow().plot_graph_dot(&ids)
    }

    fn uniform<T: Scalar>(self, shape: impl Into<Shape>, range: Range<T>) -> Result<Tensor<Self>, ZyxError> {
        todo!()
    }

    fn randn(self, shape: impl Into<Shape>, dtype: DType) -> Result<Tensor<Self>, ZyxError> {
        todo!()
    }

    fn store<T: Scalar, IT>(self, iter: IT) -> Result<Tensor<Self>, ZyxError> where IT: IntoIterator<Item=T>, IT::IntoIter: ExactSizeIterator {
        Ok(tensor(self.0.borrow_mut().store(iter)?, self))
    }

    fn realize(self, tensors: BTreeSet<Id>) -> Result<(), ZyxError> {
        self.0.borrow_mut().realize(tensors)
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

    fn push(self, node: Node) -> Result<Tensor<Self>, ZyxError> {
        Ok(tensor(self.0.borrow_mut().push(node)?, self))
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
fn t0() -> Result<(), ZyxError> {
    let dev = device()?;
    let Q = dev.tensor([[0.0, 0.75], [0.0, 0.0]]);
    let E = dev.eye(2, DType::F32);
    let p = dev.tensor([30000., 20000.]);
    let d = 0.909090909090909090;
    let R = dev.tensor([[0.0, 0.25], [0.8, 0.2]]);
    let n = dev.tensor([0.0, 100.0]);
    let inv = dev.tensor([[1.0, 0.681818], [0.0, 1.0]]);

    let y = p.dot(inv).dot(R).dot(n);

    std::println!("{y}");
    //std::println!("{}", n.transpose());
    panic!();
    Ok(())
}*/

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
    for i in 0..1000000 {
        if i % 100000 == 0 {
            std::println!("i: {i}");
        }
        x = x + 1;
    }

    std::println!("{x}");
    panic!();
    Ok(())
}*/
