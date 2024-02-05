#![no_std]
//! # zyx-core
//!
//! zyx-core is core part of zyx machine learning library.
//! zyx-core contains definitions and functions for tensor, backend, generic
//! runtime with autograd implementation, dtype, shape, scalar, axes, view
//! and generic compiler for backends like opencl, cuda and wgpu.
//!
#![forbid(unsafe_code)]
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

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

/// See [Axes](axes::Axes)
pub mod axes;
/// See [Backend](backend::Backend)
pub mod backend;
/// See [Compiler](compiler::Compiler)
pub mod compiler;
/// See [DType](dtype::DType)
pub mod dtype;
/// See [ZyxError](error::ZyxError)
pub mod error;
/// Saving and loading of tensors from disk
#[cfg(feature = "std")]
pub mod io;
/// See [Node](node::Node)
pub mod node;
/// See [Runtime](runtime::Runtime)
pub mod runtime;
/// See [Scalar](scalar::Scalar)
pub mod scalar;
/// See [Shape](shape::Shape)
pub mod shape;
/// See [Tensor](tensor::Tensor)
pub mod tensor;
/// Some common utilities.
pub mod utils;
/// See [View](view::View)
pub mod view;
