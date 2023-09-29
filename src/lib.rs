#![no_std]
#![doc = include_str!("../README.md")]

#![warn(clippy::pedantic)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]

#![cfg_attr(not(feature = "opencl"), forbid(unsafe_code))]
#![warn(unsafe_op_in_unsafe_fn)]
#![forbid(missing_docs)]
#![forbid(rustdoc::broken_intra_doc_links)]
#![forbid(rustdoc::private_intra_doc_links)]
#![forbid(rustdoc::missing_crate_level_docs)]
#![forbid(rustdoc::private_doc_tests)]
#![forbid(rustdoc::invalid_codeblock_attributes)]
#![forbid(rustdoc::invalid_html_tags)]
#![forbid(rustdoc::invalid_rust_codeblocks)]
#![forbid(rustdoc::bare_urls)]
#![forbid(non_camel_case_types)]
#![forbid(absolute_paths_not_starting_with_crate)]
#![forbid(unused_lifetimes)]
#![deny(single_use_lifetimes)]
#![forbid(elided_lifetimes_in_paths)]
#![forbid(explicit_outlives_requirements)]
#![forbid(keyword_idents)]
#![forbid(macro_use_extern_crate)]
#![forbid(meta_variable_misuse)]
#![forbid(missing_abi)]
#![forbid(missing_copy_implementations)]
#![forbid(missing_debug_implementations)]
#![forbid(non_ascii_idents)]
#![forbid(noop_method_call)]
#![forbid(pointer_structural_match)]
#![forbid(trivial_casts)]
#![forbid(trivial_numeric_casts)]
#![forbid(unreachable_pub)]
#![forbid(unused_crate_dependencies)]
#![forbid(unused_extern_crates)]
#![forbid(unused_import_braces)]
#![forbid(unused_macro_rules)]
#![forbid(unused_tuple_struct_fields)]

// Nightly lints
//#![feature(stmt_expr_attributes)]
//#![feature(strict_provenance)]
//#![forbid(fuzzy_provenance_casts)]
//#![forbid(lossy_provenance_casts)]
//#![feature(must_not_suspend)]
//#![forbid(must_not_suspend)]
//#![feature(rustdoc_missing_doc_code_examples)]
//#![warn(rustdoc::missing_doc_code_examples)]

#[cfg(any(feature = "debug1", feature = "io"))]
extern crate std;

pub mod axes;
pub mod context;
pub mod dtype;
mod graph;
mod node_id;
pub mod nn;
pub mod optim;
pub mod parameters;
pub mod prelude;
pub mod shape;
pub mod tensor;
mod device;

/// # `OutOfMemoryError`
///
/// Returned from realize function when backend requested allocation of too much memory.
/// ```
/// # use zyx::context::Context;
/// let mut ctx = Context::new();
/// let x = ctx.tensor([3., 4., 2.]);
/// let mut y = x.exp();
/// y.realize()?;
/// # Ok::<(), zyx::OutOfMemoryError>(())
/// ```
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutOfMemoryError;
