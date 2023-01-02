#![no_std]
//#![feature(adt_const_params)]
//#![feature(generic_const_exprs)]

// Nightly lints
//#![feature(strict_provenance)]
//#![warn(fuzzy_provenance_casts)]
//#![warn(lossy_provenance_casts)]

//#![feature(must_not_suspend)]
//#![warn(must_not_suspend)]

//#![warn(rustdoc::missing_doc_code_examples)]

// Standard lints
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::private_intra_doc_links)]
#![warn(rustdoc::missing_crate_level_docs)]
#![warn(rustdoc::private_doc_tests)]
#![warn(rustdoc::invalid_codeblock_attributes)]
#![warn(rustdoc::invalid_html_tags)]
#![warn(rustdoc::invalid_rust_codeblocks)]
#![warn(rustdoc::bare_urls)]
// Clippy lints
#![deny(absolute_paths_not_starting_with_crate)]
#![deny(elided_lifetimes_in_paths)]
#![deny(explicit_outlives_requirements)]
#![deny(keyword_idents)]
#![deny(macro_use_extern_crate)]
#![deny(meta_variable_misuse)]
#![deny(missing_abi)]
#![deny(missing_copy_implementations)]
#![warn(missing_debug_implementations)]
#![deny(non_ascii_idents)]
#![warn(noop_method_call)]
#![deny(pointer_structural_match)]
#![deny(rust_2021_incompatible_closure_captures)]
#![deny(rust_2021_incompatible_or_patterns)]
#![deny(rust_2021_prefixes_incompatible_syntax)]
#![deny(rust_2021_prelude_collisions)]
#![deny(single_use_lifetimes)]
#![deny(trivial_casts)]
#![deny(trivial_numeric_casts)]
#![deny(unreachable_pub)]
//#![warn(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(unused_crate_dependencies)]
#![deny(unused_extern_crates)]
#![deny(unused_import_braces)]
pub mod device;
pub mod nn;
pub mod ops;
pub mod optim;
pub mod prelude;
pub mod shape;
#[deny(unused_lifetimes)]
#[warn(unused_macro_rules)]
#[warn(unused_tuple_struct_fields)]
pub mod tensor;

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests;

// Names for generics
//
// A - Axes
// S - Shape
// T - DType
// D - Device
// B - Buffer
// V - Variable

// TODO:
// saving of models and buffers (probably via .to_vec() and .shape(), and .from_slice())
// convolution ops for tensor
// remove need for alloc crate if user is not using cpu::Buffer,
// simple dataloading methods
// #[derive(Model)] and #[derive(HasParameters)] macros
