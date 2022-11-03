#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::private_intra_doc_links)]
#![warn(rustdoc::missing_crate_level_docs)]
//#![warn(rustdoc::missing_doc_code_examples)]
#![warn(rustdoc::private_doc_tests)]
#![warn(rustdoc::invalid_codeblock_attributes)]
#![warn(rustdoc::invalid_html_tags)]
#![warn(rustdoc::invalid_rust_codeblocks)]
#![warn(rustdoc::bare_urls)]

// clippy lints
#![deny(absolute_paths_not_starting_with_crate)]
#![deny(elided_lifetimes_in_paths)]
#![deny(explicit_outlives_requirements)]
//#![feature(strict_provenance)]
//#![warn(fuzzy_provenance_casts)]
//#![warn(lossy_provenance_casts)]
#![deny(keyword_idents)]
#![deny(macro_use_extern_crate)]
#![deny(meta_variable_misuse)]
#![deny(missing_abi)]
#![deny(missing_copy_implementations)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
//#![feature(must_not_suspend)]
//#![warn(must_not_suspend)]
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
//#![deny(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(unused_crate_dependencies)]
#![deny(unused_extern_crates)]
#![deny(unused_import_braces)]
#[deny(unused_lifetimes)]
#[warn(unused_macro_rules)]
#[warn(unused_tuple_struct_fields)]

pub mod tensor;
pub mod accel;
pub mod ops;
pub mod shape;
pub mod module;
pub mod nn;
pub mod optim;
pub mod prelude;
pub mod init;
mod dtype;

#[cfg(test)]
mod tests;

// TODO: saving of models and buffers (probably via .to_vec() and .shape(), and ::from_vec())
// power and convolution ops for tensor
// opencl buffer
// lazy Buffer (both opencl and cpu (I know, opencl can run on cpu as well))

// release 0.5.0 with changes to how parameters to optimizers are handled
