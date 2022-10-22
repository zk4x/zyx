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
