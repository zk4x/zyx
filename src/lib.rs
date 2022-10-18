#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

pub mod tensor;
pub mod accel;
pub mod ops;
pub mod shape;
pub mod module;
pub mod nn;
pub mod optim;
pub mod prelude;
pub mod init;

#[cfg(test)]
mod tests;

// TODO: saving of models and buffers (probably via .to_vec() and .shape(), and ::from_vec())
// power and convolution ops for tensor
// opencl buffer
// lazy Buffer (both opencl and cpu (I know, opencl can run on cpu as well))

// and then release 0.5.0 with changes to how parameters to optimizers are handled
