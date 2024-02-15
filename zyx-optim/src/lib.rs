#![no_std]

extern crate alloc;

mod sgd;
pub use sgd::SGD;

mod adam;
pub use adam::Adam;
