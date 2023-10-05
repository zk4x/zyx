//! # nn
//!
//! Contains high level constructs, all of which implement Module.
//! All of them have one or more parameters, which means there is no `nn::ReLU`,
//! as `ReLU` does not have any parameters.

mod linear;
pub use linear::Linear;

use crate::{
    parameters::Parameters,
    tensor::Tensor,
};

/// Module
pub trait Module {
    /// Forward pass
    fn forward(&self, x: &Tensor) -> Tensor;
    /// Access all parameters in module
    fn parameters(&mut self) -> Parameters<'_>;
}
