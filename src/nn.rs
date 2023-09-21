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
    /*
    /// Zero gradients of all parameters in module
    fn zero_grad(&mut self) {
        self.parameters().zero_grad();
    }
    /// Realize all parameters in module
    /// # Errors
    /// Returns [`OutOfMemoryError`] if backend failed to allocate
    /// necessary memory for result and intermediate tensors.
    fn realize(&mut self) -> Result<(), OutOfMemoryError> {
        self.parameters().realize()
    }
    /// Realize all gradients of parameters in module
    /// # Errors
    /// Returns [`OutOfMemoryError`] if backend failed to allocate
    /// necessary memory for result and intermediate tensors.
    fn realize_grad(&mut self) -> Result<(), OutOfMemoryError> {
        self.parameters().realize_grads()
    }
    /// Get number of all scalars in all parameters
    fn numel(&mut self) -> usize {
        self.parameters().numel()
    }
    /// Save all module parameters to file
    /// # Errors
    /// Returns io erorr if there was problem writing file to filesystem.
    #[cfg(feature = "io")]
    fn save(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), std::io::Error> {
        self.parameters().save(path)
    }
    /// Load all module parameters from file
    /// # Errors
    /// Returns io erorr if there was problem writing file to filesystem.
    #[cfg(feature = "io")]
    fn load(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), std::io::Error> {
        self.parameters().load(path)
    }*/
}
