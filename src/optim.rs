//! # Optimizers

use crate::{prelude::IntoParameters, OutOfMemoryError};

/// # SGD optimizer
///
/// Implements stochastic gradient descent.
/// <center>x.data = x.data - x.grad * optim.learning_rate</center>
#[derive(Debug, Copy, Clone)]
pub struct SGD {
    learning_rate: f32,
}

impl Default for SGD {
    fn default() -> Self {
        Self::new()
    }
}

impl SGD {
    /// Create new SGD optimizer
    /// ```
    /// use zyx::optim::SGD;
    /// let optim = SGD::new().set_lr(0.01);
    /// ```
    #[must_use]
    pub fn new() -> SGD {
        Self {
            learning_rate: 0.01,
        }
    }

    /// Set learning rate for this optimizer
    #[allow(clippy::return_self_not_must_use)]
    pub fn set_lr(&mut self, learning_rate: f32) -> SGD {
        Self { learning_rate }
    }

    /// Step through parameters using this optimzier
    /// # Errors
    /// Returns [`OutOfMemoryError`] if backend failed to allocate
    /// necessary memory for result and intermediate tensors.
    pub fn step<'p>(
        &'p mut self,
        parameters: impl IntoParameters<'p>,
    ) -> Result<(), OutOfMemoryError> {
        let mut parameters = parameters.into_parameters().into_vec();
        for param in &mut parameters {
            if let Some(grad) = param.grad() {
                param.set_data(param.data() + grad * self.learning_rate);
                param.zero_grad();
            }
        }
        parameters.realize()
    }
}
