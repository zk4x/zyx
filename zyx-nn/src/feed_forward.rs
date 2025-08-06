use crate::{Activation, Linear};
use zyx::{Tensor, ZyxError};
use zyx_derive::Module;

/// FeedForward module implementing a position-wise feed-forward network,
/// similar to `torchtune.modules.FeedForward` in PyTorch.
///
/// It consists of a gate projection (`gate_proj`), an optional up projection (`up_proj`),
/// a down projection (`down_proj`), an activation function, and dropout applied after the up projection.
#[derive(Debug, Module)]
pub struct FeedForward {
    /// - `gate_proj`: Linear layer projecting input to hidden dimension for gating.
    pub gate_proj: Linear,
    /// - `down_proj`: Linear layer projecting from hidden dimension back to output dimension.
    pub down_proj: Linear,
    /// - `up_proj`: Optional Linear layer applied after activation (commonly used in SwiGLU or similar).
    pub up_proj: Option<Linear>,
    /// - `activation`: Activation function applied after gate projection.
    pub activation: Activation,
    /// - `dropout_prob`: Dropout probability applied after the up projection.
    pub dropout_prob: f32,
}

impl FeedForward {
    /// Creates a new `FeedForward` module.
    ///
    /// # Arguments
    ///
    /// * `gate_proj` - Linear layer performing the gate projection.
    /// * `down_proj` - Linear layer projecting back to output dimension.
    /// * `up_proj` - Optional Linear layer applied after activation.
    /// * `activation` - Activation function to apply after the gate projection.
    /// * `dropout_prob` - Dropout probability applied after the up projection (0.0 means no dropout).
    ///
    /// # Returns
    ///
    /// A new instance of `FeedForward`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let gate_proj = Linear::new(input_dim, hidden_dim, true, dtype)?;
    /// let down_proj = Linear::new(hidden_dim, output_dim, true, dtype)?;
    /// let up_proj = Some(Linear::new(input_dim, hidden_dim, true, dtype)?);
    /// let ff = FeedForward::new(gate_proj, down_proj, up_proj, Activation::Gelu, 0.1);
    /// ```
    pub fn new(
        gate_proj: Linear,
        down_proj: Linear,
        up_proj: Option<Linear>,
        activation: Activation,
        dropout_prob: f32,
    ) -> Self {
        Self {
            gate_proj,
            down_proj,
            up_proj,
            activation,
            dropout_prob,
        }
    }

    /// Runs the forward pass of the feed-forward network.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[batch_size, seq_len, input_dim]`.
    ///
    /// # Returns
    ///
    /// A `Result` wrapping the output tensor of shape `[batch_size, seq_len, output_dim]`,
    /// or a `ZyxError` if any layer fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let output = ff.forward(input_tensor)?;
    /// ```
    pub fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let gate = self.gate_proj.forward(x)?;
        let activated = self.activation.forward(&gate);

        let up = if let Some(up_layer) = &self.up_proj {
            up_layer.forward(&activated)?
        } else {
            activated
        };

        let dropped = up.dropout(self.dropout_prob);

        let output = self.down_proj.forward(&dropped)?;

        Ok(output)
    }
}
