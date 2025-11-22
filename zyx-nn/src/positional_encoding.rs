use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// Sinusoidal positional encoding module for transformers.
///
/// This module adds fixed (non-learnable) positional encodings to input embeddings.
/// It uses the same formulation as in the original "Attention is All You Need" paper,
/// based on sine and cosine functions of different frequencies.
///
/// It supports both `f32` and `f64` types and applies dropout after adding the encodings.
#[derive(Debug, Module)]
pub struct PositionalEncoding {
    /// Precomputed positional encodings of shape `[max_len, d_model]`.
    pe: Tensor,

    /// Dropout probability to apply after adding the positional encoding.
    dropout_prob: f32,
}

impl PositionalEncoding {
    /// Creates a new `PositionalEncoding` module.
    ///
    /// # Arguments
    ///
    /// * `d_model` - The embedding dimension (must match the input's last dimension).
    /// * `max_len` - Maximum sequence length this module will support.
    /// * `dropout_prob` - Dropout probability applied after adding the positional encoding.
    /// * `dtype` - Data type of the encoding (must be `DType::F32` or `DType::F64`).
    ///
    /// # Errors
    ///
    /// Returns a [`ZyxError::ShapeError`] if a non-floating-point dtype is used.
    ///
    /// # Example
    ///
    /// ```
    /// let pe = PositionalEncoding::new(512, 1024, 0.1, DType::F32)?;
    /// ```
    pub fn new(
        d_model: usize,
        max_len: usize,
        dropout_prob: f32,
        dtype: DType,
    ) -> Result<Self, ZyxError> {
        // Enforce floating point type
        match dtype {
            DType::F32 | DType::F64 => {}
            _ => {
                return Err(ZyxError::ShapeError(
                    "PositionalEncoding requires dtype F32 or F64".into(),
                ))
            }
        }

        // position: [max_len, 1]
        let position = Tensor::arange(0i64, max_len as i64, 1i64)?
            .cast(dtype)
            .unsqueeze(1)?;

        // div_term: [d_model // 2]
        let div_term_i64 = Tensor::arange(0i64, d_model as i64, 2i64)?;
        let div_term = div_term_i64.cast(dtype) / Tensor::from(d_model as f64).cast(dtype);

        let div_term = Tensor::from(10000.0f64).pow(&div_term)?; // [d_model // 2]

        let angle_rates = &position / div_term.unsqueeze(0)?; // [max_len, d_model // 2]
        let sin_part = angle_rates.sin(); // [max_len, d_model // 2]
        let cos_part = angle_rates.cos(); // [max_len, d_model // 2]

        // Interleave sin and cos: [max_len, d_model]
        let mut parts = Vec::with_capacity(d_model);
        for i in 0..(d_model / 2) {
            parts.push(sin_part.slice((0..max_len, i))?.unsqueeze(1)?);
            parts.push(cos_part.slice((0..max_len, i))?.unsqueeze(1)?);
        }

        // Pad if d_model is odd
        if d_model % 2 != 0 {
            let pad = sin_part
                .slice((0..max_len, d_model / 2 - 1))?
                .unsqueeze(1)?;
            parts.push(pad);
        }

        let pe = Tensor::cat(&parts, 1)?; // [max_len, d_model]

        Ok(Self { pe, dropout_prob })
    }

    /// Applies positional encoding to the input tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - A tensor of shape `[batch_size, seq_len, d_model]`.
    ///
    /// # Returns
    ///
    /// A new tensor with the same shape as the input, with positional encodings added and
    /// dropout applied.
    ///
    /// # Errors
    ///
    /// Returns a [`ZyxError::ShapeError`] if:
    /// - Input tensor is not 3-dimensional.
    /// - The input dimension `d_model` does not match the positional encoding.
    /// - The sequence length exceeds the configured `max_len`.
    pub fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let x = x.into();
        let shape = x.shape();

        if shape.len() != 3 {
            return Err(ZyxError::ShapeError(
                "Expected input of shape [batch, seq, dim]".into(),
            ));
        }

        let seq_len = shape[1];
        let dim = shape[2];

        if dim != self.pe.shape()[1] {
            return Err(ZyxError::ShapeError(
                format!(
                    "Mismatch between input dim {} and positional encoding dim {}",
                    dim,
                    self.pe.shape()[1]
                )
                .into(),
            ));
        }

        if seq_len > self.pe.shape()[0] {
            return Err(ZyxError::ShapeError(
                format!(
                    "Input sequence length {} exceeds positional encoding max_len {}",
                    seq_len,
                    self.pe.shape()[0]
                )
                .into(),
            ));
        }

        let pe_slice = self.pe.slice((0..seq_len, 0..dim))?; // [seq_len, dim]
        let pe_expanded = pe_slice.unsqueeze(0)?; // [1, seq_len, dim]

        let out = (x + pe_expanded).dropout(self.dropout_prob);
        Ok(out)
    }
}
