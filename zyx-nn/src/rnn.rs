// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// An Elman RNN cell with optional nonlinearity.
///
/// # Overview
/// The `RNNCell` module implements a basic RNN cell similar to `torch.nn.RNNCell`.
/// It maintains separate weight matrices for the input and hidden states and optional biases.
///
/// The forward pass computes the next hidden state given an input tensor and previous hidden state.
///
/// # Fields
/// - `weight_ih`: Input-to-hidden weights with shape `(hidden_size, input_size)`.
/// - `weight_hh`: Hidden-to-hidden weights with shape `(hidden_size, hidden_size)`.
/// - `bias_ih`: Optional input bias with shape `(hidden_size)`.
/// - `bias_hh`: Optional hidden bias with shape `(hidden_size)`.
/// - `hidden_size`: The size of the hidden state.
/// - `nonlinearity`: The nonlinear activation function to use (tanh or relu).
///
/// # Example
/// ```rust
/// use zyx::{DType, Tensor};
/// use zyx_nn::RNNCell;
///
/// let input_size = 10;
/// let hidden_size = 20;
/// let rnn = RNNCell::new(input_size, hidden_size, true, "tanh", Some(DType::F32)).unwrap();
///
/// let x = Tensor::zeros([5, input_size], DType::F32); // batch_size = 5
/// let h = Tensor::zeros([5, hidden_size], DType::F32);
///
/// let h_next = rnn.forward(&x, &h).unwrap();
/// ```
#[derive(Debug, Module)]
#[cfg_attr(feature = "py", pyo3::pyclass)]
pub struct RNNCell {
    /// the learnable input-hidden weights, of shape (hidden_size, input_size)
    pub weight_ih: Tensor,
    /// the learnable hidden-hidden weights, of shape (hidden_size, hidden_size)
    pub weight_hh: Tensor,
    /// the learnable input-hidden bias, of shape (hidden_size)
    pub bias_ih: Option<Tensor>,
    /// the learnable hidden-hidden bias, of shape (hidden_size)
    pub bias_hh: Option<Tensor>,
    /// the hidden size
    pub hidden_size: u64,
    /// the nonlinearity to use ("tanh" or "relu")
    nonlinearity: &'static str,
}

impl RNNCell {
    /// Creates a new `RNNCell`.
    ///
    /// # Arguments
    /// - `input_size`: Number of input features.
    /// - `hidden_size`: Number of features in the hidden state.
    /// - `bias`: Whether to include bias terms.
    /// - `nonlinearity`: The nonlinearity to use - either `"tanh"` (default) or `"relu"`.
    /// - `dtype`: Optional data type of the weights and biases (default `F32`).
    ///
    /// # Returns
    /// A `Result` wrapping the created `RNNCell` or a `ZyxError` if initialization fails.
    pub fn new(
        input_size: u64,
        hidden_size: u64,
        bias: bool,
        nonlinearity: &'static str,
        dtype: Option<DType>,
    ) -> Result<RNNCell, ZyxError> {
        let dtype = dtype.unwrap_or(DType::F32);
        let gain = match nonlinearity {
            "relu" => 1.0 / (3_u32.pow(2) as f32).sqrt(),
            _ => 1.0,
        };
        let scale = gain / (hidden_size as f32).sqrt();

        let weight_ih = Tensor::uniform([hidden_size, input_size], -scale..scale)?.cast(dtype);
        let weight_hh = Tensor::uniform([hidden_size, hidden_size], -scale..scale)?.cast(dtype);

        let bias_ih = if bias {
            Some(Tensor::zeros([hidden_size], dtype))
        } else {
            None
        };
        let bias_hh = if bias {
            Some(Tensor::zeros([hidden_size], dtype))
        } else {
            None
        };

        Ok(RNNCell {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            hidden_size,
            nonlinearity,
        })
    }

    /// Forward function for RNNCell.
    ///
    /// Takes x (input) and hidden state. Returns new hidden state.
    ///
    /// # Arguments
    /// - `x`: Input tensor of shape `(batch_size, input_size)`.
    /// - `hx`: Previous hidden state of shape `(batch_size, hidden_size)`.
    ///
    /// # Returns
    /// A `Result` containing the next hidden state tensor of shape `(batch_size, hidden_size)`.
    ///
    /// # Example
    /// ```rust ignore
    /// let h_next = rnn.forward(&x, &h).unwrap();
    /// ```
    pub fn forward(&self, x: &Tensor, hx: &Tensor) -> Result<Tensor, ZyxError> {
        let h_new = x.matmul(&self.weight_ih.t())? + hx.matmul(&self.weight_hh.t())?;

        let h_new = if let Some(b) = &self.bias_ih {
            h_new + b
        } else {
            h_new
        };

        let h_new = if let Some(b) = &self.bias_hh {
            h_new + b
        } else {
            h_new
        };

        let h_new = match self.nonlinearity {
            "relu" => h_new.relu(),
            _ => h_new.tanh(),
        };

        Ok(h_new)
    }
}
