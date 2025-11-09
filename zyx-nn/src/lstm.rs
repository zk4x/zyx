use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// A single LSTM (Long Short-Term Memory) cell.
///
/// # Overview
/// The `LSTMCell` module implements a basic LSTM cell similar to `torch.nn.LSTMCell`.
/// It maintains separate weight matrices for the input and hidden states and optional biases.
///
/// The forward pass computes the next hidden and cell states given an input tensor and previous states.
///
/// # Fields
/// - `w_ih`: Input-to-hidden weights with shape `(4 * hidden_size, input_size)`.
/// - `w_hh`: Hidden-to-hidden weights with shape `(4 * hidden_size, hidden_size)`.
/// - `b_ih`: Optional input bias with shape `(4 * hidden_size)`.
/// - `b_hh`: Optional hidden bias with shape `(4 * hidden_size)`.
/// - `hidden_size`: The size of the hidden state.
///
/// # Example
/// ```rust
/// use zyx::{DType, Tensor};
/// use my_crate::LSTMCell;
///
/// let input_size = 10;
/// let hidden_size = 20;
/// let lstm = LSTMCell::new(input_size, hidden_size, true, Some(DType::F32)).unwrap();
///
/// let x = Tensor::zeros([5, input_size], DType::F32); // batch_size = 5
/// let h = Tensor::zeros([5, hidden_size], DType::F32);
/// let c = Tensor::zeros([5, hidden_size], DType::F32);
///
/// let (h_next, c_next) = lstm.forward(&x, &h, &c).unwrap();
/// ```
#[derive(Debug, Module)]
pub struct LSTMCell {
    // weights for input and hidden
    w_ih: Tensor,         // (4 * hidden_size, input_size)
    w_hh: Tensor,         // (4 * hidden_size, hidden_size)
    b_ih: Option<Tensor>, // (4 * hidden_size)
    b_hh: Option<Tensor>, // (4 * hidden_size)
    hidden_size: usize,
}

impl LSTMCell {
    /// Creates a new `LSTMCell`.
    ///
    /// # Arguments
    /// - `input_size`: Number of input features.
    /// - `hidden_size`: Number of features in the hidden state.
    /// - `bias`: Whether to include bias terms.
    /// - `dtype`: Optional data type of the weights and biases (default `F32`).
    ///
    /// # Returns
    /// A `Result` wrapping the created `LSTMCell` or a `ZyxError` if initialization fails.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        bias: bool,
        dtype: Option<DType>,
    ) -> Result<Self, ZyxError> {
        let dtype = dtype.unwrap_or(DType::F32);

        // In PyTorch, parameters are stacked as [i, f, g, o]
        let k = (1.0 / (hidden_size as f32).sqrt()) as f32;
        let w_ih = Tensor::uniform([4 * hidden_size, input_size], -k..k)?.cast(dtype);
        let w_hh = Tensor::uniform([4 * hidden_size, hidden_size], -k..k)?.cast(dtype);

        let (b_ih, b_hh) = if bias {
            (
                Some(Tensor::zeros([4 * hidden_size], dtype)),
                Some(Tensor::zeros([4 * hidden_size], dtype)),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            hidden_size,
        })
    }

    /// Performs a forward pass through the LSTM cell.
    ///
    /// # Arguments
    /// - `x`: Input tensor of shape `(batch_size, input_size)`.
    /// - `h`: Previous hidden state of shape `(batch_size, hidden_size)`.
    /// - `c`: Previous cell state of shape `(batch_size, hidden_size)`.
    ///
    /// # Returns
    /// A `Result` containing a tuple `(h_next, c_next)`:
    /// - `h_next`: Next hidden state `(batch_size, hidden_size)`.
    /// - `c_next`: Next cell state `(batch_size, hidden_size)`.
    ///
    /// # Example
    /// ```rust
    /// let (h_next, c_next) = lstm.forward(&x, &h, &c).unwrap();
    /// ```
    pub fn forward(
        &self,
        x: &Tensor,
        h: &Tensor,
        c: &Tensor,
    ) -> Result<(Tensor, Tensor), ZyxError> {
        let hs = self.hidden_size;

        // Gates computation — lazy, will be fused
        let mut gates = x.matmul(&self.w_ih.t())? + h.matmul(&self.w_hh.t())?;
        if let Some(b) = &self.b_ih {
            gates = &gates + b;
        }
        if let Some(b) = &self.b_hh {
            gates = &gates + b;
        }

        // Split gates: [i, f, g, o]
        let i = gates.narrow(1, 0, hs)?.sigmoid();
        let f = gates.narrow(1, hs, hs)?.sigmoid();
        let g = gates.narrow(1, 2 * hs, hs)?.tanh();
        let o = gates.narrow(1, 3 * hs, hs)?.sigmoid();

        // Next states
        let c_next = &f * c + &i * &g;
        let h_next = &o * c_next.tanh();

        Ok((h_next, c_next))
    }
}
