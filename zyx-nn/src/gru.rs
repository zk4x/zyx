use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// GRU cell (PyTorch-style)
#[derive(Debug, Module)]
pub struct GRUCell {
    /// weight ih
    pub weight_ih: Tensor, // (3*hidden_size, input_size)
    /// weight hh
    pub weight_hh: Tensor, // (3*hidden_size, hidden_size)
    /// bias ih
    pub bias_ih: Option<Tensor>, // (3*hidden_size)
    /// bias hh
    pub bias_hh: Option<Tensor>, // (3*hidden_size)
    /// hidden
    pub hidden_size: usize,
}

impl GRUCell {
    /// GRU new
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        bias: bool,
        dtype: DType,
    ) -> Result<Self, ZyxError> {
        let limit = 1.0 / (hidden_size as f32).sqrt();

        Ok(GRUCell {
            weight_ih: Tensor::uniform([3 * hidden_size, input_size], -limit..limit)?.cast(dtype),
            weight_hh: Tensor::uniform([3 * hidden_size, hidden_size], -limit..limit)?.cast(dtype),
            bias_ih: if bias {
                Some(Tensor::uniform([3 * hidden_size], -limit..limit)?.cast(dtype))
            } else {
                None
            },
            bias_hh: if bias {
                Some(Tensor::uniform([3 * hidden_size], -limit..limit)?.cast(dtype))
            } else {
                None
            },
            hidden_size,
        })
    }

    /// Forward pass: x (batch, input_size), h (batch, hidden_size)
    pub fn forward(&self, input: Tensor, hx: Tensor) -> Result<Tensor, ZyxError> {
        let hs = self.hidden_size;

        // 🔹 Linear for input-to-hidden: x @ W_ih^T + b_ih
        let mut gates = input.matmul(&self.weight_ih.t())?;
        if let Some(b_ih) = &self.bias_ih {
            gates = gates + b_ih.reshape([1, 3 * hs])?;
        }

        // 🔹 Linear for hidden-to-hidden: hx @ W_hh^T + b_hh
        let mut gates_h = hx.matmul(&self.weight_hh.t())?;
        if let Some(b_hh) = &self.bias_hh {
            gates_h = gates_h + b_hh.reshape([1, 3 * hs])?;
        }

        // 🔹 Split gates: (z, r, n)
        let z = (gates.get((.., 0..hs))? + gates_h.get((.., 0..hs))?).sigmoid();
        let r = (gates.get((.., hs..2 * hs))? + gates_h.get((.., hs..2 * hs))?).sigmoid();
        let n_input = gates.get((.., 2 * hs..3 * hs))?;
        let n_hidden = gates_h.get((.., 2 * hs..3 * hs))?;

        // 🔹 Candidate hidden: n = tanh(x_n + r * (h @ W_hh_n + b_hh_n))
        let n = (n_input + r * n_hidden).tanh();

        // 🔹 Final hidden state: h_next = (1 - z) * n + z * hx
        let one = Tensor::ones_like(&z);
        let h_next = (one - &z) * n + &z * hx;

        Ok(h_next)
    }
}
