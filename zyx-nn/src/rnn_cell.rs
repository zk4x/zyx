use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// An Elman RNN cell without nonlinearity
#[derive(Module)]
pub struct RNNCell {
    /// the learnable input-hidden weights, of shape (hidden_size, input_size)
    pub weight_ih: Tensor,
    /// the learnable hidden-hidden weights, of shape (hidden_size, hidden_size)
    pub weight_hh: Tensor,
    /// the learnable input-hidden bias, of shape (hidden_size)
    pub bias_ih: Option<Tensor>,
    /// the learnable hidden-hidden bias, of shape (hidden_size)
    pub bias_hh: Option<Tensor>,
}

impl RNNCell {
    /// Initialize linear layer in device self
    pub fn new(self, input_size: usize, hidden_size: usize, dtype: DType) -> Result<RNNCell, ZyxError> {
        use zyx::Scalar;
        let l = Scalar::sqrt(-(1. / (hidden_size as f32)));
        let u = Scalar::sqrt(1. / (hidden_size as f32));
        Ok(RNNCell {
            weight_ih: Tensor::uniform([hidden_size, input_size], l..u)?.cast(dtype),
            weight_hh: Tensor::uniform([hidden_size, hidden_size], l..u)?.cast(dtype),
            bias_ih: Some(Tensor::uniform([hidden_size], l..u)?.cast(dtype)),
            bias_hh: Some(Tensor::uniform([hidden_size], l..u)?.cast(dtype)),
        })
    }

    /// Forward function for RNNCell.
    /// Takes x (input) and hidden layer. Outputs output and new hidden layer.
    /// returns (x, self.weight_ih.dot(x) + self.bias_ih + self.weight_hh.dot(hidden) + self.bias_hh)
    ///
    /// This function does not apply nonlinearity and it does not change x
    pub fn forward(&self, x: impl Into<Tensor>, hidden: impl Into<Tensor>) -> (Tensor, Tensor) {
        let x = x.into();
        let mut hx = self.weight_hh.dot(hidden);
        if let Some(b) = &self.bias_hh {
            hx = hx + b
        }
        hx = hx + self.weight_ih.dot(&x);
        if let Some(b) = &self.bias_ih {
            hx = hx + b
        }
        return (x, hx);
    }
}
