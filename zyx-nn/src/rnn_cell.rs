use zyx_core::backend::Backend;
use zyx_core::tensor::{IntoTensor, Tensor};

/// An Elman RNN cell without nonlinearity
pub struct RNNCell<B: Backend> {
    /// the learnable input-hidden weights, of shape (hidden_size, input_size)
    pub weight_ih: Tensor<B>,
    /// the learnable hidden-hidden weights, of shape (hidden_size, hidden_size)
    pub weight_hh: Tensor<B>,
    /// the learnable input-hidden bias, of shape (hidden_size)
    pub bias_ih: Option<Tensor<B>>,
    /// the learnable hidden-hidden bias, of shape (hidden_size)
    pub bias_hh: Option<Tensor<B>>,
}

/// Initilization trait for linear layer
pub trait RNNCellInit: Backend {
    /// Initilize linear layer in device self
    fn rnn_cell(self, input_size: usize, hidden_size: usize) -> RNNCell<Self> {
        let l = -(1./(hidden_size as f32)).sqrt();
        let u = (1./(hidden_size as f32)).sqrt();
        RNNCell {
            weight_ih: self.uniform([hidden_size, input_size], l..u).unwrap(),
            weight_hh: self.uniform([hidden_size, hidden_size], l..u).unwrap(),
            bias_ih: Some(self.uniform([hidden_size], l..u).unwrap()),
            bias_hh: Some(self.uniform([hidden_size], l..u).unwrap()),
        }
    }
}

impl<B: Backend> RNNCellInit for B {}

impl<'a, B: Backend> IntoIterator for &'a RNNCell<B> {
    type Item = &'a Tensor<B>;
    type IntoIter = alloc::vec::IntoIter<&'a Tensor<B>>;
    fn into_iter(self) -> Self::IntoIter {
        let mut res = alloc::vec![&self.weight_ih];
        if let Some(b) = &self.bias_ih {
            res.push(b);
        }
        res.push(&self.weight_hh);
        if let Some(b) = &self.bias_hh {
            res.push(b);
        }
        res.into_iter()
    }
}

impl<'a, B: Backend> IntoIterator for &'a mut RNNCell<B> {
    type Item = &'a mut Tensor<B>;
    type IntoIter = alloc::vec::IntoIter<&'a mut Tensor<B>>;
    fn into_iter(self) -> Self::IntoIter {
        let mut res = alloc::vec![&mut self.weight_ih];
        if let Some(b) = &mut self.bias_ih {
            res.push(b);
        }
        res.push(&mut self.weight_hh);
        if let Some(b) = &mut self.bias_hh {
            res.push(b);
        }
        res.into_iter()
    }
}

impl<B: Backend> RNNCell<B> {
    /// Forward function for RNNCell.
    /// Takes x (input) and hidden layer. Outputs output and new hidden layer.
    /// returns (x, self.weight_ih.dot(x) + self.bias_ih + self.weight_hh.dot(hidden) + self.bias_hh)
    ///
    /// This function does not apply nonlinearity and it does not change x
    pub fn forward(&self, x: impl IntoTensor<B>, hidden: impl IntoTensor<B>) -> (Tensor<B>, Tensor<B>) {
        let x = self.weight_ih.backend().tensor(x).unwrap();
        let mut hx = self.weight_hh.dot(hidden);
        if let Some(b) = &self.bias_hh { hx = hx + b }
        hx = hx + self.weight_ih.dot(&x);
        if let Some(b) = &self.bias_ih { hx = hx + b }
        return (x, hx);
    }
}
