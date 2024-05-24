use zyx::Tensor;

/// An Elman RNN cell without nonlinearity
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

impl<'a> IntoIterator for &'a RNNCell {
    type Item = &'a Tensor;
    type IntoIter = alloc::vec::IntoIter<&'a Tensor>;
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

impl<'a> IntoIterator for &'a mut RNNCell {
    type Item = &'a mut Tensor;
    type IntoIter = alloc::vec::IntoIter<&'a mut Tensor>;
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

impl RNNCell {
    /// Initilize linear layer in device self
    fn new(self, input_size: usize, hidden_size: usize) -> RNNCell {
        let l = -(1./(hidden_size as f32)).sqrt();
        let u = (1./(hidden_size as f32)).sqrt();
        RNNCell {
            weight_ih: Tensor::uniform([hidden_size, input_size], l..u),
            weight_hh: Tensor::uniform([hidden_size, hidden_size], l..u),
            bias_ih: Some(Tensor::uniform([hidden_size], l..u)),
            bias_hh: Some(Tensor::uniform([hidden_size], l..u)),
        }
    }

    /// Forward function for RNNCell.
    /// Takes x (input) and hidden layer. Outputs output and new hidden layer.
    /// returns (x, self.weight_ih.dot(x) + self.bias_ih + self.weight_hh.dot(hidden) + self.bias_hh)
    ///
    /// This function does not apply nonlinearity and it does not change x
    pub fn forward(&self, x: impl Into<Tensor>, hidden: impl Into<Tensor>) -> (Tensor, Tensor) {
        let x = x.into();
        let mut hx = self.weight_hh.dot(hidden);
        if let Some(b) = &self.bias_hh { hx = hx + b }
        hx = hx + self.weight_ih.dot(&x);
        if let Some(b) = &self.bias_ih { hx = hx + b }
        return (x, hx);
    }
}
