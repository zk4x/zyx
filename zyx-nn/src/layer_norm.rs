/// Lyaer norm layer
pub struct LayerNorm {
    /// weight
    pub weight: Option<Tensor>,
    /// bias
    pub bias: Option<Tensor>,
    /// a value added to the denominator for numerical stability
    pub eps: f32,
    d_dims: usize,
}

impl<'a> IntoIterator for &'a LayerNorm {
    type Item = &'a Tensor;
    type IntoIter = alloc::vec::IntoIter<&'a Tensor>;
    fn into_iter(self) -> Self::IntoIter {
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => alloc::vec![w, b].into_iter(),
            (Some(w), None) => alloc::vec![w].into_iter(),
            (None, Some(b)) => alloc::vec![b].into_iter(),
            (None, None) => alloc::vec![].into_iter(),
        }
    }
}

impl<'a, B: Backend> IntoIterator for &'a mut LayerNorm<B> {
    type Item = &'a mut Tensor<B>;
    type IntoIter = alloc::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        match (&mut self.weight, &mut self.bias) {
            (Some(w), Some(b)) => alloc::vec![w, b].into_iter(),
            (Some(w), None) => alloc::vec![w].into_iter(),
            (None, Some(b)) => alloc::vec![b].into_iter(),
            (None, None) => alloc::vec![].into_iter(),
        }
    }
}

impl LayerNorm {
    /// Initialize layer_norm layer in device self
    fn new(self, normalized_shape: impl Into<Shape>) -> LayerNorm<Self> {
        let normalized_shape = normalized_shape.into();
        LayerNorm {
            d_dims: normalized_shape.rank(),
            weight: Some(self.randn(normalized_shape.clone(), DType::F32).unwrap()),
            bias: Some(self.randn(normalized_shape, DType::F32).unwrap()),
            eps: 1e-5,
        }
    }

    /// Forward function for layer_norm.
    pub fn forward(&self, x: &Tensor<B>) -> Tensor<B> {
        let axes = -(self.d_dims as i64)..=-1;
        let mut x = (x - x.mean(&axes)) / (x.var(axes) + self.eps).sqrt();
        if let Some(w) = &self.weight {
            x = x * w;
        }
        if let Some(b) = &self.bias {
            x = x + b;
        }
        return x;
    }
}
