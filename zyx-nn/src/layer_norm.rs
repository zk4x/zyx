use zyx_core::backend::Backend;
use zyx_core::dtype::DType;
use zyx_core::shape::Shape;
use zyx_core::tensor::Tensor;

/// Lyaer norm layer
pub struct LayerNorm<B: Backend> {
    /// weight
    pub weight: Option<Tensor<B>>,
    /// bias
    pub bias: Option<Tensor<B>>,
    /// a value added to the denominator for numerical stability
    pub eps: f32,
    d_dims: usize,
}

/// Initilization trait for layer_norm layer
pub trait LayerNormInit: Backend {
    /// Initilize layer_norm layer in device self
    fn layer_norm(self, normalized_shape: impl Into<Shape>) -> LayerNorm<Self> {
        let normalized_shape = normalized_shape.into();
        LayerNorm {
            d_dims: normalized_shape.rank(),
            weight: Some(self.randn(normalized_shape.clone(), DType::F32).unwrap()),
            bias: Some(self.randn(normalized_shape, DType::F32).unwrap()),
            eps: 1e-5,
        }
    }
}

impl<B: Backend> LayerNormInit for B {}

impl<'a, B: Backend> IntoIterator for &'a LayerNorm<B> {
    type Item = &'a Tensor<B>;
    type IntoIter = alloc::vec::IntoIter<&'a Tensor<B>>;
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

impl<B: Backend> LayerNorm<B> {
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
