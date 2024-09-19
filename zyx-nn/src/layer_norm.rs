use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// Layer norm layer
#[derive(Module)]
pub struct LayerNorm {
    /// weight
    pub weight: Option<Tensor>,
    /// bias
    pub bias: Option<Tensor>,
    /// a value added to the denominator for numerical stability
    pub eps: f32,
    d_dims: usize,
}

impl LayerNorm {
    /// Initialize LayerNorm layer
    pub fn new(normalized_shape: impl zyx::IntoShape, dtype: DType) -> Result<LayerNorm, ZyxError> {
        Ok(LayerNorm {
            d_dims: normalized_shape.rank(),
            weight: Some(Tensor::randn(normalized_shape.clone(), dtype)?),
            bias: Some(Tensor::randn(normalized_shape, dtype)?),
            eps: 1e-5,
        })
    }

    /// Forward function for layer_norm.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let axes = -(self.d_dims as isize)..=-1;
        let mut x = (x - x.mean(axes.clone())) / (x.var(axes) + self.eps).sqrt();
        if let Some(w) = &self.weight {
            x = x * w;
        }
        if let Some(b) = &self.bias {
            x = x + b;
        }
        return x;
    }
}
