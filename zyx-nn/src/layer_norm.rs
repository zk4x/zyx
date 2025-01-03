use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// Layer norm layer
#[derive(Debug, Module)]
pub struct LayerNorm {
    /// weight
    pub weight: Option<Tensor>,
    /// bias
    pub bias: Option<Tensor>,
    /// a value added to the denominator for numerical stability
    pub eps: f64,
    /// Number of dims across which normalization happens
    pub d_dims: usize,
}

impl LayerNorm {
    /// Initialize LayerNorm layer
    pub fn init(normalized_shape: impl zyx::IntoShape, bias: bool, dtype: DType) -> Result<LayerNorm, ZyxError> {
        Ok(LayerNorm {
            d_dims: normalized_shape.rank(),
            weight: Some(Tensor::randn(normalized_shape.clone(), dtype)?),
            bias: if bias { Some(Tensor::randn(normalized_shape, dtype)?) } else { None },
            eps: 1e-5,
        })
    }

    /// Forward function for layer_norm.
    pub fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let x = x.into();
        let axes = -(self.d_dims as isize)..=-1;
        let eps = Tensor::constant(self.eps).cast(x.dtype());
        let a = &x - x.mean_kd(axes.clone()).unwrap();
        let b = (x.var_kd(axes, 1).unwrap() + eps).sqrt();
        let mut x = a/b;
        if let Some(w) = &self.weight {
            x = x * w;
        }
        if let Some(b) = &self.bias {
            x = x + b;
        }
        return Ok(x);
    }
}
