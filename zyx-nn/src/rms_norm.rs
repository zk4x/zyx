use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// RMS norm layer
#[derive(Debug, Module)]
pub struct RMSNorm {
    /// weight, scale
    pub scale: Tensor,
    /// small value to avoid division by zero
    pub eps: f64,
}

impl RMSNorm {
    /// Initialize RMSNorm layer
    pub fn new(dim: usize, dtype: DType) -> RMSNorm {
        RMSNorm {
            scale: Tensor::ones(dim, dtype),
            eps: 1e-6,
        }
    }

    /// RMSNorm forward function
    pub fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let x = x.into();
        let dtype = x.dtype();
        let x_normed =
            &x * (x.pow(2)?.mean_axes_keepdim([-1])? + Tensor::from(self.eps).cast(dtype)).rsqrt();
        return Ok(x_normed * &self.scale);
    }
}
