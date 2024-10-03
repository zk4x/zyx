use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// RMS norm layer
#[derive(Module)]
pub struct RMSNorm {
    /// weight, scale
    pub scale: Tensor,
    /// small value to avoid division by zero
    pub eps: f32,
}

impl RMSNorm {
    /// Initialize RMSNorm layer
    pub fn new(dim: usize, dtype: DType) -> RMSNorm {
        RMSNorm {
            scale: Tensor::ones(dim, dtype),
            eps: 1e-6,
        }
    }

    ///RMSNorm forward function
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, ZyxError> {
        let x_normed = x * (x.pow(2)?.mean_kd([-1])? + self.eps).rsqrt();
        return Ok(x_normed * &self.scale);
    }
}
