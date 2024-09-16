use zyx::{DType, Tensor};
use zyx_derive::Module;

/// Linear layer
#[derive(Module)]
pub struct Linear {
    /// weight
    pub weight: Tensor,
    /// bias
    pub bias: Option<Tensor>,
}

impl Linear {
    /// Initilize linear layer in device self
    pub fn new(in_features: usize, out_features: usize, bias: bool, dtype: DType) -> Linear {
        use zyx::Scalar;
        let l = -Scalar::sqrt(1.0 / (in_features as f32));
        let u = Scalar::sqrt(1.0 / (in_features as f32));
        Linear {
            weight: Tensor::uniform([in_features, out_features], l..u).cast(dtype),
            bias: if bias { Some(Tensor::uniform([out_features], l..u).cast(dtype)) } else { None },
        }
    }

    /// Forward function for linear.
    /// Calculates x.dot(&self.weight) + self.bias
    pub fn forward(&self, x: impl Into<Tensor>) -> Tensor {
        let x = x.into().dot(&self.weight);
        if let Some(bias) = &self.bias {
            return x + bias;
        }
        return x;
    }
}
