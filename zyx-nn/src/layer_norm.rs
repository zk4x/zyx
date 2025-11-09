use zyx::{DType, IntoShape, Tensor, ZyxError};

pub struct LayerNorm {
    normalized_shape: Vec<usize>,
    eps: f64,
    elementwise_affine: bool,
    bias: bool,
    weight: Option<Tensor>,
    bias_tensor: Option<Tensor>,
    dtype: DType,
}

impl LayerNorm {
    /// Initialize LayerNorm layer
    pub fn new(
        normalized_shape: impl IntoShape,
        eps: f64,
        elementwise_affine: bool,
        bias: bool,
        dtype: DType,
    ) -> Result<LayerNorm, ZyxError> {
        // Convert normalized_shape to Vec<usize>
        let normalized_shape: Vec<usize> = normalized_shape.into_shape().collect();
        // Initialize the weight and bias tensors if elementwise_affine and bias are enabled
        let weight = if elementwise_affine {
            Some(Tensor::ones(normalized_shape.clone(), dtype))
        } else {
            None
        };
        let bias_tensor = if bias {
            Some(Tensor::zeros(normalized_shape.clone(), dtype))
        } else {
            None
        };
        // Return the LayerNorm instance
        Ok(LayerNorm {
            normalized_shape,
            eps,
            elementwise_affine,
            bias,
            weight,
            bias_tensor,
            dtype,
        })
    }

    /// Forward pass of the LayerNorm
    pub fn forward(&self, input: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let input = input.into();

        // Ensure that the last dimensions match `normalized_shape`
        let normalized_dim = self.normalized_shape.len();
        let input_dim = input.rank();

        if input_dim < normalized_dim {
            return Err(ZyxError::shape_error("input_dim > normalized_dim".into()));
            // Invalid shape dimensions
        }

        // Compute mean and variance along the specified axes
        let axes: Vec<i32> = (input_dim - normalized_dim..input_dim)
            .map(|a| a as i32)
            .collect(); // Normalize over the last dimensions
        let mean = input.mean_axes(axes.clone())?; // Mean along the axes
        let variance = input.var_axes(axes)?; // Variance along the axes

        // Normalize: (x - mean) / sqrt(var + epsilon)
        let normalized = (input - &mean) / (variance + self.eps).sqrt();

        // If affine, apply gamma (weight) and beta (bias)
        let mut output = normalized;

        if let Some(ref weight) = self.weight {
            output = output * weight; // Apply the scaling factor (gamma)
        }

        if let Some(ref bias_tensor) = self.bias_tensor {
            output = output + bias_tensor; // Apply the shifting factor (beta)
        }

        Ok(output)
    }
}
