use zyx::{DType, IntoShape, Tensor, ZyxError};
use zyx_derive::Module;

/// A Layer Normalization layer.
///
/// Layer Normalization normalizes the inputs across the specified dimensions (typically the last N dimensions)
/// for each example independently. It optionally supports learnable scale (`weight`) and bias (`bias_tensor`) parameters.
#[derive(Debug, Module)]
pub struct LayerNorm {
    normalized_shape: Vec<usize>,
    eps: f64,
    weight: Option<Tensor>,
    bias_tensor: Option<Tensor>,
}

impl LayerNorm {
    /// Creates a new `LayerNorm` layer.
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - The shape of the dimensions to normalize. Usually corresponds to the last N dimensions of the input tensor.
    /// * `eps` - A small value added to the denominator for numerical stability.
    /// * `elementwise_affine` - If `true`, includes a learnable scale parameter (`weight`).
    /// * `bias` - If `true`, includes a learnable bias parameter (`bias_tensor`).
    /// * `dtype` - The data type of the optional learnable parameters.
    ///
    /// # Returns
    ///
    /// Returns `Ok(LayerNorm)` if initialization is successful, or a `ZyxError` if there is an issue with shape or tensor creation.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{DType, Tensor};
    /// # use your_crate::LayerNorm;
    /// let layer_norm = LayerNorm::new([10, 20], 1e-5, true, true, DType::F32).unwrap();
    /// ```
    pub fn new(
        normalized_shape: impl IntoShape,
        eps: f64,
        elementwise_affine: bool,
        bias: bool,
        dtype: DType,
    ) -> Result<Self, ZyxError> {
        let normalized_shape: Vec<usize> = normalized_shape.into_shape().collect();

        // Optional learnable parameters
        let weight = if elementwise_affine {
            Some(Tensor::ones(&normalized_shape, dtype))
        } else {
            None
        };

        let bias_tensor = if bias {
            Some(Tensor::zeros(&normalized_shape, dtype))
        } else {
            None
        };

        Ok(Self {
            normalized_shape,
            eps,
            weight,
            bias_tensor,
        })
    }

    /// Performs the forward pass of the LayerNorm layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor to normalize.
    ///
    /// # Returns
    ///
    /// Returns a new `Tensor` that is normalized along the last `normalized_shape.len()` dimensions.
    ///
    /// # Errors
    ///
    /// Returns a `ZyxError` if the input tensor rank is smaller than the rank of `normalized_shape`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::{DType, Tensor};
    /// # use your_crate::LayerNorm;
    /// let layer_norm = LayerNorm::new([10, 20], 1e-5, true, true, DType::F32).unwrap();
    /// let input = Tensor::randn(&[2, 10, 20], DType::F32);
    /// let output = layer_norm.forward(input).unwrap();
    /// ```
    pub fn forward(&self, input: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let input = input.into();
        let input_shape = input.shape();
        let input_rank = input_shape.len();
        let norm_rank = self.normalized_shape.len();

        if input_rank < norm_rank {
            return Err(ZyxError::shape_error(
                format!(
                    "LayerNorm: input rank ({}) smaller than normalized_shape rank ({})",
                    input_rank, norm_rank
                )
                .into(),
            ));
        }

        // Determine axes to normalize over (last `norm_rank` dims)
        let axes: Vec<i32> = (input_rank - norm_rank..input_rank)
            .map(|i| i as i32)
            .collect();

        // Compute mean and variance along those axes (keep dims for broadcasting)
        let mean = input.mean_axes_keepdim(axes.clone())?;
        let variance = input.var_axes_keepdim(axes)?;

        // Normalize: (x - mean) / sqrt(var + eps)
        let normalized = (input - &mean) / (variance + self.eps).sqrt();

        // Apply learnable affine transformation if enabled
        let mut output = normalized;

        if let Some(ref weight) = self.weight {
            output = output * weight;
        }

        if let Some(ref bias) = self.bias_tensor {
            output = output + bias;
        }

        Ok(output)
    }
}
