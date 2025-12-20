use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// Group normalization
#[derive(Debug, Module)]
pub struct GroupNorm {
    /// number of groups
    pub num_groups: usize,
    /// epsilon
    pub eps: f32,
    /// shape: [C]
    pub weight: Option<Tensor>,
    /// shape: [C]
    pub bias: Option<Tensor>,
}

impl GroupNorm {
    /// Creates a new GroupNorm module.
    ///
    /// Group Normalization divides the channels into groups and normalizes
    /// the activations within each group, making it independent of batch size.
    ///
    /// # Arguments
    /// - `num_groups`: Number of groups to divide channels into.
    /// - `num_channels`: Total number of input channels (must be divisible by `num_groups`).
    /// - `affine`: If `true`, includes learnable scale (`weight`) and bias (`bias`) parameters.
    ///
    /// # Returns
    /// A `GroupNorm` module with optional learnable parameters.
    ///
    /// # Example
    /// ```rust ignore
    /// let gn = GroupNorm::new(32, 64, true, DType::F32)?;
    /// let out = gn.forward(x)?;
    /// ```
    pub fn new(
        num_groups: usize,
        num_channels: usize,
        affine: bool,
        dtype: DType,
    ) -> Result<Self, ZyxError> {
        if num_channels % num_groups != 0 {
            return Err(ZyxError::ShapeError(
                format!(
                    "num_channels ({}) must be divisible by num_groups ({})",
                    num_channels, num_groups
                )
                .into(),
            ));
        }

        let (weight, bias) = if affine {
            (
                Some(Tensor::ones([num_channels], dtype)),
                Some(Tensor::zeros([num_channels], dtype)),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            num_groups,
            eps: 1e-5,
            weight,
            bias,
        })
    }

    /// Applies group normalization to the input tensor.
    ///
    /// The input is expected to have shape `[N, C, ...]` where:
    /// - `N` is the batch size
    /// - `C` is the number of channels
    /// - Remaining dimensions are treated as spatial or temporal axes
    ///
    /// Normalization is applied per sample, per group:
    /// - Input is reshaped to `[N, num_groups, C / num_groups, ...]`
    /// - Mean and variance are computed across group channels and spatial dims
    /// - Output is normalized and optionally scaled and shifted by `weight` and `bias`
    ///
    /// # Arguments
    /// - `x`: Input tensor of shape `[N, C, *]`
    ///
    /// # Returns
    /// A normalized tensor of the same shape as input.
    ///
    /// # Errors
    /// Returns an error if the input shape is invalid or incompatible with `num_groups`.
    ///
    /// # Example
    /// ```rust ignore
    /// let gn = GroupNorm::new(8, 64, true, DType::F32)?;
    /// let out = gn.forward(x)?;
    /// ```
    pub fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let x = x.into();
        let shape = x.shape();

        if shape.len() < 2 {
            return Err(ZyxError::ShapeError(
                format!("GroupNorm requires at least 2D input, got {:?}", shape).into(),
            ));
        }

        let n = shape[0];
        let c = shape[1];
        let rest = &shape[2..];

        if c % self.num_groups != 0 {
            return Err(ZyxError::ShapeError(
                format!(
                    "num_channels ({}) must be divisible by num_groups ({})",
                    c, self.num_groups
                )
                .into(),
            ));
        }

        let group_size = c / self.num_groups;

        // Reshape: [N, C, ...] → [N, G, C//G, ...]
        let mut new_shape = vec![n, self.num_groups, group_size];
        new_shape.extend_from_slice(rest);
        let x = x.reshape(new_shape.clone())?;

        // Axes to normalize over: [2, 3, 4, ...]
        let axes = 2..(new_shape.len() as i32);

        let eps = Tensor::from(self.eps).cast(x.dtype());
        let mean = x.mean_keepdim(axes.clone())?;
        let var = x.var_keepdim(axes.clone())?;

        let x = (x - mean) / (var + eps).sqrt();

        // Reshape back to original shape
        let mut x = x.reshape(shape)?;

        if let Some(weight) = &self.weight {
            x = x * weight;
        }
        if let Some(bias) = &self.bias {
            x = x + bias;
        }

        Ok(x)
    }
}
