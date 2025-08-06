use zyx::{DType, IntoShape, Tensor, ZyxError};
use zyx_derive::Module;

/// Applies a 2D convolution over an input signal composed of several input planes.
///
/// See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d
#[derive(Debug, Module)]
pub struct Conv2d {
    stride: Vec<usize>,
    dilation: Vec<usize>,
    groups: usize,
    padding: Vec<usize>,
    /// weight
    pub weight: Tensor,
    /// bias
    pub bias: Option<Tensor>,
}

impl Conv2d {
    /// Initialize Conv2d
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: impl IntoShape,
        stride: impl IntoShape,
        padding: impl IntoShape,
        dilation: impl IntoShape,
        groups: usize,
        bias: bool,
        dtype: DType,
    ) -> Result<Self, ZyxError> {
        let mut kernel_size: Vec<usize> = kernel_size.into_shape().collect();
        kernel_size.push(2);
        let scale = 1f32 / ((in_channels * kernel_size.iter().product::<usize>()) as f32).sqrt();
        let mut weight_shape = vec![out_channels, in_channels / groups];
        weight_shape.extend(kernel_size);
        Ok(Conv2d {
            stride: stride.into_shape().collect(),
            dilation: dilation.into_shape().collect(),
            groups,
            padding: padding.into_shape().collect(),
            weight: Tensor::uniform(weight_shape, -scale..scale)?.cast(dtype),
            bias: if bias {
                Some(Tensor::uniform(out_channels, -scale..scale)?.cast(dtype))
            } else {
                None
            },
        })
    }

    /// Forward conv2d layer
    pub fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        x.into().conv(
            &self.weight,
            self.bias.as_ref(),
            self.groups,
            &self.stride,
            &self.dilation,
            &self.padding,
        )
    }
}
