// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// Applies a 2D convolution over an input signal composed of several input planes.
///
/// See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d
#[derive(Debug, Module)]
#[cfg_attr(feature = "py", pyo3::pyclass)]
pub struct Conv2d {
    stride: Vec<i64>,
    dilation: Vec<i64>,
        groups: u64,
    padding: Vec<i64>,
    /// weight
    pub weight: Tensor,
    /// bias
    pub bias: Option<Tensor>,
}

impl Conv2d {
    /// Initialize Conv2d
    #[allow(clippy::too_many_arguments)] // mirrors PyTorch API with multiple config parameters
    pub fn new(
        in_channels: i64,
        out_channels: i64,
        kernel_size: impl IntoIterator<Item = impl Into<Tensor>>,
        stride: impl IntoIterator<Item = impl Into<Tensor>>,
        padding: impl IntoIterator<Item = impl Into<Tensor>>,
        dilation: impl IntoIterator<Item = impl Into<Tensor>>,
    groups: u64,
        bias: bool,
        dtype: DType,
    ) -> Result<Self, ZyxError> {
        let mut kernel_size: Vec<i64> = kernel_size
            .into_iter()
            .map(|s| s.into().item::<i64>())
            .collect();
        if kernel_size.len() == 1 {
            kernel_size.push(kernel_size[0]);
        }
        let scale = 1f32 / ((in_channels * kernel_size.iter().product::<i64>()) as f32).sqrt();
        let mut weight_shape = vec![out_channels, in_channels / groups as i64];
        weight_shape.extend(kernel_size);
        Ok(Conv2d {
            stride: stride
                .into_iter()
                .map(|s| s.into().item::<i64>())
                .collect(),
            dilation: dilation
                .into_iter()
                .map(|s| s.into().item::<i64>())
                .collect(),
            groups,
            padding: padding
                .into_iter()
                .map(|s| s.into().item::<i64>())
                .collect(),
            weight: Tensor::uniform(weight_shape.iter().copied(), -scale..scale)?.cast(dtype),
            bias: if bias {
                Some(Tensor::uniform([out_channels], -scale..scale)?.cast(dtype))
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
            self.stride.clone(),
            self.dilation.clone(),
            self.padding.clone(),
        )
    }
}
