// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// Batch norm
///
/// By default this module has learnable affine parameters,
/// set weight and bias to None to remove them.
#[derive(Debug, Module)]
#[cfg_attr(feature = "py", pyo3::pyclass)]
pub struct BatchNorm {
    /// a value added to the denominator for numerical stability. Default: 1e-5
    pub eps: f32,
    /// the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
    pub momentum: f32,
    /// When set to True, this module tracks the running mean and variance, and when set to False, this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None. When these buffers are None, this module always uses batch statistics
    pub track_running_stats: bool,
    /// weight
    pub weight: Option<Tensor>,
    /// bias
    pub bias: Option<Tensor>,
    /// weight
    pub running_mean: Tensor,
    /// bias
    pub running_var: Tensor,
    /// Number of tracked batches
    pub num_batches_tracked: Tensor,
}

impl BatchNorm {
    /// Initilize layer_norm layer in device self
    pub fn new(self, num_features: i64, dtype: DType) -> BatchNorm {
        BatchNorm {
            eps: 1e-5,
            momentum: 0.1,
            track_running_stats: true,
            weight: Some(Tensor::ones([num_features], dtype)),
            bias: Some(Tensor::zeros([num_features], dtype)),
            running_mean: Tensor::zeros([num_features], dtype),
            running_var: Tensor::ones([num_features], dtype),
            num_batches_tracked: Tensor::zeros([1], dtype),
        }
    }

    /// Forward function for layer_norm.
    pub fn forward(&mut self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let batch_mean;
        let batch_invstd;

        let x = x.into();

        if Tensor::training() {
            batch_mean = x.mean([0, 2, 3])?;
            let shape4d = vec![1i64.into(), batch_mean.numel(), 1i64.into(), 1i64.into()];
            let y = &x - batch_mean.reshape(shape4d.clone())?;
            let batch_var = (&y * &y).mean([0, 2, 3])?;
            batch_invstd = (self
                .running_var
                .reshape(shape4d.clone())?
                .expand(x.shape())?
                + self.eps)
                .rsqrt();

            if self.track_running_stats {
                self.running_mean =
                    &self.running_mean * (1.0 - self.momentum) + &batch_mean * self.momentum;
                let n = y.numel();
                let bessel = &n / (&n - &y.shape()[1]);
                self.running_var = &self.running_var * (1.0 - self.momentum)
                    + batch_var * self.momentum * bessel;
                self.num_batches_tracked = &self.num_batches_tracked + 1;
            }
        } else {
            batch_mean = self.running_mean.clone();
            batch_invstd = (self
                .running_var
                .reshape(vec![1i64.into(), self.running_var.numel(), 1i64.into(), 1i64.into()])?
                .expand(x.shape())?
                + self.eps)
                .rsqrt()
        }

        let shape = vec![1i64.into(), batch_mean.numel(), 1i64.into(), 1i64.into()];
        let mut x = x - batch_mean.reshape(shape.clone())?;
        if let Some(weight) = &self.weight {
            x = weight.reshape(shape.clone())? * x;
        }
        x = x * if batch_invstd.rank() == 1 {
            batch_invstd.reshape(shape.clone())?
        } else {
            batch_invstd
        };
        if let Some(bias) = &self.bias {
            Ok(x + bias.reshape(shape)?)
        } else {
            Ok(x)
        }
    }
}
