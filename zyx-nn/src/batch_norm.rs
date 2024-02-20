use zyx_core::backend::Backend;
use zyx_core::dtype::DType;
use zyx_core::shape::Shape;
use zyx_core::tensor::{IntoTensor, Tensor};

/// Batch norm
///
/// By default this module has learnable affine parameters,
/// set weight and bias to None to remove them.
pub struct BatchNorm<B: Backend> {
    /// a value added to the denominator for numerical stability. Default: 1e-5
    pub eps: f32,
    /// the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
    pub momentum: f32,
    /// When set to True, this module tracks the running mean and variance, and when set to False, this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None. When these buffers are None, this module always uses batch statistics
    pub track_running_stats: bool,
    /// Is it training or inference? (for running mean and var)
    pub training: bool,
    /// weight
    pub weight: Option<Tensor<B>>,
    /// bias
    pub bias: Option<Tensor<B>>,
    /// weight
    pub running_mean: Tensor<B>,
    /// bias
    pub running_var: Tensor<B>,
    /// Number of tracked batches
    pub num_batches_tracked: Tensor<B>,
}

/// Initilization trait for batch_norm layer
pub trait BatchNormInit: Backend {
    /// Initilize layer_norm layer in device self
    fn batch_norm(self, num_features: usize) -> BatchNorm<Self> {
        BatchNorm {
            eps: 1e-5,
            momentum: 0.1,
            track_running_stats: true,
            weight: Some(self.ones(num_features, DType::F32)),
            bias: Some(self.zeros(num_features, DType::F32)),
            running_mean: self.zeros(num_features, DType::F32),
            running_var: self.ones(num_features, DType::F32),
            training: true,
            num_batches_tracked: self.zeros(1, DType::F32),
        }
    }
}

impl<B: Backend> BatchNormInit for B {}

impl<'a, B: Backend> IntoIterator for &'a BatchNorm<B> {
    type Item = &'a Tensor<B>;
    type IntoIter = alloc::vec::IntoIter<&'a Tensor<B>>;
    fn into_iter(self) -> Self::IntoIter {
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => alloc::vec![w, b].into_iter(),
            (Some(w), None) => alloc::vec![w].into_iter(),
            (None, Some(b)) => alloc::vec![b].into_iter(),
            (None, None) => alloc::vec![].into_iter(),
        }
    }
}

impl<'a, B: Backend> IntoIterator for &'a mut BatchNorm<B> {
    type Item = &'a mut Tensor<B>;
    type IntoIter = alloc::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        match (&mut self.weight, &mut self.bias) {
            (Some(w), Some(b)) => alloc::vec![w, b].into_iter(),
            (Some(w), None) => alloc::vec![w].into_iter(),
            (None, Some(b)) => alloc::vec![b].into_iter(),
            (None, None) => alloc::vec![].into_iter(),
        }
    }
}

impl<B: Backend> BatchNorm<B> {
    /// Forward function for layer_norm.
    pub fn forward(&mut self, x: &Tensor<B>) -> Tensor<B> {
        let batch_mean;
        let batch_invstd;

        if self.training {
            batch_mean = x.mean([0, 2, 3]);
            let y = (x - batch_mean.reshape([1, batch_mean.numel(), 1, 1]));
            let batch_var = (&y * &y).mean([0, 2, 3]);
            batch_invstd = (self
                .running_var
                .reshape([1, self.running_var.numel(), 1, 1])
                .expand(x.shape())
                + self.eps)
                .rsqrt();

            if self.track_running_stats {
                self.running_mean =
                    &self.running_mean * (1.0 - self.momentum) + &batch_mean * self.momentum;
                self.running_var = &self.running_var * (1.0 - self.momentum)
                    + batch_var * self.momentum * y.numel() as f32
                        / (y.numel() - y.shape()[1]) as f32;
                self.num_batches_tracked = &self.num_batches_tracked + 1;
            }
        } else {
            batch_mean = self.running_mean.clone();
            batch_invstd = (self
                .running_var
                .reshape([1, self.running_var.numel(), 1, 1])
                .expand(x.shape())
                + self.eps)
                .rsqrt()
        }

        let shape = [1, batch_mean.numel(), 1, 1];
        let mut x = x - batch_mean.reshape(shape);
        if let Some(weight) = &self.weight {
            x = weight.reshape(shape) * x;
        }
        x = x * if batch_invstd.rank() == 1 {
            batch_invstd.reshape(shape)
        } else {
            batch_invstd
        };
        if let Some(bias) = &self.bias {
            x + bias.reshape(shape)
        } else {
            x
        }
    }
}
