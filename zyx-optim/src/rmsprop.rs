// In your code base or inside zyx_optim crate
use zyx::Tensor;

/// RMSProp optimizer for adaptive learning rate training.
pub struct RMSprop {
    /// Step size multiplier
    pub learning_rate: f32,
    /// Controls how quickly the cache forgets old gradients
    pub alpha: f32,
    /// Small constant to avoid division by zero
    pub eps: f32,
    /// Momentum
    pub momentum: f32,
    /// Centered
    pub centered: bool,
    /// Weight decay
    pub weight_decay: f32,
    /// t
    pub t: usize,
    /// Squared grad avg
    buffer: Vec<Tensor>,
    /// Momentum buffer
    momentum_buf: Vec<Tensor>,
    /// Gradient average for centered variant
    grad_avg: Vec<Tensor>,
}

impl Default for RMSprop {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            alpha: 0.99,
            eps: 1e-8,
            momentum: 0.0,
            centered: false,
            weight_decay: 0.0,
            t: 0,
            buffer: Vec::new(),
            momentum_buf: Vec::new(),
            grad_avg: Vec::new(),
        }
    }
}

impl RMSprop {
    /// update
    pub fn update<'a>(
        &mut self,
        parameters: impl IntoIterator<Item = &'a mut Tensor>,
        gradients: impl IntoIterator<Item = Option<Tensor>>,
    ) {
        for (i, (param, grad)) in parameters.into_iter().zip(gradients).enumerate() {
            let Some(grad) = grad else {
                // Lazy init for new parameters
                if self.buffer.len() <= i {
                    self.buffer.push(Tensor::zeros_like(&*param));
                    self.momentum_buf.push(Tensor::zeros_like(&*param));
                    if self.centered {
                        self.grad_avg.push(Tensor::zeros_like(&*param));
                    }
                }
                continue;
            };

            // Lazy init state if missing
            if self.buffer.len() <= i {
                self.buffer.push(&grad * &grad * (1.0 - self.alpha));
                self.momentum_buf.push(Tensor::zeros_like(&*param));
                if self.centered {
                    self.grad_avg.push(&grad * (1.0 - self.alpha));
                }
            }

            // Exponential moving average of squared gradients
            self.buffer[i] = &self.buffer[i] * self.alpha + &grad * &grad * (1.0 - self.alpha);

            let denom = if self.centered {
                // Centered RMSProp: subtract moving avg of grad
                self.grad_avg[i] = &self.grad_avg[i] * self.alpha + &grad * (1.0 - self.alpha);
                let avg = &self.grad_avg[i];
                (&self.buffer[i] - avg * avg).relu().sqrt() + self.eps
            } else {
                self.buffer[i].sqrt() + self.eps
            };

            let update = &grad / denom * self.learning_rate;

            if self.momentum > 0.0 {
                self.momentum_buf[i] = &self.momentum_buf[i] * self.momentum + &update;
                *param = &*param - &self.momentum_buf[i];
            } else {
                *param = &*param - update;
            }

            if self.weight_decay > 0.0 {
                *param = &*param * (1.0 - self.learning_rate * self.weight_decay);
            }
        }

        self.t += 1;
    }
}
