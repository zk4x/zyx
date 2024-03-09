use alloc::vec::Vec;
use zyx_core::backend::Backend;
use zyx_core::tensor::Tensor;

/// # Stochastic gradient descent optimizer
pub struct SGD<B: Backend> {
    /// learning rate (default: 0.001)
    pub learning_rate: f32,
    /// momentum factor (default: 0.0)
    pub momentum: f32,
    /// weight decay (L2 penalty) (default: 0.0)
    pub weight_decay: f32,
    /// dampening for momentum (default: 0.0)
    pub dampening: f32,
    /// enables Nesterov momentum (default: false)
    pub nesterov: bool,
    /// maximize the objective with respect to the params, instead of minimizing (default: false)
    pub maximize: bool,
    /// stores momentum, starts empty and will be initialized on demand
    pub bias: Vec<Tensor<B>>,
}

impl<B: Backend> Default for SGD<B> {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
            maximize: false,
            bias: Vec::new(),
        }
    }
}

impl<B: Backend> SGD<B> {
    /// Updates parameters with gradients.
    /// Number of parameters must be the same as number of gradients.
    /// Gradients can be None, those are simply skipped.
    pub fn update<'a>(
        &mut self,
        parameters: impl IntoIterator<Item = &'a mut Tensor<B>>,
        gradients: impl IntoIterator<Item = Option<Tensor<B>>>,
    ) where
        B: 'a,
    {
        let params: Vec<&mut Tensor<B>> = parameters.into_iter().collect();
        let grads: Vec<Option<Tensor<B>>> = gradients.into_iter().collect();

        assert_eq!(
            params.len(),
            grads.len(),
            "Number of parameters != number of gradients."
        );

        for (i, (param, grad)) in params.into_iter().zip(grads).enumerate() {
            if let Some(mut grad) = grad {
                if self.weight_decay != 0.0 {
                    grad = grad + &*param * self.weight_decay;
                }
                if self.momentum != 0.0 {
                    if let Some(bias) = self.bias.get_mut(i) {
                        *bias = &*bias * self.momentum + &grad * (1.0 - self.dampening);
                    } else {
                        self.bias.push(grad.clone());
                    }
                    if self.nesterov {
                        grad = grad + &self.bias[i] * self.momentum;
                    } else {
                        grad = self.bias[i].clone();
                    }
                }
                if self.maximize {
                    // Cast since learning_rate is f32, but parameters can have different precision.
                    // Can this cast be somehow avoided? Is it better to always work with original dtype?
                    *param = (&*param + grad * self.learning_rate).cast(param.dtype());
                } else {
                    *param = (&*param - grad * self.learning_rate).cast(param.dtype());
                }
            }
        }
    }
}
