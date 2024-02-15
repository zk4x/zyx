use alloc::vec::Vec;
use zyx_core::backend::Backend;
use zyx_core::tensor::Tensor;

/// # Stochastic gradient descent optimizer
///
/// ## Parameters
/// - learning_rate – learning rate (default: 0.001)
/// - momentum – momentum factor (default: 0.0)
/// - weight_decay – weight decay (L2 penalty) (default: 0.0)
/// - dampening – dampening for momentum (default: 0.0)
/// - nesterov – enables Nesterov momentum (default: false)
/// - maximize – maximize the objective with respect to the params, instead of minimizing (default: false)
pub struct SGD<B: Backend> {
    pub learning_rate: f32,
    pub momentum: f32,
    pub dampening: f32,
    pub weight_decay: f32,
    pub nesterov: bool,
    pub maximize: bool,
    pub bias: Option<Vec<Tensor<B>>>,
}

impl<B: Backend> Default for SGD<B> {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            maximize: false,
            bias: None,
        }
    }
}

impl<B: Backend> SGD<B> {
    /// Updates parameters with gradients.
    /// Number of parameters must be the same as number of gradients.
    /// Gradients can be None, those are simply skipped.
    pub fn update<'a>(&mut self, parameters: impl IntoIterator<Item = &'a mut Tensor<B>>, gradients: impl IntoIterator<Item = Option<Tensor<B>>>)
    where
        B: 'a
    {
        let params: Vec<&mut Tensor<B>> = parameters.into_iter().collect();
        let grads: Vec<Option<Tensor<B>>> = gradients.into_iter().collect();

        assert_eq!(params.len(), grads.len(), "Number of parameters != number of gradients.");

        for (i, (param, grad)) in params.into_iter().zip(grads).enumerate() {
            if let Some(mut grad) = grad {
                if self.weight_decay != 0.0 {
                    grad = grad + &*param * self.weight_decay;
                }
                if self.momentum != 0.0 {
                    if let Some(bias) = &mut self.bias {
                        if let Some(bias) = bias.get_mut(i) {
                            *bias = &*bias * self.momentum + &grad * (1.0 - self.dampening);
                        } else {
                            bias.push(grad.clone());
                        }
                    } else {
                        self.bias = Some(alloc::vec![grad.clone()]);
                    }
                    if self.nesterov {
                        grad = grad + &self.bias.as_ref().unwrap()[i] * self.momentum;
                    } else {
                        grad = self.bias.as_ref().unwrap()[i].clone();
                    }
                }
                if self.maximize {
                    *param = &*param + grad * self.learning_rate;
                } else {
                    *param = &*param - grad * self.learning_rate;
                }
            }
        }
    }
}
