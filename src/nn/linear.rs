use crate::{
    context::Context,
    nn::Module,
    parameters::{IntoParameters, Parameters},
    tensor::Tensor,
};

/// Linear layer
///
/// Linear layer contains weight and bias.
/// It applies linear transformation on input.
/// y = weight.dot(x) + bias
/// It currently accepts and returns transposed tensors
#[derive(Clone, Debug)]
pub struct Linear {
    /// Weight of linear layer
    pub weight: Tensor,
    /// Bias of linear layer
    pub bias: Option<Tensor>,
}

impl Module for Linear {
    #[must_use]
    fn forward(&self, x: &Tensor) -> Tensor {
        if let Some(bias) = &self.bias {
            self.weight.t_dot(x) + bias
        } else {
            self.weight.t_dot(x)
        }
    }

    #[must_use]
    fn parameters(&mut self) -> Parameters<'_> {
        if let Some(b) = &mut self.bias {
            [&mut self.weight, b].into_parameters()
        } else {
            [&mut self.weight].into_parameters()
        }
    }
}

impl Context {
    /// Create new linear layer
    ///
    /// Bias is by default set to true.
    /// Both weight and bias are initialized from uniform distribution
    /// from range -k..k where k = `sqrt(1/in_features)`
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    #[cfg(feature = "rand")]
    pub fn linear(&mut self, in_features: usize, out_features: usize) -> Linear {
        // TODO add scaling
        let k = crate::libm::powf(1. / in_features as f32, 0.5);
        Linear {
            weight: self
                .uniform((in_features, out_features), -k..k)
                .set_label("linear w"),
            bias: Some(self.uniform((out_features, 1), -k..k).set_label("linear b")),
        }
    }
}

impl Linear {
    /// Set gain for linear layer.
    /// This function multiplies weight by gain.
    #[allow(clippy::return_self_not_must_use)]
    pub fn set_gain(&mut self, gain: f32) -> Self {
        self.weight = &self.weight * gain;
        self.clone()
    }

    /// Set bias for linear layer
    #[allow(clippy::return_self_not_must_use)]
    #[allow(clippy::missing_panics_doc)]
    #[allow(clippy::cast_precision_loss)]
    #[cfg(feature = "rand")]
    pub fn set_bias(&mut self, bias: bool) -> Self {
        if bias {
            if self.bias.is_none() {
                let [in_features, out_features]: [usize; 2] =
                    self.weight.shape().try_into().unwrap();
                let k = crate::libm::powf(1. / in_features as f32, 0.5);
                self.bias = Some(
                    self.weight
                        .context()
                        .uniform((out_features, 1), -k..k)
                        .set_label("linear b"),
                );
            }
        } else {
            self.bias = None;
        }
        self.clone()
    }
}
