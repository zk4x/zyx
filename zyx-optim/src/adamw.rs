use zyx::Tensor;
use zyx_derive::Module;

/// # Adaptive momentum estimation optimizer
#[derive(Module)]
pub struct AdamW {
    /// learning rate (default: 1e-3)
    pub learning_rate: f32,
    /// coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
    pub betas: (f32, f32),
    /// term added to the denominator to improve numerical stability (default: 1e-8)
    pub eps: f32,
    /// weight decay (L2 penalty) (default: 0)
    pub weight_decay: f32,
    /// whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond (default: false)
    pub amsgrad: bool,
    /// m
    pub m: Vec<Tensor>,
    /// v
    pub v: Vec<Tensor>,
    /// vm
    pub vm: Vec<Tensor>,
    /// t
    pub t: usize,
}

impl Default for AdamW {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            m: Vec::new(),
            v: Vec::new(),
            vm: Vec::new(),
            t: 0,
        }
    }
}

impl AdamW {
    /// Updates parameters with gradients.
    /// Number of parameters must be the same as number of gradients.
    /// Gradients can be None, those are simply skipped.
    pub fn update<'a>(
        &mut self,
        parameters: impl IntoIterator<Item = &'a mut Tensor>,
        gradients: impl IntoIterator<Item = Option<Tensor>>,
    ) {
        use zyx::Scalar;
        for (i, (param, grad)) in parameters.into_iter().zip(gradients).enumerate() {
            let Some(grad) = grad else {
                // Initialize moment estimates for new params (lazy)
                if self.m.len() <= i {
                    self.m.push(Tensor::zeros_like(&*param));
                    self.v.push(Tensor::zeros_like(&*param));
                    if self.amsgrad {
                        self.vm.push(Tensor::zeros_like(&*param));
                    }
                }
                continue;
            };

            // Update biased first moment estimate
            if let Some(m) = self.m.get_mut(i) {
                *m = &*m * self.betas.0 + &grad * (1.0 - self.betas.0);
            } else {
                self.m.push(&grad * (1.0 - self.betas.0));
            }

            // Update biased second moment estimate
            if let Some(v) = self.v.get_mut(i) {
                *v = &*v * self.betas.1 + &grad * &grad * (1.0 - self.betas.1);
            } else {
                self.v.push(&grad * &grad * (1.0 - self.betas.1));
            }

            // Compute bias-corrected moments
            let mh = &self.m[i] / (1.0 - self.betas.0.pow(self.t as f32));
            let vh = &self.v[i] / (1.0 - self.betas.1.pow(self.t as f32));

            if self.amsgrad {
                if let Some(vm) = self.vm.get_mut(i) {
                    *vm = vm.cmplt(&vh).unwrap().where_(vh, &*vm).unwrap();
                } else {
                    self.vm.push(vh);
                }
                // Parameter update with AMSGrad max
                *param = (&*param - mh / ((self.vm[i].sqrt() + self.eps) * self.learning_rate))
                    .cast(param.dtype());
            } else {
                // Parameter update standard AdamW
                *param = (&*param - mh / ((vh.sqrt() + self.eps) * self.learning_rate))
                    .cast(param.dtype());
            }

            // Decoupled weight decay step applied directly to parameter
            if self.weight_decay != 0.0 {
                *param = &*param * (1.0 - self.learning_rate * self.weight_decay);
            }
        }
        self.t += 1;
    }
}
