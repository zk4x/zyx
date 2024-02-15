use alloc::vec::Vec;
use zyx_core::backend::Backend;
use zyx_core::tensor::Tensor;


/// # Adaptive momentum estimation optimizer
pub struct Adam<B: Backend> {
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
    /// maximize the objective with respect to the params, instead of minimizing (default: false)
    pub maximize: bool,
    m: Vec<Tensor<B>>,
    v: Vec<Tensor<B>>,
    vm: Vec<Tensor<B>>,
    t: usize,
}

impl<B: Backend> Default for Adam<B> {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            maximize: false,
            m: Vec::new(),
            v: Vec::new(),
            vm: Vec::new(),
            t: 0,
        }
    }
}

impl<B: Backend> Adam<B> {
    pub fn update<'a>(&mut self, parameters: impl IntoIterator<Item = &'a mut Tensor<B>>, gradients: impl IntoIterator<Item = Option<Tensor<B>>>)
    where
        B: 'a
    {
        let params: Vec<&mut Tensor<B>> = parameters.into_iter().collect();
        let grads: Vec<Option<Tensor<B>>> = gradients.into_iter().collect();

        assert_eq!(params.len(), grads.len(), "Number of parameters != number of gradients.");

        for (i, (param, grad)) in params.into_iter().zip(grads).enumerate() {
            if let Some(mut grad) = grad {
                if self.maximize {
                    grad = -grad;
                }
                if self.weight_decay != 0.0 {
                    grad = grad + &*param * self.weight_decay;
                }
                if let Some(m) = self.m.get_mut(i) {
                    *m = &*m * self.betas.0 + &grad * (1.0 - self.betas.0);
                } else {
                    self.m.push(&grad * (1.0 - self.betas.0));
                }
                if let Some(v) = self.m.get_mut(i) {
                    *v = &*v * self.betas.1 + &grad * &grad * (1.0 - self.betas.1);
                } else {
                    self.v.push(&grad * &grad * (1.0 - self.betas.1));
                }
                let mh = &self.m[i] / (1.0 - self.betas.0.powi(self.t as i32));
                let vh = &self.v[i] / (1.0 - self.betas.1.powi(self.t as i32));
                if self.amsgrad {
                    if let Some(vm) = self.vm.get_mut(i) {
                        *vm = vm.cmplt(&vh).where_(vh, &*vm);
                    } else {
                        self.vm.push(vh);
                    }
                    *param = &*param - mh / ((self.vm[i].sqrt() + self.eps) * self.learning_rate);
                } else {
                    *param = &*param - mh / ((vh.sqrt() + self.eps) * self.learning_rate);
                }
            }
        }
    }
}
