use zyx::Tensor;

/// Activation
#[derive(Debug, Default)]
pub enum Activation {
    //#[serde(alias = "gelu")]
    /// Gelu
    #[default]
    Gelu,
    //#[serde(alias = "gelu_new")]
    //NewGelu,
    /// Relu
    Relu,
    /// Relu2
    Relu2,
    /// Relu6
    Relu6,
    //Silu,
    /// Sigmoid
    Sigmoid,
    /// Hard sigmoid
    HardSigmoid,
    //Swiglu,
    /// Swish
    Swish,
    //HardSwish,
    /// Elu
    Elu(f64),
    /// Leaky relu
    LeakyRelu(f64),
    //#[serde(alias = "gelu_pytorch_tanh")]
    //GeluPytorchTanh,
}

impl Activation {
    /// Activation forward
    pub fn forward(&self, xs: impl Into<Tensor>) -> Tensor {
        let xs = xs.into();
        match self {
            Self::Gelu => xs.gelu(),
            Self::Relu => xs.relu(),
            Self::Relu2 => xs.relu().pow(2).unwrap(),
            Self::Relu6 => xs.clamp(0f32, 6f32).unwrap(),
            //Self::Silu => xs * xs.silu(),
            Self::Sigmoid => xs.sigmoid(),
            Self::HardSigmoid => xs.hard_sigmoid(),
            //Self::Swiglu => xs.swiglu(),
            Self::Swish => xs.swish(),
            //Self::HardSwish => xs * xs.hard_swish(),
            &Self::Elu(alpha) => xs.elu(alpha),
            &Self::LeakyRelu(negative_slope) => xs.leaky_relu(negative_slope),
            //Self::GeluPytorchTanh => xs.gelu(),
        }
    }
}
