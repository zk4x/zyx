use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// Linear layer
#[derive(Debug, Module)]
pub struct Linear {
    /// weight
    pub weight: Tensor,
    /// bias
    pub bias: Option<Tensor>,
}

impl Linear {
    /// Initilize linear layer in device self
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        dtype: DType,
    ) -> Result<Linear, ZyxError> {
        let l = -(1.0 / (in_features as f32)).sqrt();
        let u = (1.0 / (in_features as f32)).sqrt();
        Ok(Linear {
            weight: Tensor::uniform([out_features, in_features], l..u)?.cast(dtype),
            bias: if bias {
                Some(Tensor::uniform([out_features], l..u)?.cast(dtype))
            } else {
                None
            },
        })
    }

    /// Forward function for linear.
    /// Calculates x.dot(&self.weight) + self.bias
    pub fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let x = x.into().dot(self.weight.t())?;
        if let Some(bias) = &self.bias {
            return Ok(x + bias);
        }
        return Ok(x);
    }
}

#[test]
fn linear() -> Result<(), ZyxError> {
    let l0 = Linear::new(4, 16, true, DType::F32)?;
    println!("{}\n{}", l0.weight, l0.bias.as_ref().unwrap());
    let x = Tensor::randn([8, 4], DType::F32)?;
    let y = l0.forward(x)?.relu();

    println!("{y}");

    Ok(())
}
