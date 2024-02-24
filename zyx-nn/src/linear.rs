use zyx_core::backend::Backend;
use zyx_core::dtype::DType;
use zyx_core::tensor::{IntoTensor, Tensor};

/// Linear layer
pub struct Linear<B: Backend> {
    /// weight
    pub weight: Tensor<B>,
    /// bias
    pub bias: Option<Tensor<B>>,
}

/// Initilization trait for linear layer
pub trait LinearInit: Backend {
    /// Initilize linear layer in device self
    fn linear(self, in_features: usize, out_features: usize) -> Linear<Self> {
        Linear {
            weight: self.randn([in_features, out_features], DType::F32),
            bias: Some(self.randn([out_features], DType::F32)),
        }
    }
}

impl<B: Backend> LinearInit for B {}

impl<'a, B: Backend> IntoIterator for &'a Linear<B> {
    type Item = &'a Tensor<B>;
    type IntoIter = alloc::vec::IntoIter<&'a Tensor<B>>;
    fn into_iter(self) -> Self::IntoIter {
        if let Some(bias) = &self.bias {
            alloc::vec![&self.weight, bias].into_iter()
        } else {
            alloc::vec![&self.weight].into_iter()
        }
    }
}

impl<'a, B: Backend> IntoIterator for &'a mut Linear<B> {
    type Item = &'a mut Tensor<B>;
    type IntoIter = alloc::vec::IntoIter<&'a mut Tensor<B>>;
    fn into_iter(self) -> Self::IntoIter {
        if let Some(bias) = &mut self.bias {
            alloc::vec![&mut self.weight, bias].into_iter()
        } else {
            alloc::vec![&mut self.weight].into_iter()
        }
    }
}

impl<B: Backend> Linear<B> {
    /// Forward function for linear.
    /// Calculates x.dot(&self.weight) + self.bias
    pub fn forward(&self, x: impl IntoTensor<B>) -> Tensor<B> {
        let x = self.weight.backend().tensor(x);
        let x = x.dot(&self.weight);
        if let Some(bias) = &self.bias {
            return x + bias;
        }
        return x;
    }
}
