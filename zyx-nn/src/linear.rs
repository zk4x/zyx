use zyx::{DType, Tensor};

/// Linear layer
pub struct Linear {
    /// weight
    pub weight: Tensor,
    /// bias
    pub bias: Option<Tensor>,
}

impl Linear {
    /// Initilize linear layer in device self
    pub fn new(in_features: usize, out_features: usize, dtype: DType) -> Linear {
        let l = -(1.0/(in_features as f32)).sqrt();
        let u = (1.0/(in_features as f32)).sqrt();
        Linear {
            weight: Tensor::uniform([in_features, out_features], l..u).cast(dtype),
            bias: Some(Tensor::uniform([out_features], l..u).cast(dtype)),
        }
    }

    /// Forward function for linear.
    /// Calculates x.dot(&self.weight) + self.bias
    pub fn forward(&self, x: impl Into<Tensor>) -> Tensor {
        let x = x.into().dot(&self.weight);
        if let Some(bias) = &self.bias {
            return x + bias;
        }
        return x;
    }
}

impl<'a> IntoIterator for &'a Linear {
    type Item = &'a Tensor;
    type IntoIter = alloc::vec::IntoIter<&'a Tensor>;
    fn into_iter(self) -> Self::IntoIter {
        if let Some(bias) = &self.bias {
            alloc::vec![&self.weight, bias].into_iter()
        } else {
            alloc::vec![&self.weight].into_iter()
        }
    }
}

impl<'a> IntoIterator for &'a mut Linear {
    type Item = &'a mut Tensor;
    type IntoIter = alloc::vec::IntoIter<&'a mut Tensor>;
    fn into_iter(self) -> Self::IntoIter {
        if let Some(bias) = &mut self.bias {
            alloc::vec![&mut self.weight, bias].into_iter()
        } else {
            alloc::vec![&mut self.weight].into_iter()
        }
    }
}
