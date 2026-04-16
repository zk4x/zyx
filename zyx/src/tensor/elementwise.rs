// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::Tensor;

impl Tensor {
    fn polyN(x: Tensor, coeffs: [f32; 5]) -> Tensor {
        let mut result: Tensor = 0.0f32.into();
        for c in coeffs {
            result = result * x.clone() + Tensor::from(c);
        }
        result
    }

    /// Returns the sign of each element: -1 if negative, 1 if positive, 0 if zero.
    #[must_use]
    pub fn sign(&self) -> Tensor {
        let zero = Tensor::zeros_like(self.clone());
        let neg_one: Tensor = (-1_i32).into();
        let pos_one: Tensor = (1_i32).into();
        let is_neg = self.clone().cmplt(zero.clone()).unwrap();
        let result = is_neg.where_(&neg_one, &pos_one).unwrap();
        self.nonzero().where_(&result, &zero).unwrap()
    }

    /// Applies ReLU6 activation: min(max(x, 0), 6)
    #[must_use]
    pub fn relu6(&self) -> Tensor {
        self.relu().clamp(Tensor::from(0f32), Tensor::from(6f32)).unwrap()
    }

    /// Applies softsign: x / (1 + |x|)
    #[must_use]
    pub fn softsign(&self) -> Tensor {
        let x = self.clone();
        let abs = x.clone().abs();
        x / (Tensor::from(1f32) + abs)
    }

    /// Applies hardtanh: clamp(x, -1, 1)
    #[must_use]
    pub fn hardtanh(&self) -> Tensor {
        self.clone().clamp(Tensor::from(-1f32), Tensor::from(1f32)).unwrap()
    }

    /// Linear interpolation: self + weight * (end - self)
    #[must_use]
    pub fn lerp(&self, end: &Tensor, weight: impl Into<Tensor>) -> Tensor {
        let w = weight.into();
        self.clone() + w * (end.clone() - self.clone())
    }

    /// Returns true where elements are finite (not inf or nan)
    #[must_use]
    pub fn isfinite(&self) -> Tensor {
        let inf = self.isinf();
        let nan = self.isnan();
        let dtype = self.dtype();
        Tensor::from(1f32).cast(dtype) - (inf + nan)
    }

    /// Parameterized ReLU: x * (x > 0) + alpha * x * (x <= 0)
    #[must_use]
    pub fn prelu(&self, alpha: impl Into<Tensor>) -> Tensor {
        let alpha = alpha.into();
        let zero = Tensor::zeros_like(self.clone());
        let pos = self.clone().cmpgt(zero.clone()).unwrap();
        pos.where_(self, alpha * self.clone()).unwrap()
    }

    /// Error function
    #[must_use]
    pub fn erf(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        let one: Tensor = 1.0f32.into();
        let t = one.clone() / (one.clone() + 0.3275911f32 * x.clone().abs());
        let coeffs = [
            1.061405429f32,
            -1.453152027f32,
            1.421413741f32,
            -0.284496736f32,
            0.254829592f32,
        ];
        let poly = Self::polyN(t.clone(), coeffs);
        x.sign() * (one - t * poly * (-x.clone() * x.clone()).exp())
    }

    /// exp(x) - 1
    #[must_use]
    pub fn expm1(&self) -> Tensor {
        self.exp() - Tensor::from(1f32)
    }

    /// log(1 + x)
    #[must_use]
    pub fn log1p(&self) -> Tensor {
        (self + Tensor::from(1f32)).log(Tensor::from(std::f32::consts::E))
    }

    /// log(exp(self) + exp(other))
    #[must_use]
    pub fn logaddexp(&self, other: &Tensor) -> Tensor {
        let m = self.clone().maximum(other.clone()).unwrap();
        ((self.clone() - m.clone()).exp() + (other.clone() - m.clone()).exp()).log(Tensor::from(std::f32::consts::E)) + m
    }
}
