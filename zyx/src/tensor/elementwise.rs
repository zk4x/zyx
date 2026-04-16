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

    /// Absolute value
    #[must_use]
    pub fn abs(&self) -> Tensor {
        self.relu() + (-self).relu()
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
}
