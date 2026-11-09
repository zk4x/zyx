// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// Linear layer
#[derive(Debug, Module)]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all))]
pub struct Linear {
    /// weight
    pub weight: Tensor,
    /// bias
    pub bias: Option<Tensor>,
}

impl Linear {
    /// Initilize linear layer in device self
    pub fn new(
        in_features: i64,
        out_features: i64,
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
    ///
    /// Mirrors llama.cpp: the matrix multiply is the only dtype-changing op.
    /// `ggml_mul_mat` converts its second operand to the weight's type and
    /// always produces `f32` (`ggml.c:3349`), so an `f16` weight turns an `f32`
    /// activation into an `f16` dot and leaves an `f32` result.
    pub fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let x = x.into();
        let x = x.cast(self.weight.dtype()).dot(self.weight.t())?.cast(DType::F32);
        if let Some(bias) = &self.bias {
            return Ok(x + bias.cast(DType::F32));
        }
        Ok(x)
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
