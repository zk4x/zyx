// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// Embedding layer
#[derive(Debug, Module)]
#[cfg_attr(feature = "py", pyo3::pyclass)]
pub struct Embedding {
    /// Vocabulary size
    pub vocab_size: Tensor,
    /// Embedding size
    pub embed_size: Tensor,
    /// Weight
    pub weight: Tensor,
    /// Arange
    pub arange: Tensor,
}

impl Embedding {
    /// new embedding layer
    pub fn new(vocab_size: i64, embed_size: i64, dtype: DType) -> Result<Embedding, ZyxError> {
        let vocab_size_t: Tensor = vocab_size.into();
        let embed_size_t: Tensor = embed_size.into();
        let one: Tensor = 1i64.into();
        let weight = Tensor::glorot_uniform([vocab_size, embed_size], dtype)?
            .reshape([one.clone(), one.clone(), vocab_size_t.clone(), embed_size_t.clone()])?;
        let arange = Tensor::arange(0, vocab_size, 1)?
            .reshape([one.clone(), one, vocab_size_t.clone(), 1i64.into()])?
            .cast(dtype);
        Ok(Embedding {
            vocab_size: vocab_size_t,
            embed_size: embed_size_t,
            weight,
            arange,
        })
    }

    /// Initialize embedding using only weight
    pub fn from_params(weight: Tensor) -> Result<Embedding, ZyxError> {
        let sh = weight.symbolic_shape();
        assert_eq!(sh.len(), 2);
        let vocab_size = sh[0].clone();
        let embed_size = sh[1].clone();
        Ok(Embedding {
            vocab_size: vocab_size.clone(),
            embed_size: embed_size.clone(),
            arange: Tensor::arange(0, vocab_size.item::<i64>(), 1)?
                .reshape([1i64.into(), 1i64.into(), vocab_size, 1i64.into()])?
                .cast(weight.dtype()),
            weight,
        })
    }

    /// Forward embedding layer
    pub fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let x: Tensor = x.into();
        let x_sh = x.symbolic_shape();
        if x.numel().item::<i64>() == 0 {
            let shape: Vec<Tensor> = x_sh
                .iter()
                .cloned()
                .chain(std::iter::once(self.embed_size.clone()))
                .collect();
            return Ok(Tensor::zeros(shape, x.dtype()));
        }
        let xdt = x.dtype();
        let wdt = self.weight.dtype();
        if xdt != wdt {
            return Err(ZyxError::DTypeError(
                format!("Embedding::forward input x has dtype {xdt} but weight has dtype {wdt}")
                    .into(),
            ));
        }
        let one: Tensor = 1i64.into();
        let big_shp: Vec<Tensor> = x_sh
            .iter()
            .cloned()
            .chain([self.vocab_size.clone(), self.embed_size.clone()])
            .collect();
        let arange = self.arange.expand(big_shp.clone())?;
        let reshape_shape: Vec<Tensor> = x_sh
            .into_iter()
            .chain(std::iter::once(one.clone()))
            .chain(std::iter::once(one))
            .collect();
        let idx = x.reshape(reshape_shape)?.expand(big_shp.clone())?;
        let vals = self.weight.expand(big_shp)?;
        (arange.equal(idx)?.cast(xdt) * vals).sum([2])
    }
}
