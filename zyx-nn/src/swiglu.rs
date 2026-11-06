// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! SwiGLU feed-forward block: `down(silu(gate(x)) * up(x))`.

use crate::Linear;
use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// Gated feed-forward network used by llama-family models.
#[derive(Debug, Module)]
#[cfg_attr(feature = "py", pyo3::pyclass)]
pub struct SwiGlu {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGlu {
    /// Builds the block with random projections.
    ///
    /// # Errors
    ///
    /// Returns a shape error if the dimensions are invalid.
    pub fn new(n_embd: i64, intermediate: i64, dtype: DType) -> Result<SwiGlu, ZyxError> {
        Ok(SwiGlu {
            gate_proj: Linear::new(n_embd, intermediate, false, dtype)?,
            up_proj: Linear::new(n_embd, intermediate, false, dtype)?,
            down_proj: Linear::new(intermediate, n_embd, false, dtype)?,
        })
    }

    /// Forward pass, `[batch, seq, n_embd]` in and out.
    ///
    /// # Errors
    ///
    /// Returns a shape error if `x` does not have `n_embd` as its last dim.
    pub fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let x = x.into();
        let gate = self.gate_proj.forward(x.clone())?.swish();
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(gate * up)
    }
}
