// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Quantized projection weights.

use zyx::{DType, QDType, Tensor, ZyxError};

/// One PQ2_0 projection on device: the raw `[num_blocks, 34]` U8 tensor plus
/// the logical `[rows, cols]` shape resolved once at load.
pub struct QWeight {
    /// Raw gguf super-blocks.
    pub blocks: Tensor,
    /// Logical row count (output features).
    pub rows: i64,
    /// Logical column count (input features).
    pub cols: i64,
    /// `true` when the activation must be rotated before this projection.
    pub folded: bool,
}

impl QWeight {
    /// Validates the raw block tensor and records its logical shape.
    ///
    /// # Errors
    /// Fails when the tensor is not `[_, 34]` U8 (a PQ2_0 super-block), or
    /// when the block count does not match `rows * cols`.
    pub fn new(blocks: Tensor, rows: i64, cols: i64) -> Result<Self, ZyxError> {
        if blocks.dtype() != DType::U8 {
            return Err(ZyxError::dtype_error(
                format!("PQ2_0 weight must be raw U8 blocks, got {}", blocks.dtype()).into(),
            ));
        }
        let shape = blocks.resolve_shape();
        if shape.len() != 2 || shape[1] != 34 {
            return Err(ZyxError::shape_error(
                format!("PQ2_0 weight must be [num_blocks, 34], got {shape:?}").into(),
            ));
        }
        if shape[0] * 128 != rows * cols {
            return Err(ZyxError::shape_error(
                format!(
                    "PQ2_0 weight has {} blocks ({} values) but logical shape is [{rows}, {cols}]",
                    shape[0],
                    shape[0] * 128
                )
                .into(),
            ));
        }
        Ok(Self { blocks, rows, cols, folded: false })
    }

    /// Dequantizes to a dense `[rows, cols]` tensor of `dtype`.
    ///
    /// # Errors
    /// Propagates the dequantizer's dtype/shape errors.
    pub fn dequantize(&self, dtype: DType) -> Result<Tensor, ZyxError> {
        self.blocks
            .dequantize(QDType::PQ2_0, dtype)?
            .reshape([self.rows, self.cols])
    }
}
