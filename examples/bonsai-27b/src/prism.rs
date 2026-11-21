// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Prism's rotation-folded weights.
//!
//! The checkpoint stores most matmul weights multiplied by a blockwise
//! normalized Walsh-Hadamard matrix (and a per-input-dimension sign vector).
//! llama.cpp restores the product on the activation side: before a folded
//! weight is used, the activation becomes `H (s * x)`; the embedding table is
//! stored pre-rotated and is restored with `s * (H z)` right after the row
//! lookup. This module implements exactly those two transforms with ordinary
//! tensor ops.

use std::collections::{HashMap, HashSet};

use zyx::{Dev, GGUFMetadata, Tensor, ZyxError};

/// Rotation/sign data for one checkpoint.
pub struct Prism {
    /// Rotation block width (`prism.hadamard.block_size`).
    pub block_size: i64,
    /// `true` when `ssm_out` is kept in grouped V-head order.
    pub gdn_v_grouped: bool,
    /// Normalized Sylvester-Walsh-Hadamard matrix, `[block_size, block_size]`.
    rotation: Tensor,
    /// Sign vector per input width.
    signs: HashMap<i64, Tensor>,
    /// Weight names whose input activation must be rotated.
    folded: HashSet<String>,
    /// Lookup tables whose rows must be restored with the inverse rotation.
    inverse: HashSet<String>,
}

impl Prism {
    /// Reads the `prism.hadamard.*` metadata. A checkpoint without the block
    /// loads with an empty `Prism` (no transforms, no inverse tables).
    pub fn from_metadata(metadata: &GGUFMetadata) -> Result<Self, ZyxError> {
        let folded: HashSet<String> = if metadata.contains_key("prism.hadamard.weight_names") {
            metadata.strings("prism.hadamard.weight_names")?.into_iter().map(str::to_string).collect()
        } else {
            HashSet::new()
        };
        let inverse: HashSet<String> = if metadata.contains_key("prism.hadamard.inverse_weight_names") {
            metadata.strings("prism.hadamard.inverse_weight_names")?.into_iter().map(str::to_string).collect()
        } else {
            HashSet::new()
        };
        let block_size = metadata.opt_int("prism.hadamard.block_size")?.unwrap_or(0);
        let gdn_v_grouped = metadata.opt_bool("prism.hadamard.gdn_v_grouped")?.unwrap_or(false);

        if block_size == 0 {
            if !folded.is_empty() || !inverse.is_empty() {
                return Err(ZyxError::parse_error(
                    "prism.hadamard.weight_names is set but prism.hadamard.block_size is missing".into(),
                ));
            }
            return Ok(Self {
                block_size: 0,
                gdn_v_grouped,
                rotation: Tensor::zeros([0i64, 0i64], zyx::DType::F32),
                signs: HashMap::new(),
                folded,
                inverse,
            });
        }
        if block_size < 0 || (block_size & (block_size - 1)) != 0 {
            return Err(ZyxError::parse_error(
                format!("prism.hadamard.block_size {block_size} is not a positive power of two").into(),
            ));
        }

        let widths = if metadata.contains_key("prism.hadamard.sign_widths") {
            metadata.ints("prism.hadamard.sign_widths")?
        } else {
            Vec::new()
        };
        let values = if metadata.contains_key("prism.hadamard.sign_values") {
            metadata.ints("prism.hadamard.sign_values")?
        } else {
            Vec::new()
        };
        let total: i64 = widths.iter().sum();
        if total != values.len() as i64 {
            return Err(ZyxError::parse_error(
                format!(
                    "prism.hadamard.sign_widths sum to {total} but sign_values has {} entries",
                    values.len()
                )
                .into(),
            ));
        }
        let mut signs = HashMap::new();
        let mut offset = 0usize;
        for width in widths {
            if width <= 0 {
                return Err(ZyxError::parse_error(
                    format!("prism.hadamard.sign_widths has a non-positive width {width}").into(),
                ));
            }
            let data: Vec<f32> = values[offset..offset + width as usize]
                .iter()
                .map(|&v| v as f32)
                .collect();
            if data.iter().any(|&v| v != 1.0 && v != -1.0) {
                return Err(ZyxError::parse_error(
                    format!("prism.hadamard.sign_values for width {width} contain values other than +/-1").into(),
                ));
            }
            offset += width as usize;
            signs.insert(width, Tensor::from(data));
        }

        Ok(Self {
            block_size,
            gdn_v_grouped,
            rotation: hadamard(block_size),
            signs,
            folded,
            inverse,
        })
    }

    /// Moves the rotation and sign tensors to `dev`.
    ///
    /// # Errors
    /// Device transfer errors.
    pub fn to(&mut self, dev: Dev) -> Result<(), ZyxError> {
        self.rotation = self.rotation.to(dev)?;
        for sign in self.signs.values_mut() {
            *sign = sign.to(dev)?;
        }
        Ok(())
    }

    /// `true` when the weight name is folded and its activation is rotated.
    #[must_use]
    pub fn is_folded(&self, name: &str) -> bool {
        self.folded.contains(name)
    }

    /// `true` when the table stores rotated rows.
    #[must_use]
    pub fn is_inverse(&self, name: &str) -> bool {
        self.inverse.contains(name)
    }

    /// `H (s * x)` for an activation of last width `width`.
    ///
    /// # Errors
    /// Shape or dtype errors from the rotation matmul.
    pub fn forward(&self, x: &Tensor, width: i64) -> Result<Tensor, ZyxError> {
        let signs = self.signs.get(&width).ok_or_else(|| {
            ZyxError::parse_error(format!("prism.hadamard has no sign vector for width {width}").into())
        })?;
        let x = x * signs;
        self.rotate(&x, width)
    }

    /// `s * (H z)` for a looked-up table row of width `width`.
    ///
    /// # Errors
    /// Shape or dtype errors from the rotation matmul.
    pub fn inverse(&self, x: &Tensor, width: i64) -> Result<Tensor, ZyxError> {
        let signs = self.signs.get(&width).ok_or_else(|| {
            ZyxError::parse_error(format!("prism.hadamard has no sign vector for width {width}").into())
        })?;
        let x = self.rotate(x, width)?;
        Ok(x * signs)
    }

    /// Blockwise normalized Walsh-Hadamard rotation of the last dimension.
    ///
    /// The transform is the butterfly itself, not a matmul against
    /// [`Prism::rotation`]: `log2(block_size)` stages over the block's own
    /// elements, then one `1/sqrt(block_size)` scale. Stage `h` views each
    /// block as `[.., block/(2h), 2, h]`; the size-2 axis is bit `h` of the
    /// element index, so its low slice takes `x + x^h` and its high slice
    /// `x - x^h` — the Sylvester-Walsh matrix's `(-1)^popcount(j & c)`.
    fn rotate(&self, x: &Tensor, width: i64) -> Result<Tensor, ZyxError> {
        if width % self.block_size != 0 {
            return Err(ZyxError::shape_error(
                format!("prism.hadamard block {} does not divide width {width}", self.block_size).into(),
            ));
        }
        let block = self.block_size;
        let shape = x.shape();
        let rows = x.numel() / block;
        let block_t = Tensor::from(block);
        let mut t = x.reshape([rows.clone(), block_t.clone()])?;
        let mut h = 1i64;
        while h < block {
            let groups = block / (2 * h);
            let view = t.reshape([rows.clone(), Tensor::from(groups), Tensor::from(2i64), Tensor::from(h)])?;
            let low = view.narrow(-2, 0i64, 1i64)?;
            let high = view.narrow(-2, 1i64, 1i64)?;
            let sum = &low + &high;
            let diff = &low - &high;
            // tinygrad's `.contiguous()`: each stage is a `STAGE`/store, so a
            // kernel boundary ends the fusion there. Without it the ten stage
            // views inline into one `4^stages` tree (block 1024 never
            // compiles); with it the body per stage is O(block).
            t = Tensor::cat([&sum, &diff], -2)?.reshape([rows.clone(), block_t.clone()])?.contiguous()?;
            h *= 2;
        }
        let scale = Tensor::from(1.0f32 / (block as f32).sqrt());
        (t * scale).reshape(shape)
    }
}

/// Normalized Sylvester-Walsh-Hadamard matrix.
fn hadamard(block_size: i64) -> Tensor {
    let n = usize::try_from(block_size).expect("a validated block size fits usize");
    let scale = 1.0f32 / (n as f32).sqrt();
    let mut data = Vec::with_capacity(n * n);
    for row in 0..n {
        for col in 0..n {
            let sign = if (row & col).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
            data.push(sign * scale);
        }
    }
    Tensor::from_vec(data, [block_size, block_size]).expect("a block matrix has a valid shape")
}
