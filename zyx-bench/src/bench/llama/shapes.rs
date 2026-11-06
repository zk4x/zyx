// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! llama-3.2-3B shapes, shared by every kernel in this group.
//!
//! The numbers are `config.json`'s: hidden 3072, 24 heads / 8 kv heads,
//! head_dim 128, intermediate 8192, vocab 128256, context 4096. Decode is
//! batch 1 seq 1; prefill is batch 1 seq 128 (the kv cache is batch-less).

use zyx::{Tensor, ZyxError};

pub const HIDDEN: i64 = 3072;
pub const INTERMEDIATE: i64 = 8192;
pub const HEADS: i64 = 24;
pub const KV_HEADS: i64 = 8;
pub const HEAD_DIM: i64 = 128;
pub const VOCAB: i64 = 128_256;
pub const CTX: i64 = 4096;
pub const SEQ: i64 = 128;
/// Decode length first, then prefill.
pub const SIZES: [i64; 2] = [1, SEQ];
pub const EPS: f32 = 1e-5;
pub const SCALE: f32 = 1.0 / 11.313_708;

/// `[1, n_kv, seq, hd]` -> `[1, n_kv*n_rep, seq, hd]`, the GQA repeat.
pub fn repeat_kv(x: Tensor, n_rep: i64) -> Result<Tensor, ZyxError> {
    if n_rep == 1 {
        return Ok(x);
    }
    let [b, n_kv, seq, hd] = x.dims::<4>()?;
    x.unsqueeze(2)?
        .expand([b.clone(), n_kv.clone(), n_rep.into(), seq.clone(), hd.clone()])?
        .reshape([b, n_kv * n_rep, seq, hd])
}
