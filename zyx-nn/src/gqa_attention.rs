// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Rotary position embeddings and the attention primitives the decoder block
//! inlines: the rotary half-pair rotation and the grouped-key/value repeat.
//!
//! There is no attention block type here: `DecoderBlock` owns the projections,
//! the cache and the frequency-table build so its fields can carry the GGUF
//! tensor names, and GGUF has no level between `blk.N.` and `attn_q`.

use zyx::{Tensor, ZyxError};

/// Llama-3 style RoPE frequency scaling: a piecewise rescale of the
/// low-frequency end of the table.
#[derive(Debug, Clone, Copy, zyx_derive::gguf)]
#[cfg_attr(feature = "py", pyo3::pyclass(from_py_object))]
pub struct RopeScaling {
    /// Scale applied beyond the original context.
    #[gguf("factor")]
    pub factor: f64,
    /// Upper wavelength bound of the interpolated band.
    #[gguf("high_freq_factor")]
    pub high_freq_factor: f64,
    /// Lower wavelength bound of the interpolated band.
    #[gguf("low_freq_factor")]
    pub low_freq_factor: f64,
    /// Context length the unscaled table was built for.
    #[gguf("original_context_length")]
    pub original_max_position_embeddings: i64,
}

/// `[b, n_kv_head, seq, hd]` -> `[b, n_kv_head*n_rep, seq, hd]`, the GQA repeat.
pub(crate) fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor, ZyxError> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let [b_sz, n_kv_head, seq_len, head_dim] = xs.dims::<4>()?;
    // `seq_len` may be symbolic: pass its dim tensor directly (NOT -1) so the
    // inferred dim keeps the original TensorId — merge-time provability checks
    // require the SAME dim tensor, not merely equal values.
    xs.unsqueeze(2)?
        .expand([-1i64, -1, n_rep as i64, -1, -1])?
        .reshape([b_sz, n_kv_head * (n_rep as i64), seq_len, head_dim])
}

/// Applies rotary embeddings to `[b, h, seq, d]` at absolute offset `offset`.
pub(crate) fn rotate(
    xs: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    offset: &Tensor,
) -> Result<Tensor, ZyxError> {
    let [seq_len, _head_dim] = xs.rdims::<2>()?;
    let [b, h, seq, d] = xs.dims::<4>()?;
    let one = Tensor::from(1i64);
    let two = Tensor::from(2i64);
    let half = d.clone() / two.clone();
    // `cos`/`sin` are `[max_context, head_dim/2]` in the model's weight dtype:
    // take this step's rows and shape them to broadcast over
    // `[batch, heads, seq, half, 1]`.
    let c = cos
        .narrow(0, offset.clone(), seq_len.clone())?
        .cast(xs.dtype())
        .reshape([
            one.clone(),
            one.clone(),
            seq.clone(),
            half.clone(),
            one.clone(),
        ])?;
    let s = sin
        .narrow(0, offset, seq_len.clone())?
        .cast(xs.dtype())
        .reshape([
            one.clone(),
            one.clone(),
            seq.clone(),
            half.clone(),
            one.clone(),
        ])?;
    // tinygrad's `apply_rotary_emb`: neighbouring elements are the
    // `(a, b)` pair, rotated by `(cos, sin)` as a complex product. That is
    // the layout GGUF stores q/k in (llama.cpp permutes them to it), unlike
    // Hugging Face's split-half `rotate_half`.
    let x = xs.reshape([b.clone(), h.clone(), seq.clone(), half, two])?;
    let zero = Tensor::from(0i64);
    let a = x.narrow(-1, zero, one.clone())?;
    let y = x.narrow(-1, one.clone(), one)?;
    let ro = a.clone() * c.clone() - y.clone() * s.clone();
    let co = a * s + y * c;
    Tensor::cat([&ro, &co], -1)?.reshape([b, h, seq, d])
}
