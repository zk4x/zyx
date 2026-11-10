// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! llama's op groups, lowered on `Dev::TT(0)` (ttsim).
//!
//! Same op mixes as `12_llama_ops`, with every tensor the device actually
//! consumes placed on Tenstorrent and the results read back to the host. This
//! is the failure list for "one llama forward step on TT": each test either
//! passes (that group lowers and computes), or reports the exact op and error
//! that stops it.
//!
//! No `Tape`: the tape path plans device assignments of its own and will copy
//! a TT result to CUDA (`CUDA copy from TT(0)`, `cuda.rs:2437`), which is not
//! what is under test here. The tensors are placed on TT explicitly and each
//! result is read back with `.to(Dev::C)`, exactly as the `23_tt_lowering`
//! gates do.
//!
//! F32 throughout, not the example's F16: the question here is whether the
//! lowering exists, and F16/BF16 add the packer-narrowing path as a second
//! confound. F16 is a separate axis once the ops lower.
//!
//! Run one test per process (`ttsim`'s owner thread dies with a failed test,
//! so later tests in the same process are a cascade):
//!
//! ```text
//! cd zyx && AGENT=1 cargo test --test 24_tt --no-run
//! BIN=$(ls -t target/debug/deps/24_tt-* | grep -v '\.d$' | head -1)
//! TT_TTSIM_LIB=/home/x/Dev/rust/metalium-rs/vendor/libttsim_bh.so \
//!   AGENT=1 timeout 10 $BIN <test> --exact --nocapture
//! ```

#![cfg(feature = "tenstorrent")]

use std::result::Result;
use zyx::{DType, Dev, Tensor, ZyxError};

/// The Tenstorrent device, if one initialized. A test returns `Ok(())` when it
/// did not, so the file is runnable on machines without the simulator.
fn tt() -> Option<Dev> {
    Dev::TT(0).info().ok().map(|_| Dev::TT(0))
}

/// Read a device tensor back: to the host, then to f32.
fn to_f32(t: &Tensor) -> Result<Vec<f32>, ZyxError> {
    let v: Vec<f32> = t.clone().to(Dev::C)?.cast(DType::F32).try_into()?;
    Ok(v)
}

fn assert_close(got: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: length mismatch");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() <= tol + 0.01 * e.abs(), "{what}[{i}]: got {g}, expected {e} (tol {tol})");
    }
}

// ============================================================================
// Embedding: one-hot via arange/equal/cast + gather by sum
// ============================================================================

#[test]
fn llama_embedding_one_hot() -> Result<(), ZyxError> {
    let Some(dev) = tt() else { return Ok(()) };
    // weight rows are distinct, exactly representable constants.
    let weight_f32: Vec<f32> = (0..8).flat_map(|r| vec![r as f32 + 1.0; 8]).collect();
    let weight = Tensor::from(weight_f32).reshape([8, 8])?.to(dev)?;
    let idx = Tensor::from(vec![0u32, 3, 7, 1]).reshape([2, 2])?.to(dev)?;

    // Same op mix as llama's embedding_forward: dims() decomposition, stack of
    // dim tensors in reshapes, arange, equal, cast, broadcast mul, sum.
    let [vocab, embed] = weight.dims::<2>()?;
    let [b, seq] = idx.dims::<2>()?;
    let idx4 = idx.cast(DType::F32).reshape([b, seq, 1i64.into(), 1i64.into()])?;
    let arange = Tensor::arange(0, vocab.item::<i64>(), 1)?
        .reshape([1i64.into(), 1i64.into(), vocab.clone(), 1i64.into()])?
        .cast(DType::F32)
        .to(dev)?;
    let w = weight.reshape([1i64.into(), 1i64.into(), vocab.clone(), embed])?;
    let one_hot = arange.equal(idx4)?.cast(w.dtype());
    let out = (one_hot * w).sum([2])?;

    let got = to_f32(&out)?;
    let expected: Vec<f32> = [0usize, 3, 7, 1].iter().flat_map(|&r| vec![r as f32 + 1.0; 8]).collect();
    assert_close(&got, &expected, 1e-6, "embedding gather");
    Ok(())
}

// ============================================================================
// RMSNorm: pow/mean_keepdim/add/rsqrt/mul chain
// ============================================================================

#[test]
fn llama_rms_norm() -> Result<(), ZyxError> {
    let Some(dev) = tt() else { return Ok(()) };
    let x = Tensor::from(vec![1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, 1.0]).reshape([2, 4])?.to(dev)?;
    let scale = Tensor::from(vec![1.0f32; 4]).to(dev)?;
    let eps = Tensor::from(1e-6f32).to(dev)?;

    // Same op mix as RMSNorm::forward: x * ((x*x).mean_keepdim(-1) + eps).rsqrt() * scale.
    let xx = x.clone() * x.clone();
    let mean = xx.mean_keepdim([-1])?;
    let normed = x.clone() * (mean + eps).rsqrt() * scale;

    // f32 reference on the same inputs.
    let rows: [[f32; 4]; 2] = [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, 1.0]];
    let mut expected = Vec::new();
    for row in rows {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / 4.0 + 1e-6;
        let inv = 1.0 / ms.sqrt();
        expected.extend(row.iter().map(|v| v * inv));
    }
    let got = to_f32(&normed)?;
    assert_close(&got, &expected, 0.02, "rmsnorm");
    Ok(())
}

// ============================================================================
// RoPE: narrow cos/sin by (symbolic) offset, rotate-half kernel
// ============================================================================

#[test]
fn llama_rope() -> Result<(), ZyxError> {
    let Some(dev) = tt() else { return Ok(()) };
    let head_dim = 4i64;
    let seq = 4i64;
    // llama's precompute_rope_freqs: t [max_pos, 1] @ inv_freq [1, hd/2],
    // computed on the host and placed, because the op under test is `rope`.
    let inv_freq: Vec<f32> = (0..2).map(|i| 1.0 / 10_000f32.powf(2.0 * i as f32 / 4.0)).collect();
    let inv_freq = Tensor::from(inv_freq).reshape([1, 2])?;
    let t = Tensor::arange(0u32, 4u32, 1)?.cast(DType::F32).reshape([4, 1])?;
    let freqs = t.matmul(&inv_freq)?;
    let cos = freqs.cos().to(dev)?;
    let sin = freqs.sin().to(dev)?;
    let x = Tensor::from(vec![
        1.0f32, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, -1.0, 0.5, 3.0,
    ])
    .reshape([seq, head_dim])?
    .to(dev)?;

    // Same op mix as apply_rope: rdims decomposition, narrow by an offset
    // tensor, rope kernel.
    let [s_len, _hd] = x.rdims::<2>()?;
    let offset = Tensor::from(0i64);
    let c = cos.narrow(0, offset.clone(), &s_len)?;
    let s = sin.narrow(0, offset, &s_len)?;
    let rotated = x.rope(c, s)?;

    // Identity check: rotation by 0 (cos=1, sin=0) leaves x unchanged.
    let ones = Tensor::ones([seq, 2], DType::F32).to(dev)?;
    let zeros = Tensor::zeros([seq, 2], DType::F32).to(dev)?;
    let [s_len, _hd] = x.rdims::<2>()?;
    let c1 = ones.narrow(0, 0i64, &s_len)?;
    let s1 = zeros.narrow(0, 0i64, &s_len)?;
    let identity = x.rope(c1, s1)?;

    let ident = to_f32(&identity)?;
    let orig = to_f32(&x)?;
    assert_close(&ident, &orig, 1e-6, "rope identity");

    // Rotation preserves per-row interleaved-pair magnitudes (NeoX pairs are
    // (x[0::2], x[1::2])).
    let got = to_f32(&rotated)?;
    for r in 0..seq as usize {
        for p in 0..2usize {
            let i = r * 4 + p;
            let before = orig[i] * orig[i] + orig[i + 2] * orig[i + 2];
            let after = got[i] * got[i] + got[i + 2] * got[i + 2];
            assert!((before - after).abs() <= 0.05 + 0.05 * before, "rope magnitude [{r}][{p}]: before {before}, after {after}");
        }
    }
    Ok(())
}

// ============================================================================
// Attention: q@k^T * scale, causal mask add, softmax, @v
// ============================================================================

#[test]
fn llama_attention_softmax() -> Result<(), ZyxError> {
    let Some(dev) = tt() else { return Ok(()) };
    let seq = 4i64;
    let d = 4i64;
    let scale = 1.0f32 / (d as f32).sqrt();
    // Fixed, exactly representable q/k/v so the reference is computable host-side.
    let q = Tensor::from(vec![
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ])
    .reshape([1, 1, seq, d])?
    .to(dev)?;
    let k = q.clone();
    let v = Tensor::from(vec![
        1.0f32, 2.0, 4.0, 8.0, 1.0, 2.0, 4.0, 8.0, 1.0, 2.0, 4.0, 8.0, 1.0, 2.0, 4.0, 8.0,
    ])
    .reshape([1, 1, seq, d])?
    .to(dev)?;

    // Host-built causal mask, same as llama's get_mask.
    let mask: Vec<f32> = (0..seq).flat_map(|i| (0..seq).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })).collect();
    let mask = Tensor::from(mask).reshape([seq, seq])?.to(dev)?;

    // Same op mix as Attention::forward: matmul, transpose, scale mul,
    // broadcast add of mask, softmax(-1), matmul.
    let attn = q.matmul(k.transpose(2, 3)?)? * scale;
    let attn = attn + mask;
    let probs = attn.softmax([-1])?;
    let out = probs.matmul(&v)?;

    // v rows are all-identical, so out rows == v row regardless of weights.
    let got = to_f32(&out)?;
    let expected: Vec<f32> = [1.0, 2.0, 4.0, 8.0].repeat(4);
    assert_close(&got, &expected, 0.02, "attention out (uniform v)");

    let probs_f = to_f32(&probs)?;
    for i in 0..seq as usize {
        let row_sum: f32 = probs_f[i * seq as usize..(i + 1) * seq as usize].iter().sum();
        assert!((row_sum - 1.0).abs() <= 0.02, "softmax row {i} sums to {row_sum}");
        for j in (i + 1)..seq as usize {
            assert!(probs_f[i * seq as usize + j].abs() <= 1e-6, "causal weight [{i}][{j}] != 0");
        }
    }
    Ok(())
}

// ============================================================================
// KV cache: narrow by a VARIABLE offset + assign (single write, symbolic read)
// ============================================================================

#[test]
fn llama_kv_cache_assign_symbolic() -> Result<(), ZyxError> {
    let Some(dev) = tt() else { return Ok(()) };
    let max_ctx = 16i64;
    let n_kv = 2i64;
    let hd = 4i64;
    let seq = 4i64;
    let cache = Tensor::zeros([max_ctx, n_kv, hd], DType::F32).contiguous()?.to(dev)?;
    // Written block: distinct values, per (pos, head, dim).
    let block: Vec<f32> = (0..seq)
        .flat_map(|s| (0..n_kv).flat_map(move |h| (0..hd).map(move |d| 1.0 + (s * 100 + h * 10 + d) as f32 / 100.0)))
        .collect();
    let k_assign = Tensor::from(block.clone()).reshape([seq, n_kv, hd])?.to(dev)?;

    // Symbolic position: the llama cache-update pattern (assign with a
    // variable offset). Shape inputs must be IDX_T (I64).
    let pos = Tensor::variable(2i64);
    let [s_len, _h, _d] = k_assign.rdims::<3>()?;
    cache.narrow(0, &pos, &s_len)?.assign(&k_assign)?;

    // Symbolic read length: identical kernel shape on every decode step.
    let cache_len = Tensor::variable(6i64);
    let read = cache.narrow(0, 0i64, cache_len)?.unsqueeze(0)?.transpose(1, 2)?;
    let out = read.sum([1])?;

    // out is [1, 6, 4]: position p (0..6), dim d — the sum over both kv heads
    // of cache[p][h][d]. Positions 0..2 are untouched zeros; positions 2..6
    // hold block row s = p-2 summed over heads.
    let got = to_f32(&out)?;
    let mut expected: Vec<f32> = Vec::new();
    for p in 0..6i64 {
        for d in 0..hd {
            let s = p - 2;
            let mut v = 0.0f32;
            if s >= 0 {
                for h in 0..n_kv {
                    v += 1.0 + (s * 100 + h * 10 + d) as f32 / 100.0;
                }
            }
            expected.push(v);
        }
    }
    assert_close(&got, &expected, 0.02, "kv cache after assign");
    Ok(())
}

// ============================================================================
// KV cache: REPEATED assign into one cache (in-place versioning)
// ============================================================================

#[test]
fn llama_kv_cache_assign_repeated() -> Result<(), ZyxError> {
    let Some(dev) = tt() else { return Ok(()) };
    let max_ctx = 8i64;
    let n_kv = 1i64;
    let hd = 4i64;
    let cache = Tensor::zeros([max_ctx, n_kv, hd], DType::F32).contiguous()?.to(dev)?;
    let block_a: Vec<f32> = (0..2).flat_map(|_| vec![1.0f32, 2.0, 3.0, 4.0]).collect();
    let block_b: Vec<f32> = (0..2).flat_map(|_| vec![5.0f32, 6.0, 7.0, 8.0]).collect();
    let a = Tensor::from(block_a).reshape([2, n_kv, hd])?.to(dev)?;
    let b = Tensor::from(block_b).reshape([2, n_kv, hd])?.to(dev)?;

    // Two assigns to the same buffer: the second writes after the first
    // (After chain of in-place versioned writes).
    let pos0 = Tensor::variable(0i64);
    cache.narrow(0, &pos0, 2i64)?.assign(&a)?;
    let pos1 = Tensor::variable(2i64);
    cache.narrow(0, &pos1, 2i64)?.assign(&b)?;
    let out = cache.sum([1, 2])?;

    // Rows 0-1 = a (sum 10), rows 2-3 = b (sum 26), rows 4-7 untouched zeros.
    let got = to_f32(&out)?;
    let expected: Vec<f32> = vec![10.0, 10.0, 26.0, 26.0, 0.0, 0.0, 0.0, 0.0];
    assert_close(&got, &expected, 1e-6, "repeated assign row sums");
    Ok(())
}

// ============================================================================
// Generation loop: two successive assigns with freshly created variable scalars
// (the per-step forward pattern, one step after another).
// ============================================================================

#[test]
fn llama_kv_cache_assign_generations() -> Result<(), ZyxError> {
    let Some(dev) = tt() else { return Ok(()) };
    let max_ctx = 8i64;
    let n_kv = 1i64;
    let hd = 4i64;
    let cache = Tensor::zeros([max_ctx, n_kv, hd], DType::F32).contiguous()?.to(dev)?;
    let block_a: Vec<f32> = (0..2).flat_map(|_| vec![1.0f32, 2.0, 3.0, 4.0]).collect();
    let block_b: Vec<f32> = (0..2).flat_map(|_| vec![5.0f32, 6.0, 7.0, 8.0]).collect();
    let a = Tensor::from(block_a).reshape([2, n_kv, hd])?.to(dev)?;
    let b = Tensor::from(block_b).reshape([2, n_kv, hd])?.to(dev)?;

    // Step 0 and step 1: a new variable each, the same cache.
    for (step, block) in [(0i64, &a), (2, &b)] {
        let pos = Tensor::variable(step);
        cache.narrow(0, &pos, 2i64)?.assign(block)?;
        let out = cache.sum([1, 2])?;
        let _ = to_f32(&out)?;
    }

    let got = to_f32(&cache)?;
    let expected: Vec<f32> = (0..2)
        .flat_map(|_| vec![1.0f32, 2.0, 3.0, 4.0])
        .chain((0..2).flat_map(|_| vec![5.0f32, 6.0, 7.0, 8.0]))
        .chain(std::iter::repeat_n(0.0, 16))
        .collect();
    assert_close(&got, &expected, 1e-6, "generation assign rows");
    Ok(())
}

// ============================================================================
// Cross-op tensor reuse: a realized eager value consumed by a binary op
// ============================================================================

#[test]
fn llama_promote_realized_eager_load() -> Result<(), ZyxError> {
    let Some(dev) = tt() else { return Ok(()) };
    let x = Tensor::ones([4], DType::F32).contiguous()?.to(dev)?;

    // y is realized by its own read, then consumed by a binary op with x.
    let y = x.sum([0])?;
    let z = &y * &x;

    let got = to_f32(&z)?;
    assert_close(&got, &[4.0; 4], 1e-6, "realized eager load");

    // A DISOWNED eager load: the handle is dropped after it entered the graph.
    let z3 = {
        let w = Tensor::from(vec![1.0f32, 2.0, 3.0, 4.0]).to(dev)?;
        &w * &x
    };
    let got3 = to_f32(&z3)?;
    assert_close(&got3, &[1.0, 2.0, 3.0, 4.0], 1e-6, "disowned eager load");
    Ok(())
}

// ============================================================================
// Causal mask: broadcast add of a host-built -inf mask
// ============================================================================

#[test]
fn llama_causal_mask_compare() -> Result<(), ZyxError> {
    let Some(dev) = tt() else { return Ok(()) };
    let seq = 4i64;
    // Mirror llama's get_mask: the causal mask is built on the HOST with -inf
    // entries and added to the attention scores.
    let mask: Vec<f32> =
        (0..seq as usize).flat_map(|i| (0..seq as usize).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })).collect();
    let masked = Tensor::from(mask).reshape([seq, seq])?.to(dev)?;

    let x = Tensor::from(vec![1.0f32; 16]).reshape([seq, seq])?.to(dev)?;

    let out = x.clone() + masked;

    let got = to_f32(&out)?;
    for i in 0..seq as usize {
        for j in 0..seq as usize {
            let want = if j > i { f32::NEG_INFINITY } else { 1.0 };
            assert_eq!(got[i * seq as usize + j], want, "mask[{i}][{j}]");
        }
    }
    Ok(())
}

// ============================================================================
// Feed-forward: swish(gate) * up @ down (GLU)
// ============================================================================

#[test]
fn llama_feedforward_glu() -> Result<(), ZyxError> {
    let Some(dev) = tt() else { return Ok(()) };
    let d = 8i64;
    let i = 16i64;
    let gate = Tensor::from(vec![0.5f32; (d * i) as usize]).reshape([d, i])?.to(dev)?;
    let up = Tensor::from(vec![2.0f32; (d * i) as usize]).reshape([d, i])?.to(dev)?;
    let down_w = Tensor::from(vec![0.25f32; (i * d) as usize]).reshape([i, d])?.to(dev)?;

    // Same op mix as MLP::forward: gate.swish() * up, then matmul with down.
    let hidden = gate.swish() * up;
    let out = hidden.matmul(&down_w)?;

    // swish(0.5) = 0.5 * sigmoid(0.5) ≈ 0.31123; per-element out =
    // 0.62246 * 0.25 * 16 = 2.48984; row sum over d=8 elements.
    let got = to_f32(&out)?;
    let swish_05 = 0.5f32 / (1.0 + (-0.5f32).exp());
    let want = swish_05 * 2.0 * 0.25 * i as f32 * d as f32;
    for r in 0..d as usize {
        let row_sum: f32 = got[r * d as usize..(r + 1) * d as usize].iter().sum();
        assert!((row_sum - want).abs() <= 0.05 + 0.01 * want, "glu row {r} sum {row_sum}, expected {want}");
    }
    Ok(())
}

// ============================================================================
// LM head: final projection matmul
// ============================================================================

#[test]
fn llama_lm_head_projection() -> Result<(), ZyxError> {
    let Some(dev) = tt() else { return Ok(()) };
    let vocab = 16i64;
    let d = 8i64;
    let xs = Tensor::from(vec![1.0f32; (2 * d) as usize]).reshape([2, d])?.to(dev)?;
    let w_flat: Vec<f32> = (0..vocab).flat_map(|v| vec![(v % 4) as f32 + 1.0; d as usize]).collect();
    let lm_head = Tensor::from(w_flat.clone()).reshape([d, vocab])?.to(dev)?;

    let logits = xs.matmul(&lm_head)?;

    // Host reference from the same weights: out[b][v] = Σ_d w[d][v].
    let got = to_f32(&logits)?;
    for v in 0..vocab as usize {
        let want: f32 = (0..d as usize).map(|dd| w_flat[dd * vocab as usize + v]).sum();
        assert_close(&got[v..v + 1], &[want], 0.02, &format!("logit[{v}]"));
    }
    Ok(())
}
