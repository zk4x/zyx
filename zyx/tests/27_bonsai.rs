// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Ternary-Bonsai-2-27B decode-step op sequences, at the checkpoint's own
//! geometry, exactly as `examples/bonsai-27b/src/model.rs` writes them.
//!
//! One test per sequence, traced on a `Tape` so the graph kernelizer and the
//! autotune passes see what the model's forward sees. Every test checks its
//! values against a scalar reference, so a wrong fusion fails as loudly as a
//! wrong lowering. The point of the file is the lowering: run it with
//! `ZYX_DEBUG=2` (egraph) for the kernel list, or with
//! `ZYX_DEBUG=1 ZYX_NO_CACHE=1` to make the autotune search print the winning
//! kernel's measured time and bandwidth.
//!
//! Decode shape: `seq = 1` at the last cache position (`POS = 4095`), so the
//! prefixes are the real 5120-wide stream, the FFN 17408, the GQA cache 4096
//! rows of 4 KV heads x 256, and the gated delta rule one token of 48 value
//! heads x 128.
//!
//! The trunk geometry comes from the GGUF metadata and is repeated here only
//! as constants: `hidden 5120`, `ffn 17408`, `n_head 24`, `n_kv 4`,
//! `head_dim 256`, `rot_dim 64`, `eps 1e-6`, `ssm_state 128`, `ssm_k_heads 16`,
//! `ssm_v_heads 48`, `ssm_v_head_dim 128`, `conv_kernel 4`, Hadamard block
//! 1024, vocab 248320.

use zyx::{DType, QDType, Scalar, Tape, Tensor, ZyxError, f16};

/// Residual stream width.
const HIDDEN: i64 = 5_120;
/// SwiGLU inner width.
const FFN: i64 = 17_408;
/// Full-attention heads.
const N_HEAD: i64 = 24;
/// Full-attention key/value heads.
const N_KV: i64 = 4;
/// Full-attention head width.
const HEAD_DIM: i64 = 256;
/// Rotated dimensions per head (partial RoPE).
const ROT_DIM: i64 = 64;
/// RMSNorm epsilon.
const EPS: f64 = 1e-6;
/// Width of one linear-attention key head.
const SSM_STATE: i64 = 128;
/// Linear-attention key heads.
const K_HEADS: i64 = 16;
/// Linear-attention value heads.
const V_HEADS: i64 = 48;
/// Width of one linear-attention value head.
const V_HEAD_DIM: i64 = 128;
/// Depthwise conv kernel width.
const CONV_KERNEL: i64 = 4;
/// Conv history rows kept across steps.
const HISTORY: i64 = CONV_KERNEL - 1;
/// `ssm_key_dim()`.
const KEY_DIM: i64 = SSM_STATE * K_HEADS;
/// `ssm_value_dim()`.
const VALUE_DIM: i64 = V_HEADS * V_HEAD_DIM;
/// `conv_dim()`.
const CONV_DIM: i64 = 2 * KEY_DIM + VALUE_DIM;
/// `n_head * head_dim`, the flattened attention output width.
const Q_DIM: i64 = N_HEAD * HEAD_DIM;
/// Cache length the decode tests run at (`max_context`; the model allows up
/// to 262144, 8192 is the parity harness's context).
const MAX_CONTEXT: i64 = 4_096;
/// Decode position: the last cache slot.
const POS: i64 = MAX_CONTEXT - 1;
/// Cache length read after writing this step.
const CACHE_LEN: i64 = POS + 1;
/// Hadamard block width.
const BLOCK: i64 = 1_024;
/// Vocabulary size.
const VOCAB: i64 = 248_320;
/// PQ2_0 blocks per embedding row.
const BLOCKS_PER_ROW: i64 = HIDDEN / 128;

/// Deterministic values in `[-1, 1)`; `seed` varies the pattern per tensor.
fn seq_vals(n: i64, seed: i64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = (i * 17 + seed * 29) % 23;
            (v as f32 - 11.0) / 11.0
        })
        .collect()
}

/// SiLU at the same formula as `Tensor::swish` (`x * sigmoid(x)`).
fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Normalized Sylvester-Walsh-Hadamard matrix (the same construction as
/// `prism.rs:187-200`).
fn hadamard(block: i64) -> Vec<f32> {
    let n = usize::try_from(block).expect("block fits usize");
    let scale = 1.0f32 / (n as f32).sqrt();
    let mut data = Vec::with_capacity(n * n);
    for row in 0..n {
        for col in 0..n {
            let sign = if (row & col).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
            data.push(sign * scale);
        }
    }
    data
}

/// The checkpoint's folded-weight sign vector, at an arbitrary but fixed
/// pattern (the test supplies it as a `Tape` input, so only its absolute
/// values matter).
fn signs_for(width: i64) -> Vec<f32> {
    (0..width).map(|i| if i % 3 == 1 { -1.0f32 } else { 1.0 }).collect()
}

/// `Prism::forward` (`prism.rs:151-183`): `H (s * x)` on the last width.
fn prism_forward(x: &Tensor, signs: &Tensor, rot: &Tensor) -> Result<Tensor, ZyxError> {
    let x = x * signs;
    let shape = x.shape();
    let rows = x.numel() / BLOCK;
    x.reshape([rows, Tensor::from(BLOCK)])?.matmul(rot)?.reshape(shape)
}

/// `Prism::inverse` (`prism.rs:159-183`): `s * (H z)` on the last width.
fn prism_inverse(x: &Tensor, signs: &Tensor, rot: &Tensor) -> Result<Tensor, ZyxError> {
    let shape = x.shape();
    let rows = x.numel() / BLOCK;
    let x = x.reshape([rows, Tensor::from(BLOCK)])?.matmul(rot)?.reshape(shape)?;
    Ok(x * signs)
}

/// Scalar `H (s * x)` over a whole `[rows, BLOCK]` strip (row-major).
fn rotate_ref(x: &[f32], rot: &[f32]) -> Vec<f32> {
    let rows = x.len() / BLOCK as usize;
    let mut out = vec![0f32; x.len()];
    for r in 0..rows {
        for j in 0..BLOCK as usize {
            let mut acc = 0f32;
            for c in 0..BLOCK as usize {
                acc += x[r * BLOCK as usize + c] * rot[j * BLOCK as usize + c];
            }
            out[r * BLOCK as usize + j] = acc;
        }
    }
    out
}

/// A PQ2_0 weight as the loader hands it over: `[rows*cols/128, 34]` U8 blocks
/// with a 1.0 f16 scale, so the dequantized values are exactly `-1, 0, 1`.
fn pq2_weight(rows: i64, cols: i64, seed: i64) -> (Vec<u8>, Tensor) {
    let blocks = rows * cols / 128;
    let mut packed = vec![0u8; (blocks * 34) as usize];
    for b in 0..blocks {
        let base = (b * 34) as usize;
        packed[base + 1] = 0x3c;
        for j in 0..32usize {
            packed[base + 2 + j] = ((b * 31 + seed + j as i64 * 11) % 251) as u8;
        }
    }
    let t = Tensor::from(packed.clone()).reshape([blocks, 34]).expect("blocks shape");
    (packed, t)
}

/// Dequantized `w` as `[rows, cols]` from its packed blocks (1.0 scale).
fn pq2_values(packed: &[u8], rows: i64, cols: i64) -> Vec<f32> {
    let mut out = vec![0f32; (rows * cols) as usize];
    for r in 0..rows {
        for c in 0..cols {
            // A 128-weight block spans several rows, so the offset inside the
            // block is the flat element index, not the column.
            let e = (r * cols + c) as usize;
            let b = e / 128;
            let j = e % 128;
            let base = b * 34;
            let q = (packed[base + 2 + j / 4] >> (2 * (j % 4))) & 3;
            out[e] = f32::from(q) - 1.0;
        }
    }
    out
}

/// F16 rounding, because the model's projections cast their activation.
fn round16(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

fn assert_close(got: &[f32], want: &[f32], abs: f32, rel: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, e)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - e).abs();
        assert!(d <= abs + rel * e.abs(), "{what}[{i}]: zyx {g} != {e} (diff {d})");
    }
}

/// `RMSNorm::forward` (`zyx-nn/src/rms_norm.rs:24-29`) on the decode row:
/// `x * rsqrt(mean(x^2) + eps) * weight`.
#[test]
fn rms_norm_forward() -> Result<(), ZyxError> {
    let x_vals = seq_vals(HIDDEN, 1);
    let w_vals: Vec<f32> = (0..HIDDEN).map(|i| 0.5 + (i % 5) as f32 * 0.25).collect();
    let x = Tensor::from_vec(x_vals.clone(), [1, HIDDEN])?;
    let weight = Tensor::from_vec(w_vals.clone(), [HIDDEN])?;

    let tape = Tape::new([&x, &weight])?;
    let dtype = x.dtype();
    let x_normed = &x * ((&x * &x).mean_keepdim([-1])? + Tensor::from(EPS).cast(dtype)).rsqrt();
    let out = x_normed * &weight;
    tape.realize([&out])?;

    let got: Vec<f32> = out.try_into()?;
    let eps = EPS as f32;
    let ss: f32 = x_vals.iter().map(|v| v * v).sum();
    let inv = 1.0 / (ss / HIDDEN as f32 + eps).sqrt();
    for c in 0..HIDDEN as usize {
        let want = x_vals[c] * inv * w_vals[c];
        assert!(got[c].is_equal(want), "rms_norm[{c}]: zyx {} != {want}", got[c]);
    }
    Ok(())
}

/// The linear block's beta gate (`model.rs:547`): `beta.sigmoid() * valid`.
#[test]
fn ssm_beta_gate() -> Result<(), ZyxError> {
    let beta_vals = seq_vals(V_HEADS, 2);
    let valid_vals = vec![1.0f32];
    let beta = Tensor::from_vec(beta_vals.clone(), [1, V_HEADS])?;
    let valid = Tensor::from_vec(valid_vals.clone(), [1])?;

    let tape = Tape::new([&beta, &valid])?;
    let b = beta.sigmoid() * valid.unsqueeze(1)?;
    tape.realize([&b])?;

    let got: Vec<f32> = b.try_into()?;
    for h in 0..V_HEADS as usize {
        let x = beta_vals[h];
        let want = (1.0 / (1.0 + (-x).exp())) * valid_vals[0];
        assert!(got[h].is_equal(want), "beta_gate[{h}]: zyx {} != {want}", got[h]);
    }
    Ok(())
}

/// The alpha gate (`model.rs:548`) and the decode path's `g.exp()`
/// (`model.rs:645`):
/// `((alpha + ssm_dt).softplus(1, 20) * ssm_a) * valid`, then `exp`.
/// Some heads sit above the softplus threshold so both select branches run.
#[test]
fn ssm_alpha_gate() -> Result<(), ZyxError> {
    let alpha_vals: Vec<f32> = (0..V_HEADS).map(|i| if i % 7 == 0 { 25.0 } else { -0.5 - (i % 5) as f32 }).collect();
    let dt_vals: Vec<f32> = (0..V_HEADS).map(|i| 0.5 + (i % 3) as f32 * 0.25).collect();
    let a_vals: Vec<f32> = (0..V_HEADS).map(|i| -0.25 - (i % 4) as f32 * 0.25).collect();
    let valid_vals = vec![1.0f32];
    let alpha = Tensor::from_vec(alpha_vals.clone(), [1, V_HEADS])?;
    let ssm_dt = Tensor::from_vec(dt_vals.clone(), [V_HEADS])?;
    let ssm_a = Tensor::from_vec(a_vals.clone(), [V_HEADS])?;
    let valid = Tensor::from_vec(valid_vals.clone(), [1])?;

    let tape = Tape::new([&alpha, &ssm_dt, &ssm_a, &valid])?;
    let g = ((&alpha + &ssm_dt).softplus(1.0f32, 20.0f32) * &ssm_a) * valid.unsqueeze(1)?;
    let gexp = g.squeeze([0]).exp();
    tape.realize([&gexp])?;

    let got: Vec<f32> = gexp.try_into()?;
    for h in 0..V_HEADS as usize {
        let x = alpha_vals[h] + dt_vals[h];
        // `softplus(beta, threshold)`: ln(exp(min(x, threshold)) + 1) when
        // x < threshold, else x.
        let sp = if x < 20.0 { ((x.min(20.0)).exp() + 1.0).ln() } else { x };
        let want = (sp * a_vals[h] * valid_vals[0]).exp();
        assert!(got[h].is_equal(want), "alpha_gate[{h}]: zyx {} != {want}", got[h]);
    }
    Ok(())
}

/// `repeat_tiled` (`model.rs:623-630`), the GQA repeat applied to q/k
/// (`model.rs:544-545`): `16` key heads to `48` value heads, `rep = 3`.
#[test]
fn repeat_tiled_gqa() -> Result<(), ZyxError> {
    const REP: i64 = V_HEADS / K_HEADS;
    let x_vals = seq_vals(K_HEADS * SSM_STATE, 3);
    let x = Tensor::from_vec(x_vals.clone(), [1, K_HEADS, SSM_STATE])?;

    let tape = Tape::new([&x])?;
    let out = x.unsqueeze(1)?.expand([-1i64, REP, -1, -1])?.reshape([1, K_HEADS * REP, SSM_STATE])?;
    tape.realize([&out])?;

    let got: Vec<f32> = out.try_into()?;
    for h in 0..V_HEADS as usize {
        for d in 0..SSM_STATE as usize {
            let want = x_vals[(h % K_HEADS as usize) * SSM_STATE as usize + d];
            let g = got[h * SSM_STATE as usize + d];
            assert!(g.is_equal(want), "repeat[{h},{d}]: zyx {g} != {want}");
        }
    }
    Ok(())
}

/// The depthwise causal conv of `LinearBlock::forward` (`model.rs:524-531`)
/// on the decode chunk: three history rows plus this token, four taps, SiLU.
#[test]
fn depthwise_conv_swish() -> Result<(), ZyxError> {
    let h_vals = seq_vals(HISTORY * CONV_DIM, 4);
    let qkv_vals = seq_vals(CONV_DIM, 5);
    let w_vals = seq_vals(CONV_DIM * CONV_KERNEL, 6);
    let history = Tensor::from_vec(h_vals.clone(), [HISTORY, CONV_DIM])?;
    let qkv = Tensor::from_vec(qkv_vals.clone(), [1, CONV_DIM])?;
    let conv_w = Tensor::from_vec(w_vals.clone(), [CONV_DIM, CONV_KERNEL])?;

    let tape = Tape::new([&history, &qkv, &conv_w])?;
    let cat = Tensor::cat([&history, &qkv], 0)?;
    let mut mixed = Tensor::zeros([1, CONV_DIM], DType::F32);
    for k in 0..CONV_KERNEL {
        let window = cat.narrow(0, k, 1i64)?;
        let weight = conv_w.narrow(1, k, 1i64)?.squeeze([1]);
        mixed = mixed + window * weight.unsqueeze(0)?;
    }
    let mixed = mixed.swish();
    tape.realize([&mixed])?;

    let got: Vec<f32> = mixed.try_into()?;
    for c in 0..CONV_DIM as usize {
        let mut acc = 0f32;
        for k in 0..CONV_KERNEL as usize {
            let row = if k < HISTORY as usize { &h_vals } else { &qkv_vals };
            let idx = if k < HISTORY as usize {
                k * CONV_DIM as usize + c
            } else {
                (k - HISTORY as usize) * CONV_DIM as usize + c
            };
            acc += row[idx] * w_vals[c * CONV_KERNEL as usize + k];
        }
        let want = swish(acc);
        assert!(got[c].is_equal(want), "conv[{c}]: zyx {} != {want}", got[c]);
    }
    Ok(())
}

/// The conv-state roll and in-place write (`model.rs:514-515`, `524`,
/// `533-534`): `keep`, `cat`, `narrow(0, real, 3)`, `assign`.
#[test]
fn conv_state_update() -> Result<(), ZyxError> {
    const REAL: i64 = 1;
    let s_vals = seq_vals(HISTORY * CONV_DIM, 7);
    let qkv_vals = seq_vals(CONV_DIM, 8);
    let conv_state = Tensor::from_vec(s_vals.clone(), [HISTORY, CONV_DIM])?;
    let qkv = Tensor::from_vec(qkv_vals.clone(), [1, CONV_DIM])?;
    let pos = Tensor::from(POS);

    let tape = Tape::new([&conv_state, &qkv, &pos])?;
    let keep = Tensor::from(1.0f32) - pos.equal(0i64)?.cast(DType::F32);
    let history = &conv_state * &keep;
    let cat = Tensor::cat([&history, &qkv], 0)?;
    let conv_state_new = cat.narrow(0, REAL, CONV_KERNEL - 1)?.contiguous()?;
    conv_state.clone().assign(&conv_state_new)?;
    tape.realize([&conv_state])?;

    let got: Vec<f32> = conv_state.try_into()?;
    for r in 0..HISTORY as usize {
        for c in 0..CONV_DIM as usize {
            let src = REAL as usize + r;
            let want = if src < HISTORY as usize {
                s_vals[src * CONV_DIM as usize + c]
            } else {
                qkv_vals[(src - HISTORY as usize) * CONV_DIM as usize + c]
            };
            let g = got[r * CONV_DIM as usize + c];
            assert!(g.is_equal(want), "conv_state[{r},{c}]: zyx {g} != {want}");
        }
    }
    Ok(())
}

/// The `split_qkv` helper (`model.rs:613-621`) on the decode row.
#[test]
fn split_qkv() -> Result<(), ZyxError> {
    let m_vals = seq_vals(CONV_DIM, 9);
    let mixed = Tensor::from_vec(m_vals.clone(), [1, CONV_DIM])?;

    let tape = Tape::new([&mixed])?;
    let q = mixed.narrow(1, 0i64, KEY_DIM)?.reshape([1, K_HEADS, SSM_STATE])?;
    let k = mixed.narrow(1, KEY_DIM, KEY_DIM)?.reshape([1, K_HEADS, SSM_STATE])?;
    let v = mixed.narrow(1, 2 * KEY_DIM, VALUE_DIM)?.reshape([1, V_HEADS, V_HEAD_DIM])?;
    tape.realize([&q, &k, &v])?;

    let q_got: Vec<f32> = q.try_into()?;
    let k_got: Vec<f32> = k.try_into()?;
    let v_got: Vec<f32> = v.try_into()?;
    for h in 0..K_HEADS as usize {
        for d in 0..SSM_STATE as usize {
            let i = h * SSM_STATE as usize + d;
            let want_q = m_vals[h * SSM_STATE as usize + d];
            let want_k = m_vals[KEY_DIM as usize + h * SSM_STATE as usize + d];
            assert!(q_got[i].is_equal(want_q), "q[{h},{d}]: zyx {} != {want_q}", q_got[i]);
            assert!(k_got[i].is_equal(want_k), "k[{h},{d}]: zyx {} != {want_k}", k_got[i]);
        }
    }
    for h in 0..V_HEADS as usize {
        for d in 0..V_HEAD_DIM as usize {
            let i = h * V_HEAD_DIM as usize + d;
            let want = m_vals[2 * KEY_DIM as usize + i];
            assert!(v_got[i].is_equal(want), "v[{h},{d}]: zyx {} != {want}", v_got[i]);
        }
    }
    Ok(())
}

/// The q/k RMS scaling (`model.rs:536-541`) on the decode row: normalise each
/// 128-wide head, scale q by `1/sqrt(ssm_state)`.
#[test]
fn qk_norm_scale() -> Result<(), ZyxError> {
    let q_vals = seq_vals(K_HEADS * SSM_STATE, 10);
    let k_vals = seq_vals(K_HEADS * SSM_STATE, 11);
    let q_in = Tensor::from_vec(q_vals.clone(), [1, K_HEADS, SSM_STATE])?;
    let k_in = Tensor::from_vec(k_vals.clone(), [1, K_HEADS, SSM_STATE])?;

    let tape = Tape::new([&q_in, &k_in])?;
    let q = &q_in * ((q_in.square().sum_keepdim([-1])? + EPS).rsqrt());
    let k = &k_in * ((k_in.square().sum_keepdim([-1])? + EPS).rsqrt());
    let scale = Tensor::from(1.0f32 / (SSM_STATE as f32).sqrt());
    let q = q * scale;
    tape.realize([&q, &k])?;

    let q_got: Vec<f32> = q.try_into()?;
    let k_got: Vec<f32> = k.try_into()?;
    let eps = EPS as f32;
    let scale_v = 1.0f32 / (SSM_STATE as f32).sqrt();
    for h in 0..K_HEADS as usize {
        let base = h * SSM_STATE as usize;
        let q_ss: f32 = (0..SSM_STATE as usize).map(|d| q_vals[base + d] * q_vals[base + d]).sum();
        let k_ss: f32 = (0..SSM_STATE as usize).map(|d| k_vals[base + d] * k_vals[base + d]).sum();
        let q_inv = 1.0 / (q_ss + eps).sqrt();
        let k_inv = 1.0 / (k_ss + eps).sqrt();
        for d in 0..SSM_STATE as usize {
            let want_q = q_vals[base + d] * q_inv * scale_v;
            let want_k = k_vals[base + d] * k_inv;
            assert!(q_got[base + d].is_equal(want_q), "q_norm[{h},{d}]: zyx {} != {want_q}", q_got[base + d]);
            assert!(k_got[base + d].is_equal(want_k), "k_norm[{h},{d}]: zyx {} != {want_k}", k_got[base + d]);
        }
    }
    Ok(())
}

/// `Rope::apply` (`model.rs:399-407`) on the decode row at `pos = 8191`:
/// the partial rotation over the first 64 of 256 dims, 24 heads.
#[test]
fn rope_partial() -> Result<(), ZyxError> {
    const HALF: i64 = ROT_DIM / 2;
    const ROPE_THETA: f64 = 10_000_000.0;

    // `Rope::new` (`model.rs:371-387`).
    let mut cos_vals = vec![0.0f32; (MAX_CONTEXT * HALF) as usize];
    let mut sin_vals = vec![0.0f32; (MAX_CONTEXT * HALF) as usize];
    for p in 0..MAX_CONTEXT {
        for pair in 0..HALF {
            let angle = p as f64 * ROPE_THETA.powf(-2.0 * pair as f64 / ROT_DIM as f64);
            let (s, c) = angle.sin_cos();
            cos_vals[(p * HALF + pair) as usize] = c as f32;
            sin_vals[(p * HALF + pair) as usize] = s as f32;
        }
    }
    let cos = Tensor::from_vec(cos_vals.clone(), [MAX_CONTEXT, HALF])?;
    let sin = Tensor::from_vec(sin_vals.clone(), [MAX_CONTEXT, HALF])?;

    let x_vals: Vec<f32> = (0..N_HEAD * HEAD_DIM).map(|i| ((i as i64 % 9) - 4) as f32 * 0.25).collect();
    let x = Tensor::from_vec(x_vals.clone(), [1, N_HEAD, HEAD_DIM])?;

    let tape = Tape::new([&cos, &sin])?;
    tape.add(&x)?;

    let pos = Tensor::variable(POS);
    let cos = cos.narrow(0, pos.clone(), 1i64)?;
    let sin = sin.narrow(0, pos, 1i64)?;
    let x4 = x.transpose(0, 1)?.unsqueeze(0)?;
    let rotated = x4.narrow(3, 0i64, ROT_DIM)?;
    let rest = x4.narrow(3, ROT_DIM, HEAD_DIM - ROT_DIM)?;
    let rotated = rotated.rope(&cos, &sin)?;
    let out = Tensor::cat([&rotated, &rest], 3)?.squeeze([0]).transpose(0, 1)?;
    tape.realize([&out])?;

    let mut want = vec![0.0f32; (N_HEAD * HEAD_DIM) as usize];
    for h in 0..N_HEAD as usize {
        for d in 0..HEAD_DIM as usize {
            let i = h * HEAD_DIM as usize + d;
            want[i] = if (d as i64) < ROT_DIM {
                let pair = d as i64 % HALF;
                let x1 = x_vals[h * HEAD_DIM as usize + pair as usize];
                let x2 = x_vals[h * HEAD_DIM as usize + (pair + HALF) as usize];
                let c = cos_vals[(POS * HALF + pair) as usize];
                let sn = sin_vals[(POS * HALF + pair) as usize];
                if (d as i64) < HALF {
                    x1 * c - x2 * sn
                } else {
                    x1 * sn + x2 * c
                }
            } else {
                x_vals[i]
            };
        }
    }
    let got: Vec<f32> = out.try_into()?;
    assert_close(&got, &want, 1e-6, 1e-6, "partial rope");
    Ok(())
}

/// `FullBlock::forward` (`model.rs:458-490`) on the decode row: 4096-row F16
/// K/V cache, 4 KV heads repeated to 24, the scores/softmax/V dot, the
/// sigmoid gate, the folded output projection and the residual.
#[test]
fn gqa_attention() -> Result<(), ZyxError> {
    const REP: i64 = N_HEAD / N_KV;

    let q_vals: Vec<f32> = (0..N_HEAD * HEAD_DIM).map(|i| ((i as i64 % 7) - 3) as f32 * 0.25).collect();
    let gate_vals: Vec<f32> = (0..Q_DIM).map(|i| ((i as i64 % 5) - 2) as f32 * 0.5).collect();
    let x_vals: Vec<f32> = (0..HIDDEN).map(|i| ((i as i64 % 3) - 1) as f32 * 0.5).collect();
    let ck_vals: Vec<f32> = (0..CACHE_LEN * N_KV * HEAD_DIM).map(|i| ((i as i64 % 5) - 2) as f32 * 0.5).collect();
    let cv_vals: Vec<f32> = (0..CACHE_LEN * N_KV * HEAD_DIM).map(|i| ((i as i64 % 7) - 3) as f32 * 0.25).collect();
    let o_packed = pq2_weight(HIDDEN, Q_DIM, 12).0;
    let o_w = pq2_values(&o_packed, HIDDEN, Q_DIM);

    let q = Tensor::from_vec(q_vals.clone(), [1, N_HEAD, HEAD_DIM])?;
    let gate = Tensor::from_vec(gate_vals.clone(), [1, Q_DIM])?;
    let x = Tensor::from_vec(x_vals.clone(), [1, HIDDEN])?;
    let cache_k = Tensor::from_vec(ck_vals.clone(), [CACHE_LEN, N_KV, HEAD_DIM])?.cast(DType::F16);
    let cache_v = Tensor::from_vec(cv_vals.clone(), [CACHE_LEN, N_KV, HEAD_DIM])?.cast(DType::F16);
    let o_t = Tensor::from(o_packed.clone()).reshape([HIDDEN * Q_DIM / 128, 34])?;
    let rot = Tensor::from_vec(hadamard(BLOCK), [BLOCK, BLOCK])?;
    let signs = Tensor::from_vec(signs_for(Q_DIM), [Q_DIM])?;

    let tape = Tape::new([&cache_k, &cache_v, &rot, &signs])?;
    tape.add(&q)?;
    tape.add(&gate)?;
    tape.add(&x)?;
    tape.add(&o_t)?;

    let pos = Tensor::variable(POS);
    let real_t = Tensor::variable(1i64);
    let cache_len = &pos + &real_t;

    let kk = cache_k.narrow(0, 0i64, cache_len.clone())?.transpose(0, 1)?;
    let vv = cache_v.narrow(0, 0i64, cache_len.clone())?.transpose(0, 1)?;
    let kk = kk.unsqueeze(1)?.expand([-1i64, REP, -1, -1])?.reshape([
        Tensor::from(N_HEAD),
        cache_len.clone(),
        Tensor::from(HEAD_DIM),
    ])?;
    let vv = vv.unsqueeze(1)?.expand([-1i64, REP, -1, -1])?.reshape([Tensor::from(N_HEAD), cache_len, Tensor::from(HEAD_DIM)])?;

    let q = q.transpose(0, 1)?;
    let scale = Tensor::from(1.0f32 / (HEAD_DIM as f32).sqrt());
    let attn = q.cast(DType::F16).dot_dtype_output(kk.transpose(1, 2)?.cast(DType::F16), DType::F32, DType::F32)? * scale;
    let probs = attn.softmax([-1])?;
    let out = probs.cast(DType::F16).dot_dtype_output(&vv, DType::F32, DType::F32)?.transpose(0, 1)?.reshape([1, Q_DIM])?;
    let out = out * gate.sigmoid();
    let o = prism_forward(&out, &signs, &rot)?.cast(DType::F16).dot_dtype_output(
        o_t.dequantize(QDType::PQ2_0, DType::F16)?.reshape([HIDDEN, Q_DIM])?.t(),
        DType::F32,
        DType::F32,
    )?;
    let y = &x + &o;
    tape.realize([&probs, &y])?;

    // Reference: scores, softmax (p rounded to F16 for the V dot), gate, the
    // folded output rotation (rounded to F16), projection, residual.
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let mut want_probs = vec![0.0f32; (N_HEAD * CACHE_LEN) as usize];
    let mut gated = vec![0.0f32; Q_DIM as usize];
    for h in 0..N_HEAD as usize {
        let kv = h / REP as usize;
        let mut scores = vec![0.0f32; CACHE_LEN as usize];
        for j in 0..CACHE_LEN as usize {
            let mut dot = 0.0f32;
            for d in 0..HEAD_DIM as usize {
                dot += q_vals[h * HEAD_DIM as usize + d] * ck_vals[(j * N_KV as usize + kv) * HEAD_DIM as usize + d];
            }
            scores[j] = dot * scale;
        }
        let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in scores.iter_mut() {
            *v = (*v - mx).exp();
            sum += *v;
        }
        for j in 0..CACHE_LEN as usize {
            let p = scores[j] / sum;
            want_probs[h * CACHE_LEN as usize + j] = p;
            let p16 = round16(p);
            for d in 0..HEAD_DIM as usize {
                gated[h * HEAD_DIM as usize + d] += p16 * cv_vals[(j * N_KV as usize + kv) * HEAD_DIM as usize + d];
            }
        }
    }
    for c in 0..Q_DIM as usize {
        gated[c] *= 1.0 / (1.0 + (-gate_vals[c]).exp());
    }
    let signed: Vec<f32> = gated.iter().zip(signs_for(Q_DIM)).map(|(v, s)| v * s).collect();
    let rotated = rotate_ref(&signed, &hadamard(BLOCK));
    let mut want_out = vec![0.0f32; HIDDEN as usize];
    for r in 0..HIDDEN as usize {
        let mut acc = 0.0f32;
        for c in 0..Q_DIM as usize {
            acc += round16(rotated[c]) * o_w[r * Q_DIM as usize + c];
        }
        want_out[r] = x_vals[r] + acc;
    }

    let got_probs: Vec<f32> = probs.try_into()?;
    assert_close(&got_probs, &want_probs, 1e-6, 1e-3, "gqa softmax");
    let got: Vec<f32> = y.try_into()?;
    assert_close(&got, &want_out, 1e-2, 2e-3, "gqa out");
    Ok(())
}

/// The cache write (`model.rs:448-456`) on the decode token at `pos = 8191`:
/// one row of 4 KV heads x 256 written as F16, then read back over 4096 rows.
#[test]
fn kv_cache_write() -> Result<(), ZyxError> {
    let k_vals: Vec<f32> = (0..N_KV * HEAD_DIM).map(|i| ((i as i64 % 9) - 4) as f32 * 0.25).collect();
    let v_vals: Vec<f32> = (0..N_KV * HEAD_DIM).map(|i| ((i as i64 % 7) - 3) as f32 * 0.5).collect();
    let k = Tensor::from_vec(k_vals.clone(), [1, N_KV, HEAD_DIM])?;
    let v = Tensor::from_vec(v_vals.clone(), [1, N_KV, HEAD_DIM])?;

    let cache_k = Tensor::zeros([MAX_CONTEXT, N_KV, HEAD_DIM], DType::F16).contiguous()?;
    let cache_v = Tensor::zeros([MAX_CONTEXT, N_KV, HEAD_DIM], DType::F16).contiguous()?;

    let tape = Tape::empty();
    tape.add(&cache_k)?;
    tape.add(&cache_v)?;

    let pos = Tensor::variable(POS);
    let real_t = Tensor::variable(1i64);
    let cache_len = &pos + &real_t;
    let real = &cache_len - &pos;

    cache_k.clone().narrow(0, pos.clone(), real.clone())?.assign(&k.narrow(0, 0i64, real.clone())?.cast(DType::F16))?;
    cache_v.clone().narrow(0, pos.clone(), real.clone())?.assign(&v.narrow(0, 0i64, real.clone())?.cast(DType::F16))?;

    let kk = cache_k.narrow(0, 0i64, cache_len.clone())?.transpose(0, 1)?;
    let vv = cache_v.narrow(0, 0i64, cache_len)?.transpose(0, 1)?;
    tape.realize([&kk, &vv])?;

    let got_k: Vec<f32> = cache_k.cast(DType::F32).try_into()?;
    let got_v: Vec<f32> = cache_v.cast(DType::F32).try_into()?;
    let row = (POS * N_KV * HEAD_DIM) as usize;
    for c in 0..(N_KV * HEAD_DIM) as usize {
        let idx = row + c;
        assert!(got_k[idx].is_equal(k_vals[c]), "cache_k[{POS},{c}]: zyx {} != {}", got_k[idx], k_vals[c]);
        assert!(got_v[idx].is_equal(v_vals[c]), "cache_v[{POS},{c}]: zyx {} != {}", got_v[idx], v_vals[c]);
    }
    // Everything below the written row is still zero.
    for i in 0..row {
        assert_eq!(got_k[i], 0.0, "cache_k[{i}] not zero");
        assert_eq!(got_v[i], 0.0, "cache_v[{i}] not zero");
    }

    // The symbolic read sees the whole 4096-row cache transposed.
    let got_kk: Vec<f32> = kk.cast(DType::F32).try_into()?;
    for h in 0..N_KV as usize {
        for p in (0..CACHE_LEN as usize).step_by(997) {
            for d in 0..HEAD_DIM as usize {
                let want = got_k[(p * N_KV as usize + h) * HEAD_DIM as usize + d];
                let g = got_kk[(h * CACHE_LEN as usize + p) * HEAD_DIM as usize + d];
                assert_eq!(g, want, "cache_k read[{h},{p},{d}]");
            }
        }
    }
    Ok(())
}

/// `delta_step` (`model.rs:642-653`): one decode token of 48 value heads of
/// width 128, `state` `[48, 128, 128]`.
#[test]
fn delta_step_decode() -> Result<(), ZyxError> {
    let q_vals = seq_vals(V_HEADS * SSM_STATE, 13);
    let k_vals = seq_vals(V_HEADS * SSM_STATE, 14);
    let v_vals = seq_vals(V_HEADS * SSM_STATE, 15);
    let g_vals: Vec<f32> = (0..V_HEADS).map(|i| -0.01 * ((i % 4) as f32 + 1.0)).collect();
    let b_vals: Vec<f32> = (0..V_HEADS).map(|i| 0.25 + 0.01 * (i % 3) as f32).collect();
    let state_vals = seq_vals(V_HEADS * SSM_STATE * SSM_STATE, 16);

    let q = Tensor::from_vec(q_vals.clone(), [1, V_HEADS, SSM_STATE])?;
    let k = Tensor::from_vec(k_vals.clone(), [1, V_HEADS, SSM_STATE])?;
    let v = Tensor::from_vec(v_vals.clone(), [1, V_HEADS, SSM_STATE])?;
    let g = Tensor::from_vec(g_vals.clone(), [1, V_HEADS])?;
    let b = Tensor::from_vec(b_vals.clone(), [1, V_HEADS])?;
    let state = Tensor::from_vec(state_vals.clone(), [V_HEADS, SSM_STATE, SSM_STATE])?;

    let tape = Tape::new([&q, &k, &v, &g, &b, &state])?;
    let q = q.squeeze([0]);
    let k = k.squeeze([0]);
    let v = v.squeeze([0]);
    let g = g.squeeze([0]).exp();
    let b = b.squeeze([0]);

    let kv = (state.clone() * k.clone().unsqueeze(2)?).sum([1])?;
    let delta = (v - g.clone().unsqueeze(1)? * kv) * b.unsqueeze(1)?;
    let new_state = state.clone() * g.clone().unsqueeze(1)?.unsqueeze(1)? + k.unsqueeze(2)? * delta.unsqueeze(1)?;
    let core = (new_state.clone() * q.unsqueeze(2)?).sum([1])?;
    let core = core.unsqueeze(0)?;
    tape.realize([&core, &new_state])?;

    let (want_core, want_state) = delta_ref(&q_vals, &k_vals, &v_vals, &g_vals, &b_vals, &state_vals);
    let got_core: Vec<f32> = core.try_into()?;
    let got_state: Vec<f32> = new_state.try_into()?;
    assert_close(&got_core, &want_core, 2e-5, 1e-6, "core");
    assert_close(&got_state, &want_state, 2e-5, 1e-6, "state");
    Ok(())
}

/// The same decode step plus the caller's in-place write
/// (`model.rs:549-554`: `self.state.clone().assign(&new_state)`).
#[test]
fn delta_step_state_assign() -> Result<(), ZyxError> {
    let q_vals = seq_vals(V_HEADS * SSM_STATE, 17);
    let k_vals = seq_vals(V_HEADS * SSM_STATE, 18);
    let v_vals = seq_vals(V_HEADS * SSM_STATE, 19);
    let g_vals: Vec<f32> = (0..V_HEADS).map(|i| -0.01 * ((i % 3) as f32 + 1.0)).collect();
    let b_vals: Vec<f32> = (0..V_HEADS).map(|i| 0.3 + 0.005 * (i % 2) as f32).collect();
    let state_vals = seq_vals(V_HEADS * SSM_STATE * SSM_STATE, 20);

    let q = Tensor::from_vec(q_vals.clone(), [1, V_HEADS, SSM_STATE])?;
    let k = Tensor::from_vec(k_vals.clone(), [1, V_HEADS, SSM_STATE])?;
    let v = Tensor::from_vec(v_vals.clone(), [1, V_HEADS, SSM_STATE])?;
    let g = Tensor::from_vec(g_vals.clone(), [1, V_HEADS])?;
    let b = Tensor::from_vec(b_vals.clone(), [1, V_HEADS])?;
    let state = Tensor::from_vec(state_vals.clone(), [V_HEADS, SSM_STATE, SSM_STATE])?;

    let tape = Tape::new([&q, &k, &v, &g, &b, &state])?;
    let q = q.squeeze([0]);
    let k = k.squeeze([0]);
    let v = v.squeeze([0]);
    let g = g.squeeze([0]).exp();
    let b = b.squeeze([0]);

    let kv = (state.clone() * k.clone().unsqueeze(2)?).sum([1])?;
    let delta = (v - g.clone().unsqueeze(1)? * kv) * b.unsqueeze(1)?;
    let new_state = state.clone() * g.clone().unsqueeze(1)?.unsqueeze(1)? + k.unsqueeze(2)? * delta.unsqueeze(1)?;
    let core = (new_state.clone() * q.unsqueeze(2)?).sum([1])?;
    let core = core.unsqueeze(0)?;
    state.clone().assign(&new_state)?;
    tape.realize([&core, &state])?;

    let (want_core, want_state) = delta_ref(&q_vals, &k_vals, &v_vals, &g_vals, &b_vals, &state_vals);
    let got_core: Vec<f32> = core.try_into()?;
    let got_state: Vec<f32> = state.try_into()?;
    assert_close(&got_core, &want_core, 2e-5, 1e-6, "core");
    assert_close(&got_state, &want_state, 2e-5, 1e-6, "assigned state");
    Ok(())
}

/// Scalar `delta_step` over `[V_HEADS, SSM_STATE, SSM_STATE]`.
fn delta_ref(q: &[f32], k: &[f32], v: &[f32], g: &[f32], b: &[f32], state: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let h = V_HEADS as usize;
    let d = SSM_STATE as usize;
    let mut core = vec![0f32; h * d];
    let mut out = vec![0f32; h * d * d];
    for hh in 0..h {
        let gexp = g[hh].exp();
        let beta = b[hh];
        let mut kv = vec![0f32; d];
        for dd in 0..d {
            for a in 0..d {
                kv[dd] += state[hh * d * d + a * d + dd] * k[hh * d + a];
            }
        }
        let mut delta = vec![0f32; d];
        for dd in 0..d {
            delta[dd] = (v[hh * d + dd] - gexp * kv[dd]) * beta;
        }
        for a in 0..d {
            for dd in 0..d {
                out[hh * d * d + a * d + dd] = state[hh * d * d + a * d + dd] * gexp + k[hh * d + a] * delta[dd];
            }
        }
        for dd in 0..d {
            for a in 0..d {
                core[hh * d + dd] += out[hh * d * d + a * d + dd] * q[hh * d + a];
            }
        }
    }
    (core, out)
}

/// `linear` (`model.rs:585-589`) on the decode row: the folded activation
/// rotation, the PQ2_0 dequant, the transposed operand and the F16/F32 dot.
/// Mirrors `ffn_gate`: `[1, 5120] x [17408, 5120]`.
#[test]
fn linear_projection() -> Result<(), ZyxError> {
    let (packed, w) = pq2_weight(FFN, HIDDEN, 1);
    let x_vals = seq_vals(HIDDEN, 21);
    let x = Tensor::from_vec(x_vals.clone(), [1, HIDDEN])?;
    let rot = Tensor::from_vec(hadamard(BLOCK), [BLOCK, BLOCK])?;
    let signs = Tensor::from_vec(signs_for(HIDDEN), [HIDDEN])?;

    let tape = Tape::new([&w, &rot, &signs])?;
    tape.add(&x)?;
    let out = prism_forward(&x, &signs, &rot)?.cast(DType::F16).dot_dtype_output(
        w.dequantize(QDType::PQ2_0, DType::F16)?.reshape([FFN, HIDDEN])?.t(),
        DType::F32,
        DType::F32,
    )?;
    tape.realize([&out])?;

    let signed: Vec<f32> = x_vals.iter().zip(signs_for(HIDDEN)).map(|(v, s)| v * s).collect();
    let rotated = rotate_ref(&signed, &hadamard(BLOCK));
    let gw = pq2_values(&packed, FFN, HIDDEN);
    let got: Vec<f32> = out.try_into()?;
    let mut want = vec![0f32; FFN as usize];
    for r in 0..FFN as usize {
        let mut acc = 0f32;
        for c in 0..HIDDEN as usize {
            acc += round16(rotated[c]) * gw[r * HIDDEN as usize + c];
        }
        want[r] = acc;
    }
    assert_close(&got, &want, 1e-2, 2e-3, "projection");
    Ok(())
}

/// `ffn_forward` (`model.rs:599`): pre-norm, the folded gate/up projections
/// through SwiGLU, the folded down projection and the residual.
#[test]
fn ffn_swiglu() -> Result<(), ZyxError> {
    let (gate_p, gate_w) = pq2_weight(FFN, HIDDEN, 2);
    let (up_p, up_w) = pq2_weight(FFN, HIDDEN, 3);
    let (down_p, down_w) = pq2_weight(HIDDEN, FFN, 4);
    let norm: Vec<f32> = (0..HIDDEN).map(|i| 0.01 + (i % 5) as f32 * 0.0025).collect();
    let x_vals: Vec<f32> = seq_vals(HIDDEN, 22).iter().map(|v| v * 0.02).collect();

    let norm_t = Tensor::from_vec(norm.clone(), [HIDDEN])?;
    let x = Tensor::from_vec(x_vals.clone(), [1, HIDDEN])?;
    let rot = Tensor::from_vec(hadamard(BLOCK), [BLOCK, BLOCK])?;
    let signs_h = Tensor::from_vec(signs_for(HIDDEN), [HIDDEN])?;
    let signs_f = Tensor::from_vec(signs_for(FFN), [FFN])?;
    let tape = Tape::new([&gate_w, &up_w, &down_w, &norm_t, &rot, &signs_h, &signs_f])?;
    tape.add(&x)?;

    let linear = |x: &Tensor, w: &Tensor, rows: i64, cols: i64| -> Result<Tensor, ZyxError> {
        let w = w.dequantize(QDType::PQ2_0, DType::F16)?.reshape([rows, cols])?.t();
        x.cast(DType::F16).dot_dtype_output(w, DType::F32, DType::F32)
    };
    // `RMSNorm::forward`, then two folded projections of the same `h`.
    let eps = EPS as f32;
    let h = &x * ((&x * &x).mean_keepdim([-1])? + Tensor::from(EPS).cast(DType::F32)).rsqrt() * &norm_t;
    let gate = linear(&prism_forward(&h, &signs_h, &rot)?, &gate_w, FFN, HIDDEN)?;
    let up = linear(&prism_forward(&h, &signs_h, &rot)?, &up_w, FFN, HIDDEN)?;
    let mid = gate.swish() * &up;
    let down = linear(&prism_forward(&mid, &signs_f, &rot)?, &down_w, HIDDEN, FFN)?;
    let out = &x + &down;
    tape.realize([&out])?;
    let got: Vec<f32> = out.try_into()?;

    // Scalar reference, rounding where the graph casts to F16.
    let ss: f32 = x_vals.iter().map(|v| v * v).sum();
    let inv = 1.0 / (ss / HIDDEN as f32 + eps).sqrt();
    let hv: Vec<f32> = x_vals.iter().zip(&norm).map(|(v, w)| v * inv * w).collect();
    let h_signed: Vec<f32> = hv.iter().zip(signs_for(HIDDEN)).map(|(v, s)| v * s).collect();
    let h_rot = rotate_ref(&h_signed, &hadamard(BLOCK));
    let gw = pq2_values(&gate_p, FFN, HIDDEN);
    let uw = pq2_values(&up_p, FFN, HIDDEN);
    let mut mid_ref = vec![0f32; FFN as usize];
    for r in 0..FFN as usize {
        let mut gacc = 0f32;
        let mut uacc = 0f32;
        for c in 0..HIDDEN as usize {
            let a = round16(h_rot[c]);
            gacc += a * gw[r * HIDDEN as usize + c];
            uacc += a * uw[r * HIDDEN as usize + c];
        }
        mid_ref[r] = swish(gacc) * uacc;
    }
    let m_signed: Vec<f32> = mid_ref.iter().zip(signs_for(FFN)).map(|(v, s)| v * s).collect();
    let m_rot = rotate_ref(&m_signed, &hadamard(BLOCK));
    let dw = pq2_values(&down_p, HIDDEN, FFN);
    let mut want = vec![0f32; HIDDEN as usize];
    for r in 0..HIDDEN as usize {
        let mut acc = 0f32;
        for c in 0..FFN as usize {
            acc += round16(m_rot[c]) * dw[r * FFN as usize + c];
        }
        want[r] = x_vals[r] + acc;
    }
    // The rotated mid is 17408 wide and is cast to F16 before the down dot, so
    // a ~1e-6 f32 reordering in `mid` flips F16 ULPs that the 17408-term sum
    // amplifies to a few percent. The tolerance admits that, and a wrong
    // fusion (a missing rotation, swish or operand) is orders above it.
    assert_close(&got, &want, 0.15, 0.05, "ffn");
    Ok(())
}

/// The embedding step (`model.rs:289-295`) on the full 248320 x 5120 packed
/// table: the block row gather, the PQ2_0 dequant and the inverse rotation
/// (`token_embd.weight` is the one inverse weight).
#[test]
fn embedding_lookup() -> Result<(), ZyxError> {
    const PICKED: i64 = 7;
    let blocks = VOCAB * BLOCKS_PER_ROW;
    let mut packed = vec![0u8; (blocks * 34) as usize];
    for b in 0..blocks {
        let base = (b * 34) as usize;
        packed[base + 1] = 0x3c;
        packed[base + 2] = (b % 251) as u8;
    }
    let table = Tensor::from(packed.clone()).reshape([blocks, 34])?;
    let ids = Tensor::from_vec(vec![PICKED], [1])?;
    let rot = Tensor::from_vec(hadamard(BLOCK), [BLOCK, BLOCK])?;
    let signs = Tensor::from_vec(signs_for(HIDDEN), [HIDDEN])?;

    let tape = Tape::new([&table, &rot, &signs])?;
    tape.add(&ids)?;
    let row = ids.reshape([1, 1i64])? * BLOCKS_PER_ROW;
    let col = Tensor::arange(0i64, BLOCKS_PER_ROW, 1i64)?.reshape([1i64, BLOCKS_PER_ROW])?;
    let index = (row + col).reshape([BLOCKS_PER_ROW])?.cast(DType::I32);
    let picked = table.index_select(0, index)?;
    let x = picked.dequantize(QDType::PQ2_0, DType::F32)?.reshape([1, HIDDEN])?;
    let x = prism_inverse(&x, &signs, &rot)?;
    tape.realize([&x])?;

    let mut base_vals = vec![0f32; HIDDEN as usize];
    for c in 0..HIDDEN as usize {
        let b = PICKED * BLOCKS_PER_ROW + c as i64 / 128;
        let j = c % 128;
        let q = (packed[(b * 34) as usize + 2 + j / 4] >> (2 * (j % 4))) & 3;
        base_vals[c] = f32::from(q) - 1.0;
    }
    let rotated = rotate_ref(&base_vals, &hadamard(BLOCK));
    let want: Vec<f32> = rotated.iter().zip(signs_for(HIDDEN)).map(|(v, s)| v * s).collect();
    let got: Vec<f32> = x.try_into()?;
    assert_close(&got, &want, 1e-5, 1e-5, "embedding");
    Ok(())
}

/// The Prism rotation pair at the real width (`prism.rs:151-183`): `H (s * x)`
/// then `s * (H z)` must return the input. 5120 = 5 blocks of 1024.
#[test]
fn prism_rotation() -> Result<(), ZyxError> {
    let x_vals = seq_vals(HIDDEN, 23);
    let signs = signs_for(HIDDEN);
    let x = Tensor::from_vec(x_vals.clone(), [1, HIDDEN])?;
    let sign_t = Tensor::from_vec(signs.clone(), [HIDDEN])?;
    let rot = Tensor::from_vec(hadamard(BLOCK), [BLOCK, BLOCK])?;

    let tape = Tape::new([&x, &sign_t, &rot])?;
    let forwarded = prism_forward(&x, &sign_t, &rot)?;
    let back = prism_inverse(&forwarded, &sign_t, &rot)?;
    tape.realize([&forwarded, &back])?;

    let signed: Vec<f32> = x_vals.iter().zip(&signs).map(|(v, s)| v * s).collect();
    let want = rotate_ref(&signed, &hadamard(BLOCK));
    let got: Vec<f32> = forwarded.try_into()?;
    assert_close(&got, &want, 1e-5, 1e-5, "prism forward");
    let round: Vec<f32> = back.try_into()?;
    assert_close(&round, &x_vals, 1e-5, 1e-5, "prism round trip");
    Ok(())
}
