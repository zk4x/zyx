// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Ternary-Bonsai-2-27B (`qwen35`) gated-delta SSM core at the checkpoint's
//! own geometry: the depthwise causal conv over the 10240 q/k/v channels, the
//! gated delta rule (state 128, 16 key heads / 48 value heads), the gate and
//! beta application, and the state read/write — decode (1 token) and prefill
//! (128 tokens).
//!
//! The two `*_zoo` tests trace the block exactly as
//! `examples/bonsai-27b/src/model.rs` writes it (minus the qkv/gate/ssm_out
//! projections, which belong to the matvec/GEMM class) and realize it, so the
//! autotune search prints one measured line per kernel the block actually
//! launches. `ZYX_DEBUG=1` gives those numbers, `ZYX_DEBUG=8` the post-opt IR
//! they belong to.
//!
//! The `delta_step_*` tests are the custom-kernel track: a hand-written
//! recurrent kernel plus the host reference it must reproduce, and the
//! measured one-launch time.

use zyx::kernel::autotune::BeamSearch;
use zyx::kernel::{Kernel, MemScope};
use zyx::{DType, Dev, Tape, Tensor, ZyxError};

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
/// Prefill width (`max_prefill`).
const SEQ: i64 = 128;
/// Chunk length of the chunked prefill.
const CHUNK: i64 = 64;
/// RMSNorm epsilon.
const EPS: f64 = 1e-6;

/// Blocks until the compute device has finished every submitted launch.
fn sync() -> Result<(), ZyxError> {
    match Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) {
        Some(dev) => dev.synchronize(),
        None => Ok(()),
    }
}

/// Deterministic values in `[-1, 1)`; `seed` varies the pattern per tensor.
fn seq_vals(n: i64, seed: i64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = (i * 17 + seed * 29) % 23;
            (v as f32 - 11.0) / 11.0
        })
        .collect()
}

fn t2(v: Vec<f32>, shape: [i64; 2]) -> Result<Tensor, ZyxError> {
    Tensor::from_vec(v, shape)
}

fn t3(v: Vec<f32>, shape: [i64; 3]) -> Result<Tensor, ZyxError> {
    Tensor::from_vec(v, shape)
}

/// Splits the mixed conv output into q/k/v head tensors (`model.rs:613-621`).
fn split_qkv(mixed: &Tensor, seq: i64) -> Result<(Tensor, Tensor, Tensor), ZyxError> {
    let q = mixed.narrow(1, 0i64, KEY_DIM)?.reshape([seq, K_HEADS, SSM_STATE])?;
    let k = mixed.narrow(1, KEY_DIM, KEY_DIM)?.reshape([seq, K_HEADS, SSM_STATE])?;
    let v = mixed.narrow(1, 2 * KEY_DIM, VALUE_DIM)?.reshape([seq, V_HEADS, V_HEAD_DIM])?;
    Ok((q, k, v))
}

/// Repeats `H_k` key heads to `H_v` value heads, llama.cpp's tiled order
/// (`model.rs:623-630`).
fn repeat_tiled(x: &Tensor, rep: i64) -> Result<Tensor, ZyxError> {
    let [seq, heads, dim] = x.dims::<3>()?;
    x.unsqueeze(1)?.expand([-1i64, rep, -1, -1])?.reshape([seq, heads * rep, dim])
}

/// The block's conv + q/k normalisation + gates (`model.rs:514-548`).
#[allow(clippy::too_many_arguments)]
fn conv_and_gates(
    qkv: &Tensor,
    conv_w: &Tensor,
    history: &Tensor,
    beta: &Tensor,
    alpha: &Tensor,
    ssm_dt: &Tensor,
    ssm_a: &Tensor,
    valid: &Tensor,
    seq: i64,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor), ZyxError> {
    let keep = Tensor::from(1.0f32) - Tensor::variable(5i64).equal(0i64)?.cast(DType::F32);
    let history = history * &keep;
    let cat = Tensor::cat([&history, qkv], 0)?;
    let mut mixed = Tensor::zeros([seq, CONV_DIM], DType::F32);
    for k in 0..CONV_KERNEL {
        let window = cat.narrow(0, k, seq)?;
        let weight = conv_w.narrow(1, k, 1i64)?.squeeze([1]);
        mixed = mixed + window * weight.unsqueeze(0)?;
    }
    let mixed = mixed.swish();
    let conv_state = cat.narrow(0, Tensor::variable(1i64), CONV_KERNEL - 1)?.contiguous()?;

    let (q, k, v) = split_qkv(&mixed, seq)?;
    let eps = EPS;
    let q = &q * ((q.square().sum_keepdim([-1])? + eps).rsqrt());
    let k = &k * ((k.square().sum_keepdim([-1])? + eps).rsqrt());
    let scale = Tensor::from(1.0f32 / (SSM_STATE as f32).sqrt());
    let q = q * scale;
    let rep = V_HEADS / K_HEADS;
    let q = repeat_tiled(&q, rep)?;
    let k = repeat_tiled(&k, rep)?;

    let b = beta.sigmoid() * valid.unsqueeze(1)?;
    let g = ((alpha + ssm_dt).softplus(1.0f32, 20.0f32) * ssm_a) * valid.unsqueeze(1)?;
    Ok((q, k, v, g, b, conv_state))
}

/// Decode: conv + gates + `delta_step` + the ssm norm/gate (`model.rs:514-563`).
///
/// Measurement harness: the autotune search runs with the caller's
/// `~/.config/zyx/config.json` budget, so this is run explicitly
/// (`--ignored`), with `ZYX_DEBUG=8` for the IR or `1` for the per-kernel
/// measured lines and `256` for the naive one-launch form.
#[test]
#[ignore = "measurement harness: the autotune search runs at the user's configured budget"]
fn ssm_core_decode_zoo() -> Result<(), ZyxError> {
    let qkv_vals = seq_vals(CONV_DIM, 5);
    let h_vals = seq_vals(HISTORY * CONV_DIM, 4);
    let w_vals = seq_vals(CONV_DIM * CONV_KERNEL, 6);
    let beta_vals = seq_vals(V_HEADS, 2);
    let alpha_vals = seq_vals(V_HEADS, 3);
    let dt_vals = seq_vals(V_HEADS, 7);
    let a_vals = seq_vals(V_HEADS, 8);
    let z_vals = seq_vals(V_HEADS * V_HEAD_DIM, 9);
    let state_vals = seq_vals(V_HEADS * SSM_STATE * SSM_STATE, 10);
    let norm_vals: Vec<f32> = (0..SSM_STATE).map(|i| 0.5 + (i % 5) as f32 * 0.25).collect();

    let qkv = t2(qkv_vals, [1, CONV_DIM])?;
    let history = t2(h_vals, [HISTORY, CONV_DIM])?;
    let conv_w = t2(w_vals, [CONV_DIM, CONV_KERNEL])?;
    let beta = t2(beta_vals, [1, V_HEADS])?;
    let alpha = t2(alpha_vals, [1, V_HEADS])?;
    let ssm_dt = Tensor::from_vec(dt_vals, [V_HEADS])?;
    let ssm_a = Tensor::from_vec(a_vals, [V_HEADS])?;
    let valid = Tensor::from_vec(vec![1.0f32], [1])?;
    let z = t3(z_vals, [1, V_HEADS, V_HEAD_DIM])?;
    let state = t3(state_vals, [V_HEADS, SSM_STATE, SSM_STATE])?;
    let norm = Tensor::from_vec(norm_vals, [SSM_STATE])?;
    let pos = Tensor::variable(1i64);

    let tape = Tape::new([&qkv, &history, &conv_w, &beta, &alpha, &ssm_dt, &ssm_a, &valid, &z, &state, &norm, &pos])?;

    let (q, k, v, g, b, conv_state) = conv_and_gates(&qkv, &conv_w, &history, &beta, &alpha, &ssm_dt, &ssm_a, &valid, 1)?;
    let q = q.squeeze([0]);
    let k = k.squeeze([0]);
    let v = v.squeeze([0]);
    let g = g.squeeze([0]).exp();
    let b = b.squeeze([0]);

    println!("ssm_core_decode: conv + delta_step + norm");
    let kv = (state.clone() * k.clone().unsqueeze(2)?).sum([1])?;
    let delta = (v - g.clone().unsqueeze(1)? * kv) * b.unsqueeze(1)?;
    let new_state = state * g.clone().unsqueeze(1)?.unsqueeze(1)? + k.unsqueeze(2)? * delta.unsqueeze(1)?;
    let core = (new_state.clone() * q.unsqueeze(2)?).sum([1])?;
    let core = core.unsqueeze(0)?;
    let normed = &core * ((&core * &core).mean_keepdim([-1])? + Tensor::from(EPS).cast(DType::F32)).rsqrt() * &norm;
    let gate = z.swish();
    let out = normed * gate;
    let start = std::time::Instant::now();
    tape.realize([&out, &new_state, &conv_state])?;
    sync()?;
    println!("ssm_core_decode wall {:.3} ms", start.elapsed().as_secs_f64() * 1e3);
    Ok(())
}

/// Prefill: conv + gates + `delta_chunked` + the ssm norm/gate at seq 128.
///
/// Measurement harness, run explicitly with `--ignored` (see
/// `ssm_core_decode_zoo`).
#[test]
#[ignore = "measurement harness: the autotune search runs at the user's configured budget"]
fn ssm_core_prefill_zoo() -> Result<(), ZyxError> {
    let h_vals = seq_vals(HISTORY * CONV_DIM, 4);
    let qkv_vals = seq_vals(SEQ * CONV_DIM, 5);
    let w_vals = seq_vals(CONV_DIM * CONV_KERNEL, 6);
    let beta_vals = seq_vals(SEQ * V_HEADS, 2);
    let alpha_vals = seq_vals(SEQ * V_HEADS, 3);
    let dt_vals = seq_vals(V_HEADS, 7);
    let a_vals = seq_vals(V_HEADS, 8);
    let z_vals = seq_vals(SEQ * V_HEADS * V_HEAD_DIM, 9);
    let state_vals = seq_vals(V_HEADS * SSM_STATE * SSM_STATE, 10);
    let norm_vals: Vec<f32> = (0..SSM_STATE).map(|i| 0.5 + (i % 5) as f32 * 0.25).collect();

    let qkv = t2(qkv_vals, [SEQ, CONV_DIM])?;
    let history = t2(h_vals, [HISTORY, CONV_DIM])?;
    let conv_w = t2(w_vals, [CONV_DIM, CONV_KERNEL])?;
    let beta = t2(beta_vals, [SEQ, V_HEADS])?;
    let alpha = t2(alpha_vals, [SEQ, V_HEADS])?;
    let ssm_dt = Tensor::from_vec(dt_vals, [V_HEADS])?;
    let ssm_a = Tensor::from_vec(a_vals, [V_HEADS])?;
    let valid = Tensor::from_vec(vec![1.0f32], [1])?;
    let z = t3(z_vals, [SEQ, V_HEADS, V_HEAD_DIM])?;
    let state = t3(state_vals, [V_HEADS, SSM_STATE, SSM_STATE])?;
    let norm = Tensor::from_vec(norm_vals, [SSM_STATE])?;
    let pos = Tensor::variable(1i64);

    let tape = Tape::new([&qkv, &history, &conv_w, &beta, &alpha, &ssm_dt, &ssm_a, &valid, &z, &state, &norm, &pos])?;

    let (q, k, v, g, b, conv_state) = conv_and_gates(&qkv, &conv_w, &history, &beta, &alpha, &ssm_dt, &ssm_a, &valid, SEQ)?;
    println!("ssm_core_prefill: conv + delta_chunked + norm at seq {SEQ}");
    let (core, new_state) = delta_chunked(&q, &k, &v, &g, &b, &state)?;
    // `core` is the transposed, trimmed `[seq, V_HEADS, V_HEAD_DIM]` value.
    let normed = &core * ((&core * &core).mean_keepdim([-1])? + Tensor::from(EPS).cast(DType::F32)).rsqrt() * &norm;
    let out = normed * z.swish();
    let start = std::time::Instant::now();
    tape.realize([&out, &new_state, &conv_state])?;
    sync()?;
    println!("ssm_core_prefill wall {:.3} ms", start.elapsed().as_secs_f64() * 1e3);
    Ok(())
}

/// Decode, delta rule only: `delta_step` + the state write-back. Isolated so a
/// single run stays short and the search's measured numbers belong to these
/// kernels alone. Measurement harness, run explicitly with `--ignored`.
#[test]
#[ignore = "measurement harness: the autotune search runs at the user's configured budget"]
fn ssm_delta_step_zoo() -> Result<(), ZyxError> {
    let q_vals = seq_vals(V_HEADS * SSM_STATE, 13);
    let k_vals = seq_vals(V_HEADS * SSM_STATE, 14);
    let v_vals = seq_vals(V_HEADS * SSM_STATE, 15);
    let g_vals: Vec<f32> = (0..V_HEADS).map(|i| -0.01 * ((i % 4) as f32 + 1.0)).collect();
    let b_vals: Vec<f32> = (0..V_HEADS).map(|i| 0.25 + 0.005 * (i % 3) as f32).collect();
    let state_vals = seq_vals(V_HEADS * SSM_STATE * SSM_STATE, 16);

    let q = t3(q_vals, [1, V_HEADS, SSM_STATE])?;
    let k = t3(k_vals, [1, V_HEADS, SSM_STATE])?;
    let v = t3(v_vals, [1, V_HEADS, SSM_STATE])?;
    let g = t2(g_vals, [1, V_HEADS])?;
    let b = t2(b_vals, [1, V_HEADS])?;
    let state = t3(state_vals, [V_HEADS, SSM_STATE, SSM_STATE])?;

    let tape = Tape::new([&q, &k, &v, &g, &b, &state])?;
    println!("ssm_delta_step: state=[{V_HEADS}x{SSM_STATE}x{SSM_STATE}]");

    let q = q.squeeze([0]);
    let k = k.squeeze([0]);
    let v = v.squeeze([0]);
    let g = g.squeeze([0]).exp();
    let b = b.squeeze([0]);
    let kv = (state.clone() * k.clone().unsqueeze(2)?).sum([1])?;
    let delta = (v - g.clone().unsqueeze(1)? * kv) * b.unsqueeze(1)?;
    let new_state = state.clone() * g.unsqueeze(1)?.unsqueeze(1)? + k.unsqueeze(2)? * delta.unsqueeze(1)?;
    let core = (new_state.clone() * q.unsqueeze(2)?).sum([1])?;
    tape.realize([&core, &new_state])?;
    Ok(())
}

/// Decode, conv only: history roll, four taps, SiLU, the q/k partial norms,
/// the gate and beta activation. Measurement harness, run explicitly with
/// `--ignored`.
#[test]
#[ignore = "measurement harness: the autotune search runs at the user's configured budget"]
fn ssm_conv_zoo() -> Result<(), ZyxError> {
    let qkv_vals = seq_vals(CONV_DIM, 5);
    let h_vals = seq_vals(HISTORY * CONV_DIM, 4);
    let w_vals = seq_vals(CONV_DIM * CONV_KERNEL, 6);
    let beta_vals = seq_vals(V_HEADS, 2);
    let alpha_vals = seq_vals(V_HEADS, 3);
    let dt_vals = seq_vals(V_HEADS, 7);
    let a_vals = seq_vals(V_HEADS, 8);

    let qkv = t2(qkv_vals, [1, CONV_DIM])?;
    let history = t2(h_vals, [HISTORY, CONV_DIM])?;
    let conv_w = t2(w_vals, [CONV_DIM, CONV_KERNEL])?;
    let beta = t2(beta_vals, [1, V_HEADS])?;
    let alpha = t2(alpha_vals, [1, V_HEADS])?;
    let ssm_dt = Tensor::from_vec(dt_vals, [V_HEADS])?;
    let ssm_a = Tensor::from_vec(a_vals, [V_HEADS])?;
    let valid = Tensor::from_vec(vec![1.0f32], [1])?;
    let pos = Tensor::variable(1i64);

    let tape = Tape::new([&qkv, &history, &conv_w, &beta, &alpha, &ssm_dt, &ssm_a, &valid, &pos])?;
    println!("ssm_conv: conv_dim={CONV_DIM} kernel={CONV_KERNEL}");
    let (q, k, v, g, b, conv_state) = conv_and_gates(&qkv, &conv_w, &history, &beta, &alpha, &ssm_dt, &ssm_a, &valid, 1)?;
    tape.realize([&conv_state, &q, &k, &v, &g, &b])?;
    Ok(())
}

/// Llama.cpp's chunked gated delta rule (`model.rs:657-...`), prefill.
fn delta_chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    b: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor), ZyxError> {
    let [seq, heads, dim] = q.dims::<3>()?;
    let seq = seq.item::<i64>();
    let h = heads.item::<i64>();
    let d = dim.item::<i64>();
    let pad = (CHUNK - seq % CHUNK) % CHUNK;
    let chunks = (seq + pad) / CHUNK;

    let pad_t = |x: &Tensor| -> Result<Tensor, ZyxError> {
        if pad == 0 {
            return Ok(x.clone());
        }
        let shape = x.shape();
        let mut pad_shape = shape.clone();
        pad_shape[1] = Tensor::from(pad);
        let zeros = Tensor::zeros(pad_shape, DType::F32);
        Tensor::cat([x, &zeros], 1)
    };

    let q = q.transpose(0, 1)?;
    let k = k.transpose(0, 1)?;
    let v = v.transpose(0, 1)?;
    let g = g.transpose(0, 1)?;
    let b = b.transpose(0, 1)?;
    let q = pad_t(&q)?;
    let k = pad_t(&k).and_then(|x| x.reshape([h, chunks, CHUNK, d]))?;
    let v = pad_t(&v).and_then(|x| x.reshape([h, chunks, CHUNK, d]))?;
    let g = pad_t(&g).and_then(|x| x.reshape([h, chunks, CHUNK]))?;
    let b = pad_t(&b).and_then(|x| x.reshape([h, chunks, CHUNK]))?;
    let q = q.reshape([h, chunks, CHUNK, d])?;

    let gcs = g.cumsum(2)?;
    let gexp = gcs.exp();
    let gi = gcs.unsqueeze(3)?;
    let gj = gcs.unsqueeze(2)?;
    let decay = (gi - gj).exp() * lower(CHUNK, 0)?;
    let strict = lower(CHUNK, -1)?;

    let b3 = b.unsqueeze(3)?;
    let v_b = &v * &b3;
    let k_b = &k * &b3;

    let a = k_b.matmul(&k.transpose(2, 3)?)? * &decay * &strict;
    let n = -a;
    let identity = Tensor::eye(CHUNK, DType::F32)
        .unsqueeze(0)?
        .unsqueeze(0)?
        .expand([h, chunks, CHUNK, CHUNK])?;
    let mut m = identity;
    let mut npow = n.clone();
    for _ in 0..6 {
        let next = &m + npow.matmul(&m)?;
        m = next;
        npow = npow.matmul(&npow)?;
    }

    let v_new = m.matmul(&v_b)?;
    let k_cd = m.matmul(&(k_b * gexp.unsqueeze(3)?))?;
    let q_g = &q * gexp.unsqueeze(3)?;

    let mut state = state.clone();
    let mut cores = Vec::with_capacity(usize::try_from(chunks).expect("chunk count fits usize"));
    for chunk in 0..chunks {
        let qc = q_g.narrow(1, chunk, 1i64)?.squeeze([1]);
        let kc = k.narrow(1, chunk, 1i64)?.squeeze([1]);
        let kcd = k_cd.narrow(1, chunk, 1i64)?.squeeze([1]);
        let vnc = v_new.narrow(1, chunk, 1i64)?.squeeze([1]);
        let gc = gcs.narrow(1, chunk, 1i64)?.squeeze([1]);
        let dc = decay.narrow(1, chunk, 1i64)?.squeeze([1]);

        let a2 = qc.matmul(&kc.transpose(1, 2)?)? * &dc * &strict;
        let v_prime = kcd.matmul(&state)?;
        let v_new_c = vnc - v_prime;
        let core = qc.matmul(&state)? + a2.matmul(&v_new_c)?;

        let g_last = gc.narrow(1, CHUNK - 1, 1i64)?;
        let kg = &kc * (&g_last - &gc).exp().unsqueeze(2)?;
        state = state * g_last.exp().unsqueeze(2)? + kg.transpose(1, 2)?.matmul(&v_new_c)?;
        cores.push(core);
    }

    let core = Tensor::cat(&cores, 1)?;
    let core = core.transpose(0, 1)?.narrow(0, 0i64, seq)?;
    Ok((core, state))
}

/// A `[n, n]` F32 lower-triangular mask (`diagonal` = 0 keeps the diagonal).
fn lower(n: i64, diagonal: i64) -> Result<Tensor, ZyxError> {
    let mut data = vec![0.0f32; (n * n) as usize];
    for i in 0..n {
        for j in 0..n {
            if j <= i + diagonal {
                data[(i * n + j) as usize] = 1.0;
            }
        }
    }
    Tensor::from_vec(data, [n, n])
}

// ---- the custom-kernel track -------------------------------------------
//
// The state is `[V_HEADS, SSM_STATE, SSM_STATE]`, i.e. `S[h][a][dd]` with `a`
// the contraction axis of `kv` and `dd` the output axis. One thread per
// `(h, dd)` owns a whole column of the state: it walks `a` once to build
// `kv[dd]`, then once more to write `S'[a][dd] = g*S[a][dd] + k[a]*delta[dd]`
// and to build `core[dd]`. No reduction crosses threads: `a` is the serial
// axis and `dd` the coalesced one (consecutive threads read consecutive
// addresses), which is why the zyx layout needs no warp shuffle.

/// `x * sx + y * sy + z`, built one `mad` at a time (the builder borrows
/// `&mut self`, so nested calls cannot share one expression).
fn mad3(k: &mut Kernel, x: zyx::kernel::OpId, sx: i64, y: zyx::kernel::OpId, sy: i64, z: zyx::kernel::OpId) -> zyx::kernel::OpId {
    let t = k.mad(x, sx, z);
    k.mad(y, sy, t)
}

/// `h * SSM_STATE + a`, the index of one row of the `[V_HEADS, SSM_STATE]`
/// operands.
fn row_idx(k: &mut Kernel, h: zyx::kernel::OpId, a: zyx::kernel::OpId) -> zyx::kernel::OpId {
    k.mad(h, SSM_STATE, a)
}

/// First naive kernel: `kv[h][dd] = sum_a S[h][a][dd] * k[h][a]`.
fn naive_kv(dev: Dev) -> Kernel {
    let mut k = Kernel::new(dev);
    let state = k.param(DType::F32);
    let kk = k.param(DType::F32);
    let kv = k.param_mut(DType::F32);
    let h = k.group_range(0, V_HEADS);
    let dd = k.local_range(0, u32::try_from(SSM_STATE).expect("state width fits u32"));
    let acc = k.storage(DType::F32, MemScope::Register, 1);
    let zero = k.const_val(0.0f32);
    k.store(acc, zero, 0i64);
    k.loop_over(SSM_STATE, |k, a| {
        let s_idx = mad3(k, h, SSM_STATE * SSM_STATE, a, SSM_STATE, dd);
        let s = k.load(state, s_idx);
        let k_idx = row_idx(k, h, a);
        let kv_a = k.load(kk, k_idx);
        let prev = k.load(acc, 0i64);
        let prod = k.mul(s, kv_a);
        let sum = k.add(prod, prev);
        k.store(acc, sum, 0i64);
    });
    let out = k.load(acc, 0i64);
    let o_idx = row_idx(&mut k, h, dd);
    k.store(kv, out, o_idx);
    k.verify();
    k
}

/// Second naive kernel: `delta[h][dd] = (v[h][dd] - exp(g[h])*kv[h][dd]) * b[h]`.
fn naive_delta(dev: Dev) -> Kernel {
    let mut k = Kernel::new(dev);
    let v = k.param(DType::F32);
    let g = k.param(DType::F32);
    let b = k.param(DType::F32);
    let kv = k.param(DType::F32);
    let delta = k.param_mut(DType::F32);
    let h = k.group_range(0, V_HEADS);
    let dd = k.local_range(0, u32::try_from(SSM_STATE).expect("state width fits u32"));
    let idx = row_idx(&mut k, h, dd);
    let vv = k.load(v, idx);
    let kvv = k.load(kv, idx);
    let gv = k.load(g, h);
    let bv = k.load(b, h);
    let gexp = k.exp(gv);
    let scaled = k.mul(gexp, kvv);
    let d = k.sub(vv, scaled);
    let d = k.mul(d, bv);
    k.store(delta, d, idx);
    k.verify();
    k
}

/// Third naive kernel: the state write and the read out.
fn naive_state(dev: Dev) -> Kernel {
    let mut k = Kernel::new(dev);
    let state = k.param(DType::F32);
    let kk = k.param(DType::F32);
    let delta = k.param(DType::F32);
    let q = k.param(DType::F32);
    let g = k.param(DType::F32);
    let core = k.param_mut(DType::F32);
    let state_out = k.param_mut(DType::F32);
    let h = k.group_range(0, V_HEADS);
    let dd = k.local_range(0, u32::try_from(SSM_STATE).expect("state width fits u32"));
    let o_idx = row_idx(&mut k, h, dd);
    let dv = k.load(delta, o_idx);
    let gv = k.load(g, h);
    let gexp = k.exp(gv);
    let acc = k.storage(DType::F32, MemScope::Register, 1);
    let zero = k.const_val(0.0f32);
    k.store(acc, zero, 0i64);
    k.loop_over(SSM_STATE, |k, a| {
        let s_idx = mad3(k, h, SSM_STATE * SSM_STATE, a, SSM_STATE, dd);
        let s = k.load(state, s_idx);
        let k_idx = row_idx(k, h, a);
        let ka = k.load(kk, k_idx);
        let gs = k.mul(gexp, s);
        let kd = k.mul(ka, dv);
        let new = k.add(gs, kd);
        k.store(state_out, new, s_idx);
        let qa = k.load(q, k_idx);
        let prev = k.load(acc, 0i64);
        let part = k.mul(new, qa);
        let sum = k.add(part, prev);
        k.store(acc, sum, 0i64);
    });
    let out = k.load(acc, 0i64);
    k.store(core, out, o_idx);
    k.verify();
    k
}

/// The gated delta rule as one kernel: `kv`, `delta`, the state write and the
/// read out share the two passes over the state column, so the column's
/// `kv`/`delta`/`core` never round-trip through DRAM.
fn fast_delta(dev: Dev) -> Kernel {
    let mut k = Kernel::new(dev);
    let state = k.param(DType::F32);
    let q = k.param(DType::F32);
    let kk = k.param(DType::F32);
    let v = k.param(DType::F32);
    let g = k.param(DType::F32);
    let b = k.param(DType::F32);
    let core = k.param_mut(DType::F32);
    let state_out = k.param_mut(DType::F32);
    let h = k.group_range(0, V_HEADS);
    let dd = k.local_range(0, u32::try_from(SSM_STATE).expect("state width fits u32"));
    let idx = row_idx(&mut k, h, dd);
    let gv = k.load(g, h);
    let gexp = k.exp(gv);
    let bv = k.load(b, h);

    // Pass 1: kv[dd] = sum_a S[a][dd] * k[a].
    let kv = k.storage(DType::F32, MemScope::Register, 1);
    let zero = k.const_val(0.0f32);
    k.store(kv, zero, 0i64);
    k.loop_over(SSM_STATE, |k, a| {
        let s_idx = mad3(k, h, SSM_STATE * SSM_STATE, a, SSM_STATE, dd);
        let s = k.load(state, s_idx);
        let k_idx = row_idx(k, h, a);
        let ka = k.load(kk, k_idx);
        let prev = k.load(kv, 0i64);
        let prod = k.mul(s, ka);
        let sum = k.add(prod, prev);
        k.store(kv, sum, 0i64);
    });
    let kvv = k.load(kv, 0i64);
    let vv = k.load(v, idx);
    let gkv = k.mul(gexp, kvv);
    let diff = k.sub(vv, gkv);
    let delta = k.mul(diff, bv);

    // Pass 2: S'[a][dd] = g*S[a][dd] + k[a]*delta[dd], core[dd] = sum_a S'*q[a].
    let acc = k.storage(DType::F32, MemScope::Register, 1);
    k.store(acc, zero, 0i64);
    k.loop_over(SSM_STATE, |k, a| {
        let s_idx = mad3(k, h, SSM_STATE * SSM_STATE, a, SSM_STATE, dd);
        let s = k.load(state, s_idx);
        let k_idx = row_idx(k, h, a);
        let ka = k.load(kk, k_idx);
        let gs = k.mul(gexp, s);
        let kd = k.mul(ka, delta);
        let new = k.add(gs, kd);
        k.store(state_out, new, s_idx);
        let qa = k.load(q, k_idx);
        let prev = k.load(acc, 0i64);
        let part = k.mul(new, qa);
        let sum = k.add(part, prev);
        k.store(acc, sum, 0i64);
    });
    let out = k.load(acc, 0i64);
    k.store(core, out, idx);
    k.verify();
    k
}

/// The same rule with the state column split four ways: each of the four
/// `a`-quarters of a column lives in one warp-group of the block, the partial
/// `kv`/`core` cross the group boundary through `MemScope::Local`. The column
/// is read once and written once, so the state's 6.3 MB is the kernel's only
/// DRAM traffic.
fn fast_delta_shared(dev: Dev) -> Kernel {
    // The value axis `dd` is split across blocks in groups of 32 (one warp's
    // worth, so a warp reads 32 consecutive `dd` = 128 B), and the
    // contraction axis `a` across four threads of the block. The four
    // `a`-slices' partial `kv`/`core` meet in `MemScope::Local`.
    const DD_GROUPS: i64 = 2;
    const SLICES: i64 = 4;
    let qw = SSM_STATE / DD_GROUPS;
    let part = SSM_STATE / SLICES;
    let mut k = Kernel::new(dev);
    let state = k.param(DType::F32);
    let q = k.param(DType::F32);
    let kk = k.param(DType::F32);
    let v = k.param(DType::F32);
    let g = k.param(DType::F32);
    let b = k.param(DType::F32);
    let core = k.param_mut(DType::F32);
    let state_out = k.param_mut(DType::F32);
    let h = k.group_range(0, V_HEADS);
    let dg = k.group_range(1, DD_GROUPS);
    let t = k.local_range(0, u32::try_from(SLICES * qw).expect("block fits u32"));
    let lane = k.mod_(t, qw);
    let w = k.div(t, qw);
    let dgo = k.mul(dg, qw);
    let dd = k.add(dgo, lane);
    let idx = row_idx(&mut k, h, dd);
    let gv = k.load(g, h);
    let gexp = k.exp(gv);
    let bv = k.load(b, h);

    // This thread's slice of the a axis starts at `a = w * part`.
    let base_a = k.mul(w, part);
    let base_off = k.mul(base_a, SSM_STATE);
    let col0 = k.mul(h, SSM_STATE * SSM_STATE);
    let col0 = k.add(col0, base_off);
    let col0 = k.add(col0, dd);
    let row_h = k.mul(h, SSM_STATE);
    let row0 = k.add(row_h, base_a);

    let smem = k.storage(DType::F32, MemScope::Local, SLICES * qw);
    let slot = t;

    // Pass 1: this slice's partial `kv`, keeping `S` for the write.
    let mut s_vals = Vec::with_capacity(usize::try_from(part).expect("part fits usize"));
    let mut kv = k.const_val(0.0f32);
    let mut i = 0i64;
    while i < part {
        let step = k.const_idx(i * SSM_STATE);
        let s_idx = k.add(col0, step);
        let s = k.load(state, s_idx);
        let ai = k.const_idx(i);
        let k_at = k.add(row0, ai);
        let ka = k.load(kk, k_at);
        let prod = k.mul(s, ka);
        kv = k.add(kv, prod);
        s_vals.push((s_idx, s));
        i += 1;
    }
    k.store(smem, kv, slot);
    k.barrier();
    let mut total = k.load(smem, lane);
    let mut j = 1i64;
    while j < SLICES {
        let js = k.const_idx(j * qw);
        let at = k.add(js, lane);
        let part_kv = k.load(smem, at);
        total = k.add(total, part_kv);
        j += 1;
    }
    let vv = k.load(v, idx);
    let gkv = k.mul(gexp, total);
    let diff = k.sub(vv, gkv);
    let delta = k.mul(diff, bv);
    k.barrier();

    // Pass 2: write the slice's new state and build this slice's `core`.
    let mut acc = k.const_val(0.0f32);
    for (n, (s_idx, s)) in s_vals.iter().enumerate() {
        let ni = k.const_idx(i64::try_from(n).expect("n fits i64"));
        let k_at = k.add(row0, ni);
        let ka = k.load(kk, k_at);
        let gs = k.mul(gexp, *s);
        let kd = k.mul(ka, delta);
        let new = k.add(gs, kd);
        k.store(state_out, new, *s_idx);
        let qa = k.load(q, k_at);
        let part = k.mul(new, qa);
        acc = k.add(acc, part);
    }
    // Second storage for the read-out partials: reusing `smem` would make the
    // epilogue CSE the second `Load` with the first one (same address, across
    // the barrier), which serves the stale `kv` partial as `core`.
    let smem2 = k.storage(DType::F32, MemScope::Local, SLICES * qw);
    k.store(smem2, acc, slot);
    k.barrier();
    let mut out = k.load(smem2, lane);
    let mut m = 1i64;
    while m < SLICES {
        let ms = k.const_idx(m * qw);
        let at = k.add(ms, lane);
        let part_core = k.load(smem2, at);
        out = k.add(out, part_core);
        m += 1;
    }
    k.store(core, out, idx);
    k.verify();
    k
}
/// Scalar `delta_step` reference, `state` as `[V_HEADS, SSM_STATE, SSM_STATE]`.
fn delta_ref(q: &[f32], k: &[f32], v: &[f32], g: &[f32], b: &[f32], state: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let heads = V_HEADS as usize;
    let d = SSM_STATE as usize;
    let mut core = vec![0f32; heads * d];
    let mut out = vec![0f32; heads * d * d];
    for hh in 0..heads {
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

/// The compute device the custom kernels bind to, or `None` on a host box.
fn kernel_dev() -> Option<Dev> {
    Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host))
}

/// The checkpoint's delta-rule operands as device tensors plus the host
/// reference they must reproduce.
fn delta_operands(dev: Dev) -> Result<(Vec<Tensor>, (Vec<f32>, Vec<f32>), Vec<f32>), ZyxError> {
    let q_vals = seq_vals(V_HEADS * SSM_STATE, 13);
    let k_vals = seq_vals(V_HEADS * SSM_STATE, 14);
    let v_vals = seq_vals(V_HEADS * SSM_STATE, 15);
    let g_vals: Vec<f32> = (0..V_HEADS).map(|i| -0.01 * ((i % 4) as f32 + 1.0)).collect();
    let b_vals: Vec<f32> = (0..V_HEADS).map(|i| 0.25 + 0.005 * (i % 3) as f32).collect();
    let state_vals = seq_vals(V_HEADS * SSM_STATE * SSM_STATE, 16);
    let want = delta_ref(&q_vals, &k_vals, &v_vals, &g_vals, &b_vals, &state_vals);
    let to = |vals: Vec<f32>, shape: [i64; 2]| -> Result<Tensor, ZyxError> { Tensor::from_vec(vals, shape)?.to(dev) };
    let tensors = vec![
        to(state_vals.clone(), [V_HEADS * SSM_STATE, SSM_STATE])?,
        to(q_vals, [V_HEADS, SSM_STATE])?,
        to(k_vals, [V_HEADS, SSM_STATE])?,
        to(v_vals, [V_HEADS, SSM_STATE])?,
        to(g_vals, [V_HEADS, 1])?,
        to(b_vals, [V_HEADS, 1])?,
    ];
    Ok((tensors, want, state_vals))
}

/// Host reference comparison with the test file's tolerances.
fn assert_close(got: &[f32], want: &[f32], abs: f32, rel: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, e)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - e).abs();
        assert!(d <= abs + rel * e.abs(), "{what}[{i}]: zyx {g} != {e} (diff {d})");
    }
}

/// Diagnostic: does a 512-thread block reduce four shared-memory slices?
#[test]
fn smem_reduce_smoke() -> Result<(), ZyxError> {
    let Some(dev) = kernel_dev() else {
        return Ok(());
    };
    const N: i64 = 512;
    let mut k = Kernel::new(dev);
    let out = k.param_mut(DType::F32);
    let t = k.local_range(0, u32::try_from(N).expect("block fits u32"));
    let dd = k.mod_(t, SSM_STATE);
    let w = k.div(t, SSM_STATE);
    let smem = k.storage(DType::F32, MemScope::Local, N);
    let tv = k.cast(t, DType::F32);
    k.store(smem, tv, t);
    k.barrier();
    let mut total = k.load(smem, dd);
    let mut j = 1i64;
    while j < 4 {
        let js = k.const_idx(j * SSM_STATE);
        let at = k.add(js, dd);
        let v = k.load(smem, at);
        total = k.add(total, v);
        j += 1;
    }
    let wf = k.cast(w, DType::F32);
    let check = k.const_val(0.0f32);
    let _ = check;
    k.store(out, total, dd);
    let _ = wf;
    k.verify();
    let got: Vec<f32> = k.compile()?.forward1(&[], vec![Tensor::from(SSM_STATE)])?.try_into()?;
    let want: Vec<f32> = (0..SSM_STATE).map(|d| (d + d + 128 + d + 256 + d + 384) as f32).collect();
    assert_close(&got, &want, 0.0, 0.0, "smem reduce");
    println!("smem_reduce_smoke: 4-slice shared reduce ok");
    Ok(())
}

/// Naive lowering: the three kernels the graph emits for the decode delta
/// rule, measured one launch each against the host reference.
#[test]
fn delta_step_naive() -> Result<(), ZyxError> {
    let Some(dev) = kernel_dev() else {
        return Ok(());
    };
    let (t, want, _) = delta_operands(dev)?;
    let (state, q, kk, v, g, b) = (&t[0], &t[1], &t[2], &t[3], &t[4], &t[5]);
    let kv_buf = Tensor::zeros([V_HEADS, SSM_STATE], DType::F32).to(dev)?;
    let delta_buf = Tensor::zeros([V_HEADS, SSM_STATE], DType::F32).to(dev)?;
    let core_buf = Tensor::zeros([V_HEADS, SSM_STATE], DType::F32).to(dev)?;
    let state_buf = Tensor::zeros([V_HEADS * SSM_STATE, SSM_STATE], DType::F32).to(dev)?;

    // Values, chained through the three kernels.
    let kv_k = naive_kv(dev).compile()?;
    let kv_t = kv_k.forward(&[state, kk], vec![[V_HEADS, SSM_STATE]])?.pop().unwrap();
    let d_k = naive_delta(dev).compile()?;
    let d_t = d_k.forward(&[v, g, b, &kv_t], vec![[V_HEADS, SSM_STATE]])?.pop().unwrap();
    let s_k = naive_state(dev).compile()?;
    let out = s_k.forward(&[state, kk, &d_t, q, g], vec![[V_HEADS, SSM_STATE], [V_HEADS * SSM_STATE, SSM_STATE]])?;
    let got_core: Vec<f32> = out[0].clone().try_into()?;
    let got_state: Vec<f32> = out[1].clone().try_into()?;
    assert_close(&got_core, &want.0, 2e-5, 1e-6, "naive core");
    assert_close(&got_state, &want.1, 2e-5, 1e-6, "naive state");

    // One measured launch each.
    let n_kv = BeamSearch::new().run([naive_kv(dev)], &[state, kk, &kv_buf], &[], |_| {}, |_| 0)?.1;
    let n_d = BeamSearch::new().run([naive_delta(dev)], &[v, g, b, &kv_buf, &delta_buf], &[], |_| {}, |_| 0)?.1;
    let n_s =
        BeamSearch::new().run([naive_state(dev)], &[state, kk, &delta_buf, q, g, &core_buf, &state_buf], &[], |_| {}, |_| 0)?.1;
    let total = n_kv + n_d + n_s;
    let bytes = 2.0 * (V_HEADS * SSM_STATE * SSM_STATE) as f64 * 4.0;
    println!(
        "delta_step naive: kv {:.1} us, delta {:.1} us, state+core {:.1} us, total {:.1} us ({:.0} GB/s)",
        n_kv as f64 * 1e-3,
        n_d as f64 * 1e-3,
        n_s as f64 * 1e-3,
        total as f64 * 1e-3,
        bytes / total as f64
    );
    Ok(())
}

/// One kernel for the whole decode delta rule, measured and validated.
#[test]
fn delta_step_fast() -> Result<(), ZyxError> {
    let Some(dev) = kernel_dev() else {
        return Ok(());
    };
    let (t, want, _) = delta_operands(dev)?;
    let (state, q, kk, v, g, b) = (&t[0], &t[1], &t[2], &t[3], &t[4], &t[5]);
    let core_buf = Tensor::zeros([V_HEADS, SSM_STATE], DType::F32).to(dev)?;
    let state_buf = Tensor::zeros([V_HEADS * SSM_STATE, SSM_STATE], DType::F32).to(dev)?;

    for (name, make) in [("one-read-split", fast_delta_shared as fn(Dev) -> Kernel), ("two-pass", fast_delta as fn(Dev) -> Kernel)] {
        let compiled = make(dev).compile()?;
        let out = compiled.forward(
            &[state, q, kk, v, g, b],
            vec![vec![V_HEADS, SSM_STATE], vec![V_HEADS * SSM_STATE, SSM_STATE]],
        )?;
        let got_core: Vec<f32> = out[0].clone().try_into()?;
        let got_state: Vec<f32> = out[1].clone().try_into()?;
        assert_close(&got_core, &want.0, 2e-5, 1e-6, name);
        assert_close(&got_state, &want.1, 2e-5, 1e-6, name);

        let nanos = BeamSearch::new()
            .run([make(dev)], &[state, q, kk, v, g, b, &core_buf, &state_buf], &[], |_| {}, |_| 0)?
            .1;
        let bytes = 2.0 * (V_HEADS * SSM_STATE * SSM_STATE) as f64 * 4.0;
        println!(
            "delta_step fast ({name}): {:.1} us, {:.1} GB/s state r+w, core+state match",
            nanos as f64 * 1e-3,
            bytes / nanos as f64
        );
    }
    Ok(())
}


