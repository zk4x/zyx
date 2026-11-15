// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The gated delta rule: qwen35's linear-attention core, both phases.
//!
//! Decode runs one token against the recurrent state ([`delta_step`]); prefill
//! runs the chunked form with the six-step Newton-Schulz inverse
//! ([`delta_chunked`]). The recurrent state is `[48, 128, 128]` F32 per block —
//! 3 MB read and written per token, the largest non-weight traffic in the
//! model — so this bench is the state update, not the projections around it.
//!
//! Both routines are ports of the reference in
//! `examples/bonsai-27b/src/model.rs`; the shapes are the checkpoint's. Each
//! phase is traced on its own [`Tape`] — one tape per timed kernel — so the
//! measured launches are the graph kernels the port runs.

use zyx::{DType, Tape, Tensor, ZyxError};

use super::shapes::{CHUNK, SEQ, SSM_STATE, SSM_V_HEADS};
use crate::bench::bench_dev;

/// Decode then prefill: one bench entry for the delta rule, both phases.
pub fn bench() -> Result<(), ZyxError> {
    step()?;
    chunked()
}

/// One decode token: `delta_step` at seq 1.
fn step() -> Result<(), ZyxError> {
    let dev = bench_dev();
    let make = |shape: [i64; 3]| Tensor::rand(shape, DType::F32)?.to(dev);
    let q = make([1, SSM_V_HEADS, SSM_STATE])?;
    let k = make([1, SSM_V_HEADS, SSM_STATE])?;
    let v = make([1, SSM_V_HEADS, SSM_STATE])?;
    let g = Tensor::rand([1i64, SSM_V_HEADS], DType::F32)?.to(dev)?;
    let b = Tensor::rand([1i64, SSM_V_HEADS], DType::F32)?.to(dev)?;
    let state = make([SSM_V_HEADS, SSM_STATE, SSM_STATE])?;

    let tape = Tape::new([&q, &k, &v, &g, &b, &state])?;
    let q = q.squeeze([0]);
    let k = k.squeeze([0]);
    let v = v.squeeze([0]);
    let g = g.squeeze([0]).exp();
    let b = b.squeeze([0]);

    println!("qwen linear_attention delta_step q=[{SSM_V_HEADS}x{SSM_STATE}] state=[{SSM_V_HEADS}x{SSM_STATE}x{SSM_STATE}]");
    let kv = (state.clone() * k.clone().unsqueeze(2)?).sum([1])?;
    let delta = (v - g.clone().unsqueeze(1)? * kv) * b.unsqueeze(1)?;
    let state = state * g.unsqueeze(1)?.unsqueeze(1)? + k.unsqueeze(2)? * delta.unsqueeze(1)?;
    let core = (state * q.unsqueeze(2)?).sum([1])?;
    let core = core.unsqueeze(0)?;
    tape.realize([&core])?;
    Ok(())
}

/// A padded prefill chunk run: `delta_chunked` at the prefill width.
fn chunked() -> Result<(), ZyxError> {
    let dev = bench_dev();
    let make = |shape: [i64; 3]| Tensor::rand(shape, DType::F32)?.to(dev);
    let q = make([SEQ, SSM_V_HEADS, SSM_STATE])?;
    let k = make([SEQ, SSM_V_HEADS, SSM_STATE])?;
    let v = make([SEQ, SSM_V_HEADS, SSM_STATE])?;
    let g = Tensor::rand([SEQ, SSM_V_HEADS], DType::F32)?.to(dev)?;
    let b = Tensor::rand([SEQ, SSM_V_HEADS], DType::F32)?.to(dev)?;
    let state = make([SSM_V_HEADS, SSM_STATE, SSM_STATE])?;

    let tape = Tape::new([&q, &k, &v, &g, &b, &state])?;
    let (core, state) = delta_chunked(&q, &k, &v, &g, &b, &state)?;
    println!("qwen linear_attention delta_chunked seq={SEQ} chunk={CHUNK} state=[{SSM_V_HEADS}x{SSM_STATE}x{SSM_STATE}]");
    // Both outputs are realized: the port writes the new state back, so the
    // state store is part of what this entry times.
    tape.realize([&core, &state])?;
    Ok(())
}

/// Llama.cpp's chunked gated delta rule (prefill).
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
