// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! End-to-end llama-pipeline confidence tests: a worst-case kernel arg-binding
//! stress (interleaved GlobalMut params, multiple stores), a mini llama forward
//! pass (rms -> qkv proj -> rope -> causal attention -> out proj -> rms) checked
//! against an in-test f32 reference at head_dim 128, and a frozen-tape replay
//! loop with a growing kv-cache (decode pattern: advancing variable offset).

use std::result::Result;
use zyx::{DType, Tape, Tensor, ZyxError};

fn to_f32(t: &Tensor) -> Result<Vec<f32>, ZyxError> {
    let v: Vec<f32> = t.clone().cast(DType::F32).try_into()?;
    Ok(v)
}

fn assert_close(got: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: length mismatch");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() <= tol + 0.01 * e.abs(), "{what}[{i}]: got {g}, expected {e} (tol {tol})");
    }
}

// ============================================================================
// Binding stress: multiple narrow+assign stores (interleaved GlobalMut params)
// plus cross-cache reads in one tape. Exercises the worst case of the
// kernel arg-binding law: several GlobalMut params, several variables, and
// merged kernels whose pre-linearize head order does NOT have muts last.
// ============================================================================

#[test]
fn binding_stress_interleaved_muts() -> Result<(), ZyxError> {
    let rows = 8i64;
    let cols = 4i64;
    let c1 = Tensor::zeros([rows, cols], DType::F32).contiguous()?;
    let c2 = Tensor::zeros([rows, cols], DType::F32).contiguous()?;
    let k1 = Tensor::from(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape([2, cols])?;
    let k2 = Tensor::from(vec![-1.0f32, -2.0, -3.0, -4.0, 0.5, 0.25, 0.125, 0.0625]).reshape([2, cols])?;
    let k3 = Tensor::from(vec![10.0f32, 20.0, 30.0, 40.0, -10.0, -20.0, -30.0, -40.0]).reshape([2, cols])?;

    let tape = Tape::empty();
    tape.add(&c1)?;
    tape.add(&c2)?;
    // Three assigns over two caches with variable offsets: three GlobalMut
    // stores whose params land interleaved with variable loads in the merged
    // kernels.
    let s1 = Tensor::variable(1i64);
    let s2 = Tensor::variable(5i64);
    let s3 = Tensor::variable(6i64);
    c1.narrow(0, &s1, 2i64)?.assign(&k1)?;
    c2.narrow(0, &s2, 2i64)?.assign(&k2)?;
    c1.narrow(0, &s3, 2i64)?.assign(&k3)?;
    // Cross-cache reads in the same tape: the final read kernel loads from
    // BOTH caches, forcing loads and store targets to coexist.
    let r1 = c1.sum([1])?;
    let r2 = c2.sum([1])?;
    let out = r1 + r2;
    tape.realize([&out, &c1, &c2])?;

    // Expected row sums. c1: rows 1,2 = k1 rows; rows 6,7 = k3 rows.
    // c2: rows 5,6 = k2 rows. Everything else untouched zeros.
    let k1r = [[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
    let k2r = [[-1.0f32, -2.0, -3.0, -4.0], [0.5, 0.25, 0.125, 0.0625]];
    let k3r = [[10.0f32, 20.0, 30.0, 40.0], [-10.0, -20.0, -30.0, -40.0]];
    let mut expected = vec![0.0f32; 8];
    for (r, row) in k1r.iter().enumerate() {
        expected[1 + r] += row.iter().sum::<f32>();
    }
    for (r, row) in k3r.iter().enumerate() {
        expected[6 + r] += row.iter().sum::<f32>();
    }
    for (r, row) in k2r.iter().enumerate() {
        expected[5 + r] += row.iter().sum::<f32>();
    }
    assert_close(&to_f32(&out)?, &expected, 1e-6, "cross-cache row sums");

    // Exact per-element placement in each cache.
    let mut e1 = vec![0.0f32; 32];
    for (r, row) in k1r.iter().enumerate() {
        e1[((1 + r) * 4) as usize..((1 + r) * 4 + 4) as usize].copy_from_slice(row);
    }
    for (r, row) in k3r.iter().enumerate() {
        e1[((6 + r) * 4) as usize..((6 + r) * 4 + 4) as usize].copy_from_slice(row);
    }
    assert_close(&to_f32(&c1)?, &e1, 1e-6, "c1 placement");

    let mut e2 = vec![0.0f32; 32];
    for (r, row) in k2r.iter().enumerate() {
        e2[((5 + r) * 4) as usize..((5 + r) * 4 + 4) as usize].copy_from_slice(row);
    }
    assert_close(&to_f32(&c2)?, &e2, 1e-6, "c2 placement");
    Ok(())
}

// ============================================================================
// Mini llama forward at real head_dim (128): rms -> qkv proj -> rope ->
// causal attention -> head concat -> out proj -> rms, symbolic seq offset.
// Checked against an exact in-test f32 reference.
// ============================================================================

#[test]
fn llama_forward_mini_reference() -> Result<(), ZyxError> {
    const SEQ: usize = 4;
    const HEADS: usize = 2;
    const HD: usize = 128;
    const D: usize = HEADS * HD;
    const HALF: usize = HD / 2;
    let eps = 1e-6f32;

    // Deterministic, varied inputs and weights.
    let x: Vec<f32> = (0..SEQ * D).map(|i| ((i * 7) % 19) as f32 / 19.0 - 0.5).collect();
    let wq: Vec<f32> = (0..D * D).map(|i| ((i * 13) % 11) as f32 / 11.0 - 0.5).collect();
    let wk: Vec<f32> = (0..D * D).map(|i| ((i * 5) % 17) as f32 / 17.0 - 0.5).collect();
    let wv: Vec<f32> = (0..D * D).map(|i| ((i * 3) % 7) as f32 / 7.0 - 0.5).collect();
    let wo: Vec<f32> = (0..D * D).map(|i| ((i * 11) % 13) as f32 / 13.0 - 0.5).collect();

    let xt = Tensor::from(x.clone()).reshape([SEQ as i64, D as i64])?;
    let wqt = Tensor::from(wq.clone()).reshape([D as i64, D as i64])?;
    let wkt = Tensor::from(wk.clone()).reshape([D as i64, D as i64])?;
    let wvt = Tensor::from(wv.clone()).reshape([D as i64, D as i64])?;
    let wot = Tensor::from(wo.clone()).reshape([D as i64, D as i64])?;
    let scale_t = Tensor::from(1.0f32 / (HD as f32).sqrt());

    // RoPE frequency tables, llama's precompute: t @ inv_freq, [seq, hd/2].
    let inv_freq: Vec<f32> = (0..HALF).map(|j| 10_000f32.powf(-2.0 * j as f32 / HD as f32)).collect();
    let freqs = Tensor::from(inv_freq).reshape([1, HALF as i64])?;
    let t = Tensor::arange(0u32, SEQ as u32, 1)?.cast(DType::F32).reshape([SEQ as i64, 1])?;
    let ft = t.matmul(&freqs)?;
    let cos_t = ft.cos();
    let sin_t = ft.sin();

    let tape = Tape::empty();
    tape.add(&xt)?;
    tape.add(&wqt)?;
    tape.add(&wkt)?;
    tape.add(&wvt)?;
    tape.add(&wot)?;

    // Forward, mirroring llama's op mix (rms, reshape with dim tensors,
    // transpose, narrow by a variable offset, rope, matmul, softmax).
    let [s_len, _d] = xt.rdims::<2>()?;
    let offset = Tensor::variable(0i64);
    let c = cos_t.narrow(0, offset.clone(), &s_len)?;
    let s = sin_t.narrow(0, offset, &s_len)?;

    let rms = |v: &Tensor| -> Result<Tensor, ZyxError> {
        let xx = v.clone() * v.clone();
        let mean = xx.mean_keepdim([-1])?;
        Ok(v.clone() * (mean + eps).rsqrt())
    };

    let h = rms(&xt)?;
    let q = {
        let q = h
            .clone()
            .matmul(&wqt)?
            .reshape([1i64.into(), s_len.clone(), (HEADS as i64).into(), (HD as i64).into()])?
            .transpose(1, 2)?;
        q.rope(c.clone(), s.clone())?
    };
    let k = {
        let k = h
            .clone()
            .matmul(&wkt)?
            .reshape([1i64.into(), s_len.clone(), (HEADS as i64).into(), (HD as i64).into()])?
            .transpose(1, 2)?;
        k.rope(c.clone(), s.clone())?
    };
    let v = h.matmul(&wvt)?.reshape([1i64.into(), s_len.clone(), (HEADS as i64).into(), (HD as i64).into()])?.transpose(1, 2)?;

    // Causal mask, host-built with real -inf data (llama's get_mask).
    let mut mask = vec![0.0f32; SEQ * SEQ];
    for i in 0..SEQ {
        for j in 0..SEQ {
            if j > i {
                mask[i * SEQ + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask_t = Tensor::from(mask).reshape([1i64, 1i64, SEQ as i64, SEQ as i64])?;

    let attn = (q.matmul(&k.transpose(2, 3)?)? * scale_t.clone() + mask_t).softmax([-1])?;
    let ctx = attn.matmul(&v)?.transpose(1, 2)?.reshape([SEQ as i64, D as i64])?;
    let o = ctx.matmul(&wot)?;
    let out = rms(&o)?;
    tape.realize([&out])?;

    // ---- Exact f32 reference ----
    let matmul = |a: &[f32], b: &[f32], m: usize, kk: usize, n: usize| -> Vec<f32> {
        (0..m).flat_map(|i| (0..n).map(move |j| (0..kk).map(|p| a[i * kk + p] * b[p * n + j]).sum::<f32>())).collect()
    };
    let rms_ref = |v: &mut [f32]| {
        for row in v.chunks_mut(D) {
            let ms = row.iter().map(|x| x * x).sum::<f32>() / D as f32 + eps;
            let inv = 1.0 / ms.sqrt();
            row.iter_mut().for_each(|x| *x *= inv);
        }
    };
    let mut h_ref = x.clone();
    rms_ref(&mut h_ref);
    let mut q = matmul(&h_ref, &wq, SEQ, D, D);
    let mut k = matmul(&h_ref, &wk, SEQ, D, D);
    let v = matmul(&h_ref, &wv, SEQ, D, D);
    // RoPE on [seq, heads, hd] view of q/k, NeoX rotate-half per interleaved pair.
    for vals in [&mut q, &mut k] {
        let mut rotated = vals.clone();
        for pos in 0..SEQ {
            for head in 0..HEADS {
                let base = pos * D + head * HD;
                for j in 0..HALF {
                    let cos_v = 10_000f32.powf(-2.0 * j as f32 / HD as f32) * pos as f32;
                    let (sin_v, cos_v) = (cos_v.sin(), cos_v.cos());
                    let (a, b) = (vals[base + j], vals[base + HALF + j]);
                    rotated[base + j] = a * cos_v - b * sin_v;
                    rotated[base + HALF + j] = b * cos_v + a * sin_v;
                }
            }
        }
        *vals = rotated;
    }
    // Attention per head, then concat heads back to [seq, D].
    let mut ctx = vec![0.0f32; SEQ * D];
    for head in 0..HEADS {
        let scale = 1.0 / (HD as f32).sqrt();
        for i in 0..SEQ {
            let mut scores = [0.0f32; SEQ];
            for j in 0..SEQ {
                if j <= i {
                    scores[j] = (0..HD).map(|p| q[i * D + head * HD + p] * k[j * D + head * HD + p]).sum::<f32>() * scale;
                } else {
                    scores[j] = f32::NEG_INFINITY;
                }
            }
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..SEQ {
                let p = exps[j] / sum;
                for dd in 0..HD {
                    ctx[i * D + head * HD + dd] += p * v[j * D + head * HD + dd];
                }
            }
        }
    }
    let mut o_ref = matmul(&ctx, &wo, SEQ, D, D);
    rms_ref(&mut o_ref);

    assert_close(&to_f32(&out)?, &o_ref, 1e-3, "llama forward mini");
    Ok(())
}

// ============================================================================
// Frozen tape replay: decode loop with a growing kv-cache. The plan is
// compiled once; each replay binds fresh variable values (advancing position)
// and must land the assign at the right offset while preserving earlier steps.
// ============================================================================

#[test]
fn frozen_replay_kv_growth() -> Result<(), ZyxError> {
    let rows = 8i64;
    let cols = 2i64;
    let cache = Tensor::zeros([rows, cols], DType::F32).contiguous()?;
    // Two distinct rows, bf16-exact / f32-exact values.
    let block = Tensor::from(vec![1.0f32, 2.0, 3.0, 4.0]).reshape([2, cols])?;

    let tape = Tape::empty();
    tape.add(&cache)?;
    let pos = Tensor::variable(0i64);
    cache.narrow(0, &pos, Tensor::from(2i64))?.assign(&block)?;
    // Full-cache read with a fixed output shape: per-row sums [8].
    let read = cache.sum([1])?;
    let frozen = tape.freeze([&read])?;

    // Leaf order = discovery order: cache, then pos, then block.
    let expected_after = |steps: &[i64]| -> Vec<f32> {
        let mut e = vec![0.0f32; rows as usize];
        for &p in steps {
            e[p as usize] += 1.0 + 2.0;
            e[p as usize + 1] += 3.0 + 4.0;
        }
        e
    };
    let check = |got: &[f32], steps: &[i64], what: &str| {
        let e = expected_after(steps);
        assert_eq!(got.len(), e.len(), "{what}: length mismatch");
        for (i, (g, ex)) in got.iter().zip(e.iter()).enumerate() {
            assert!((g - ex).abs() <= 1e-6, "{what}[{i}]: got {g}, expected {ex}");
        }
    };

    for (step, &p) in [0i64, 2, 4].iter().enumerate() {
        let pos_v = Tensor::variable(p);
        let outs = frozen.replay([&cache, &pos_v, &block])?;
        let got: Vec<f32> = outs[0].clone().try_into()?;
        let steps_done = &[0i64, 2, 4][..=step];
        check(&got, steps_done, &format!("replay step {step} (pos {p})"));
    }
    Ok(())
}

// ============================================================================
// Decode loop: per-step tape with fresh variables (advancing position, GROWING
// cache_len). The kernel IR must stay identical across steps — variables hash
// by ordinal, not value — so steps 2+ must hit the kernel cache and compile
// nothing new. Run with ZYX_DEBUG=8: the STEP markers delimit compilations;
// every step after the first must produce zero IR dumps.
// ============================================================================

#[test]
fn kv_cache_decode_loop_no_recompile() -> Result<(), ZyxError> {
    let rows = 8i64;
    let cols = 2i64;
    let cache = Tensor::zeros([rows, cols], DType::F32).contiguous()?;
    let block = Tensor::from(vec![1.0f32, 2.0, 3.0, 4.0]).reshape([2, cols])?;

    for step in 0..4i64 {
        println!("STEP {step}");
        let tape = Tape::empty();
        tape.add(&cache)?;
        // Fresh variables every step, new VALUES, identical IR — the llama
        // decode pattern (params hash by ordinal, not value).
        let pos = Tensor::variable(step * 2);
        cache.narrow(0, &pos, Tensor::from(2i64))?.assign(&block)?;
        let cache_len = Tensor::variable(step * 2 + 2);
        let read = cache.narrow(0, Tensor::from(0i64), cache_len)?.sum([1])?;
        tape.realize([&read])?;

        let got: Vec<f32> = read.clone().try_into()?;
        let written = (step * 2 + 2) as usize;
        // The read is narrow(0, 0, cache_len) summed over cols: its length IS
        // the growing cache_len — [2], [4], [6], [8] per step.
        assert_eq!(got.len(), written, "step {step}: read length != cache_len");
        for r in 0..got.len() {
            let expected = if r % 2 == 0 { 3.0 } else { 7.0 };
            assert!((got[r] - expected).abs() <= 1e-6, "step {step} row {r}: got {}, expected {expected}", got[r]);
        }
    }
    Ok(())
}
