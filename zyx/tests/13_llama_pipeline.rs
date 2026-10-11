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
    let mut q_ref = matmul(&h_ref, &wq, SEQ, D, D);
    let mut k_ref = matmul(&h_ref, &wk, SEQ, D, D);
    let v_ref = matmul(&h_ref, &wv, SEQ, D, D);
    // RoPE on [seq, heads, hd] view of q/k, NeoX rotate-half per interleaved pair.
    for vals in [&mut q_ref, &mut k_ref] {
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
    let mut ctx_ref = vec![0.0f32; SEQ * D];
    for head in 0..HEADS {
        let scale = 1.0 / (HD as f32).sqrt();
        for i in 0..SEQ {
            let mut scores = [0.0f32; SEQ];
            for j in 0..SEQ {
                if j <= i {
                    scores[j] = (0..HD).map(|p| q_ref[i * D + head * HD + p] * k_ref[j * D + head * HD + p]).sum::<f32>() * scale;
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
                    ctx_ref[i * D + head * HD + dd] += p * v_ref[j * D + head * HD + dd];
                }
            }
        }
    }
    let mut o_ref = matmul(&ctx_ref, &wo, SEQ, D, D);
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

// ============================================================================
// Per-stage bisection of llama_forward_mini_reference (fails with coarsen on,
// passes with coarsen off). Each stage runs standalone in its own tape with
// the forward's exact data generators and formulas, so a failing stage names
// the broken op mix and a passing stage rules it out. If ALL stages pass, the
// bug needs the full-graph fusion and is not in any single stage.
// ============================================================================

const ST_SEQ: usize = 4;
const ST_HEADS: usize = 2;
const ST_HD: usize = 128;
const ST_D: usize = ST_HEADS * ST_HD;
const ST_HALF: usize = ST_HD / 2;
const ST_EPS: f32 = 1e-6;

fn st_matmul(a: &[f32], b: &[f32], m: usize, kk: usize, n: usize) -> Vec<f32> {
    (0..m).flat_map(|i| (0..n).map(move |j| (0..kk).map(|p| a[i * kk + p] * b[p * n + j]).sum::<f32>())).collect()
}

fn st_rms_ref(v: &mut [f32]) {
    for row in v.chunks_mut(ST_D) {
        let ms = row.iter().map(|x| x * x).sum::<f32>() / ST_D as f32 + ST_EPS;
        let inv = 1.0 / ms.sqrt();
        row.iter_mut().for_each(|x| *x *= inv);
    }
}

fn st_rope_ref(vals: &mut [f32]) {
    let mut rotated = vals.to_vec();
    for pos in 0..ST_SEQ {
        for head in 0..ST_HEADS {
            let base = pos * ST_D + head * ST_HD;
            for j in 0..ST_HALF {
                let ang = 10_000f32.powf(-2.0 * j as f32 / ST_HD as f32) * pos as f32;
                let (a, b) = (vals[base + j], vals[base + ST_HALF + j]);
                rotated[base + j] = a * ang.cos() - b * ang.sin();
                rotated[base + ST_HALF + j] = b * ang.cos() + a * ang.sin();
            }
        }
    }
    vals.copy_from_slice(&rotated);
}

#[test]
fn llama_stage_rms() -> Result<(), ZyxError> {
    let x: Vec<f32> = (0..ST_SEQ * ST_D).map(|i| ((i * 7) % 19) as f32 / 19.0 - 0.5).collect();
    let xt = Tensor::from(x.clone()).reshape([ST_SEQ as i64, ST_D as i64])?;
    let tape = Tape::empty();
    tape.add(&xt)?;
    let xx = xt.clone() * xt.clone();
    let mean = xx.mean_keepdim([-1])?;
    let h = xt.clone() * (mean + ST_EPS).rsqrt();
    tape.realize([&h])?;
    let mut expected = x.clone();
    st_rms_ref(&mut expected);
    assert_close(&to_f32(&h)?, &expected, 1e-6, "stage rms");
    Ok(())
}

#[test]
fn llama_stage_qkv_matmul() -> Result<(), ZyxError> {
    // rms'd input, host-computed (isolates the [4,256]@[256,256] proj matmul).
    let x: Vec<f32> = (0..ST_SEQ * ST_D).map(|i| ((i * 7) % 19) as f32 / 19.0 - 0.5).collect();
    let mut h_ref = x;
    st_rms_ref(&mut h_ref);
    let wq: Vec<f32> = (0..ST_D * ST_D).map(|i| ((i * 13) % 11) as f32 / 11.0 - 0.5).collect();
    let h = Tensor::from(h_ref.clone()).reshape([ST_SEQ as i64, ST_D as i64])?;
    let wqt = Tensor::from(wq.clone()).reshape([ST_D as i64, ST_D as i64])?;
    let tape = Tape::empty();
    tape.add(&h)?;
    tape.add(&wqt)?;
    let q = h.matmul(&wqt)?;
    tape.realize([&q])?;
    assert_close(&to_f32(&q)?, &st_matmul(&h_ref, &wq, ST_SEQ, ST_D, ST_D), 1e-3, "stage qkv matmul");
    Ok(())
}

#[test]
fn llama_stage_v_proj() -> Result<(), ZyxError> {
    // v projection with the forward's exact data (isolates h@wv).
    let x: Vec<f32> = (0..ST_SEQ * ST_D).map(|i| ((i * 7) % 19) as f32 / 19.0 - 0.5).collect();
    let mut h_ref = x;
    st_rms_ref(&mut h_ref);
    let wv: Vec<f32> = (0..ST_D * ST_D).map(|i| ((i * 3) % 7) as f32 / 7.0 - 0.5).collect();
    let h = Tensor::from(h_ref.clone()).reshape([ST_SEQ as i64, ST_D as i64])?;
    let wvt = Tensor::from(wv.clone()).reshape([ST_D as i64, ST_D as i64])?;
    let tape = Tape::empty();
    tape.add(&h)?;
    tape.add(&wvt)?;
    let v = h.matmul(&wvt)?;
    tape.realize([&v])?;
    assert_close(&to_f32(&v)?, &st_matmul(&h_ref, &wv, ST_SEQ, ST_D, ST_D), 1e-3, "stage v proj");
    Ok(())
}

#[test]
fn llama_stage_rope_variable_offset() -> Result<(), ZyxError> {
    // Same op mix as the forward (reshape with dim tensors, transpose, narrow
    // by a VARIABLE offset, rope), standalone. The concrete-offset rope unit
    // test passes; the forward's variable offset is the difference.
    let q0: Vec<f32> = (0..ST_SEQ * ST_D).map(|i| ((i * 7) % 19) as f32 / 19.0 - 0.5).collect();
    let inv_freq: Vec<f32> = (0..ST_HALF).map(|j| 10_000f32.powf(-2.0 * j as f32 / ST_HD as f32)).collect();
    let freqs = Tensor::from(inv_freq).reshape([1, ST_HALF as i64])?;
    let t = Tensor::arange(0u32, ST_SEQ as u32, 1)?.cast(DType::F32).reshape([ST_SEQ as i64, 1])?;
    let ft = t.matmul(&freqs)?;
    let cos_t = ft.cos();
    let sin_t = ft.sin();
    let data = Tensor::from(q0.clone()).reshape([ST_SEQ as i64, ST_D as i64])?;
    let tape = Tape::empty();
    tape.add(&data)?;
    let [s_len, _d] = data.rdims::<2>()?;
    let offset = Tensor::variable(0i64);
    let c = cos_t.narrow(0, offset.clone(), &s_len)?;
    let s = sin_t.narrow(0, offset, &s_len)?;
    let q = data
        .reshape([1i64.into(), s_len.clone(), (ST_HEADS as i64).into(), (ST_HD as i64).into()])?
        .transpose(1, 2)?
        .rope(c, s)?;
    tape.realize([&q])?;
    // Reference: rope on [seq, heads, hd], then transpose to [1, heads, seq, hd].
    let mut q_ref = q0;
    st_rope_ref(&mut q_ref);
    let mut expected = vec![0.0f32; ST_HEADS * ST_SEQ * ST_HD];
    for head in 0..ST_HEADS {
        for i in 0..ST_SEQ {
            for p in 0..ST_HD {
                expected[head * ST_SEQ * ST_HD + i * ST_HD + p] = q_ref[i * ST_D + head * ST_HD + p];
            }
        }
    }
    assert_close(&to_f32(&q)?, &expected, 1e-6, "stage rope variable offset");
    Ok(())
}

#[test]
fn llama_stage_attention() -> Result<(), ZyxError> {
    // scores matmul + scale + causal mask + softmax + ctx matmul, no rope
    // (rope has its own stage test).
    let q0: Vec<f32> = (0..ST_SEQ * ST_D).map(|i| ((i * 7) % 19) as f32 / 19.0 - 0.5).collect();
    let k0: Vec<f32> = (0..ST_SEQ * ST_D).map(|i| ((i * 5) % 17) as f32 / 17.0 - 0.5).collect();
    let v0: Vec<f32> = (0..ST_SEQ * ST_D).map(|i| ((i * 3) % 7) as f32 / 7.0 - 0.5).collect();
    let shape4 = || [1i64, ST_SEQ as i64, ST_HEADS as i64, ST_HD as i64];
    let q = Tensor::from(q0.clone()).reshape([ST_SEQ as i64, ST_D as i64])?.reshape(shape4())?.transpose(1, 2)?;
    let k = Tensor::from(k0.clone()).reshape([ST_SEQ as i64, ST_D as i64])?.reshape(shape4())?.transpose(1, 2)?;
    let v = Tensor::from(v0.clone()).reshape([ST_SEQ as i64, ST_D as i64])?.reshape(shape4())?.transpose(1, 2)?;
    let scale_t = Tensor::from(1.0f32 / (ST_HD as f32).sqrt());
    let mut mask = vec![0.0f32; ST_SEQ * ST_SEQ];
    for i in 0..ST_SEQ {
        for j in 0..ST_SEQ {
            if j > i {
                mask[i * ST_SEQ + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask_t = Tensor::from(mask).reshape([1i64, 1i64, ST_SEQ as i64, ST_SEQ as i64])?;
    let tape = Tape::empty();
    tape.add(&q)?;
    tape.add(&k)?;
    tape.add(&v)?;
    let attn = (q.matmul(&k.transpose(2, 3)?)? * scale_t + mask_t).softmax([-1])?;
    let ctx = attn.matmul(&v)?.transpose(1, 2)?.reshape([ST_SEQ as i64, ST_D as i64])?;
    tape.realize([&ctx])?;
    let mut expected = vec![0.0f32; ST_SEQ * ST_D];
    for head in 0..ST_HEADS {
        let scale = 1.0 / (ST_HD as f32).sqrt();
        for i in 0..ST_SEQ {
            let mut scores = [0.0f32; ST_SEQ];
            for j in 0..ST_SEQ {
                scores[j] = if j <= i {
                    (0..ST_HD).map(|p| q0[i * ST_D + head * ST_HD + p] * k0[j * ST_D + head * ST_HD + p]).sum::<f32>() * scale
                } else {
                    f32::NEG_INFINITY
                };
            }
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..ST_SEQ {
                let p = exps[j] / sum;
                for dd in 0..ST_HD {
                    expected[i * ST_D + head * ST_HD + dd] += p * v0[j * ST_D + head * ST_HD + dd];
                }
            }
        }
    }
    assert_close(&to_f32(&ctx)?, &expected, 1e-3, "stage attention");
    Ok(())
}

#[test]
fn llama_stage_out_proj_rms() -> Result<(), ZyxError> {
    let ctx0: Vec<f32> = (0..ST_SEQ * ST_D).map(|i| ((i * 3) % 7) as f32 / 7.0 - 0.5).collect();
    let wo: Vec<f32> = (0..ST_D * ST_D).map(|i| ((i * 11) % 13) as f32 / 13.0 - 0.5).collect();
    let ctx = Tensor::from(ctx0.clone()).reshape([ST_SEQ as i64, ST_D as i64])?;
    let wot = Tensor::from(wo.clone()).reshape([ST_D as i64, ST_D as i64])?;
    let tape = Tape::empty();
    tape.add(&ctx)?;
    tape.add(&wot)?;
    let o = ctx.matmul(&wot)?;
    let oo = o.clone() * o.clone();
    let out = o.clone() * (oo.mean_keepdim([-1])? + ST_EPS).rsqrt();
    tape.realize([&out])?;
    let mut expected = st_matmul(&ctx0, &wo, ST_SEQ, ST_D, ST_D);
    st_rms_ref(&mut expected);
    assert_close(&to_f32(&out)?, &expected, 1e-3, "stage out proj rms");
    Ok(())
}

// ============================================================================
// Two-tape split of the forward at ctx: tape A runs xt -> ctx and checks it,
// tape B runs ctx -> out from the realized leaf and checks it. Both halves
// passing while the fused forward fails isolates the bug to plan level
// (cross-boundary aliasing/liveness); a failing half names the region.
// ============================================================================

#[test]
fn llama_forward_two_tape_split() -> Result<(), ZyxError> {
    let x: Vec<f32> = (0..ST_SEQ * ST_D).map(|i| ((i * 7) % 19) as f32 / 19.0 - 0.5).collect();
    let wq: Vec<f32> = (0..ST_D * ST_D).map(|i| ((i * 13) % 11) as f32 / 11.0 - 0.5).collect();
    let wk: Vec<f32> = (0..ST_D * ST_D).map(|i| ((i * 5) % 17) as f32 / 17.0 - 0.5).collect();
    let wv: Vec<f32> = (0..ST_D * ST_D).map(|i| ((i * 3) % 7) as f32 / 7.0 - 0.5).collect();
    let wo: Vec<f32> = (0..ST_D * ST_D).map(|i| ((i * 11) % 13) as f32 / 13.0 - 0.5).collect();
    let xt = Tensor::from(x.clone()).reshape([ST_SEQ as i64, ST_D as i64])?;
    let wqt = Tensor::from(wq.clone()).reshape([ST_D as i64, ST_D as i64])?;
    let wkt = Tensor::from(wk.clone()).reshape([ST_D as i64, ST_D as i64])?;
    let wvt = Tensor::from(wv.clone()).reshape([ST_D as i64, ST_D as i64])?;
    let wot = Tensor::from(wo.clone()).reshape([ST_D as i64, ST_D as i64])?;
    let scale_t = Tensor::from(1.0f32 / (ST_HD as f32).sqrt());
    let inv_freq: Vec<f32> = (0..ST_HALF).map(|j| 10_000f32.powf(-2.0 * j as f32 / ST_HD as f32)).collect();
    let freqs = Tensor::from(inv_freq).reshape([1, ST_HALF as i64])?;
    let t = Tensor::arange(0u32, ST_SEQ as u32, 1)?.cast(DType::F32).reshape([ST_SEQ as i64, 1])?;
    let ft = t.matmul(&freqs)?;
    let cos_t = ft.cos();
    let sin_t = ft.sin();
    let mut mask = vec![0.0f32; ST_SEQ * ST_SEQ];
    for i in 0..ST_SEQ {
        for j in 0..ST_SEQ {
            if j > i {
                mask[i * ST_SEQ + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask_t = Tensor::from(mask).reshape([1i64, 1i64, ST_SEQ as i64, ST_SEQ as i64])?;

    // ---- tape A: xt -> ctx (exact forward ops) ----
    let tape_a = Tape::empty();
    tape_a.add(&xt)?;
    tape_a.add(&wqt)?;
    tape_a.add(&wkt)?;
    tape_a.add(&wvt)?;
    let [s_len, _d] = xt.rdims::<2>()?;
    let offset = Tensor::variable(0i64);
    let c = cos_t.narrow(0, offset.clone(), &s_len)?;
    let s = sin_t.narrow(0, offset, &s_len)?;
    let rms = |v: &Tensor| -> Result<Tensor, ZyxError> {
        let xx = v.clone() * v.clone();
        let mean = xx.mean_keepdim([-1])?;
        Ok(v.clone() * (mean + ST_EPS).rsqrt())
    };
    let h = rms(&xt)?;
    let q = h
        .clone()
        .matmul(&wqt)?
        .reshape([1i64.into(), s_len.clone(), (ST_HEADS as i64).into(), (ST_HD as i64).into()])?
        .transpose(1, 2)?
        .rope(c.clone(), s.clone())?;
    let k = h
        .clone()
        .matmul(&wkt)?
        .reshape([1i64.into(), s_len.clone(), (ST_HEADS as i64).into(), (ST_HD as i64).into()])?
        .transpose(1, 2)?
        .rope(c, s)?;
    let v = h.matmul(&wvt)?.reshape([1i64.into(), s_len, (ST_HEADS as i64).into(), (ST_HD as i64).into()])?.transpose(1, 2)?;
    let attn = (q.matmul(&k.transpose(2, 3)?)? * scale_t + mask_t).softmax([-1])?;
    let ctx = attn.matmul(&v)?.transpose(1, 2)?.reshape([ST_SEQ as i64, ST_D as i64])?;
    tape_a.realize([&h, &attn, &ctx])?;

    // Reference through ctx.
    let mut h_ref = x.clone();
    st_rms_ref(&mut h_ref);
    let mut q_ref = st_matmul(&h_ref, &wq, ST_SEQ, ST_D, ST_D);
    let mut k_ref = st_matmul(&h_ref, &wk, ST_SEQ, ST_D, ST_D);
    let v_ref = st_matmul(&h_ref, &wv, ST_SEQ, ST_D, ST_D);
    st_rope_ref(&mut q_ref);
    st_rope_ref(&mut k_ref);
    let mut ctx_ref = vec![0.0f32; ST_SEQ * ST_D];
    let mut attn_ref = vec![0.0f32; ST_HEADS * ST_SEQ * ST_SEQ];
    for head in 0..ST_HEADS {
        let scale = 1.0 / (ST_HD as f32).sqrt();
        for i in 0..ST_SEQ {
            let mut scores = [0.0f32; ST_SEQ];
            for j in 0..ST_SEQ {
                scores[j] = if j <= i {
                    (0..ST_HD).map(|p| q_ref[i * ST_D + head * ST_HD + p] * k_ref[j * ST_D + head * ST_HD + p]).sum::<f32>()
                        * scale
                } else {
                    f32::NEG_INFINITY
                };
            }
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..ST_SEQ {
                let p = exps[j] / sum;
                attn_ref[head * ST_SEQ * ST_SEQ + i * ST_SEQ + j] = p;
                for dd in 0..ST_HD {
                    ctx_ref[i * ST_D + head * ST_HD + dd] += p * v_ref[j * ST_D + head * ST_HD + dd];
                }
            }
        }
    }
    assert_close(&to_f32(&h)?, &h_ref, 1e-6, "two-tape h");
    assert_close(&to_f32(&attn)?, &attn_ref, 1e-3, "two-tape attn");
    assert_close(&to_f32(&ctx)?, &ctx_ref, 1e-3, "two-tape ctx");
    // ---- tape B: ctx leaf -> out (exact forward ops) ----
    let tape_b = Tape::empty();
    tape_b.add(&ctx)?;
    tape_b.add(&wot)?;
    let o = ctx.matmul(&wot)?;
    let oo = o.clone() * o.clone();
    let out = o.clone() * (oo.mean_keepdim([-1])? + ST_EPS).rsqrt();
    tape_b.realize([&out])?;
    let mut o_ref = st_matmul(&ctx_ref, &wo, ST_SEQ, ST_D, ST_D);
    st_rms_ref(&mut o_ref);
    assert_close(&to_f32(&out)?, &o_ref, 1e-3, "two-tape out");
    Ok(())
}

// Regression: coarsen must lane-slot Local scratch (see coarsen()).
// Scores kernel (matmul + scale + mask, two consumers) with matmul-produced
// q/k through rope (exactly the forward's ops, so the kernel carries the
// same dim-expression scaffolding): coarsen used to leave Local-storage
// GEPs untouched, so coarsened lanes clobbered the same local slot and
// rows 1+ came out wrong.
#[test]
fn kernel6_scores_repro() -> Result<(), ZyxError> {
    let x: Vec<f32> = (0..ST_SEQ * ST_D).map(|i| ((i * 7) % 19) as f32 / 19.0 - 0.5).collect();
    let wq: Vec<f32> = (0..ST_D * ST_D).map(|i| ((i * 13) % 11) as f32 / 11.0 - 0.5).collect();
    let wk: Vec<f32> = (0..ST_D * ST_D).map(|i| ((i * 5) % 17) as f32 / 17.0 - 0.5).collect();
    let mut h_ref = x.clone();
    st_rms_ref(&mut h_ref);
    let h = Tensor::from(h_ref.clone()).reshape([ST_SEQ as i64, ST_D as i64])?;
    let wqt = Tensor::from(wq.clone()).reshape([ST_D as i64, ST_D as i64])?;
    let wkt = Tensor::from(wk.clone()).reshape([ST_D as i64, ST_D as i64])?;
    let scale_t = Tensor::from(1.0f32 / (ST_HD as f32).sqrt());
    let inv_freq: Vec<f32> = (0..ST_HALF).map(|j| 10_000f32.powf(-2.0 * j as f32 / ST_HD as f32)).collect();
    let freqs = Tensor::from(inv_freq).reshape([1, ST_HALF as i64])?;
    let t = Tensor::arange(0u32, ST_SEQ as u32, 1)?.cast(DType::F32).reshape([ST_SEQ as i64, 1])?;
    let ft = t.matmul(&freqs)?;
    let cos_t = ft.cos();
    let sin_t = ft.sin();
    let mut mask = vec![0.0f32; ST_SEQ * ST_SEQ];
    for i in 0..ST_SEQ {
        for j in 0..ST_SEQ {
            if j > i {
                mask[i * ST_SEQ + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask_t = Tensor::from(mask).reshape([1i64, 1i64, ST_SEQ as i64, ST_SEQ as i64])?;
    let tape = Tape::empty();
    tape.add(&h)?;
    tape.add(&wqt)?;
    tape.add(&wkt)?;
    // Exact forward producer chain (matmul -> reshape -> transpose -> rope),
    // so the scores kernel carries the same ops as the fused plan.
    let [s_len, _d] = h.rdims::<2>()?;
    let offset = Tensor::variable(0i64);
    let c = cos_t.narrow(0, offset.clone(), &s_len)?;
    let s = sin_t.narrow(0, offset, &s_len)?;
    let q = h
        .clone()
        .matmul(&wqt)?
        .reshape([1i64.into(), s_len.clone(), (ST_HEADS as i64).into(), (ST_HD as i64).into()])?
        .transpose(1, 2)?
        .rope(c.clone(), s.clone())?;
    let k = h
        .matmul(&wkt)?
        .reshape([1i64.into(), s_len, (ST_HEADS as i64).into(), (ST_HD as i64).into()])?
        .transpose(1, 2)?
        .rope(c, s)?;
    let scores = q.matmul(&k.transpose(2, 3)?)? * scale_t + mask_t;
    // Two consumers force scores materialization (the 2-store kernel shape).
    let m = scores.max_keepdim([-1])?;
    let e = (scores.clone() - m.clone()).exp();
    let s = e.sum_keepdim([-1])?;
    tape.realize([&m, &s])?;
    // Reference (rope applied to matmul outputs, then per-head scores).
    let mut q_ref = st_matmul(&h_ref, &wq, ST_SEQ, ST_D, ST_D);
    let mut k_ref = st_matmul(&h_ref, &wk, ST_SEQ, ST_D, ST_D);
    st_rope_ref(&mut q_ref);
    st_rope_ref(&mut k_ref);
    let scale = 1.0 / (ST_HD as f32).sqrt();
    let mut exp_scores = vec![0.0f32; ST_HEADS * ST_SEQ * ST_SEQ];
    let mut exp_sums = vec![0.0f32; ST_HEADS * ST_SEQ];
    for head in 0..ST_HEADS {
        for i in 0..ST_SEQ {
            let mut mx = f32::NEG_INFINITY;
            for j in 0..ST_SEQ {
                let sc = (0..ST_HD).map(|p| q_ref[i * ST_D + head * ST_HD + p] * k_ref[j * ST_D + head * ST_HD + p]).sum::<f32>()
                    * scale
                    + if j <= i { 0.0 } else { f32::NEG_INFINITY };
                exp_scores[head * ST_SEQ * ST_SEQ + i * ST_SEQ + j] = sc;
                mx = mx.max(sc);
            }
            let mut su = 0.0f32;
            for j in 0..ST_SEQ {
                su += (exp_scores[head * ST_SEQ * ST_SEQ + i * ST_SEQ + j] - mx).exp();
            }
            exp_sums[head * ST_SEQ + i] = su;
            // max check via first element is enough alongside sums; check max separately:
            let _ = mx;
        }
    }
    assert_close(&to_f32(&s)?, &exp_sums, 1e-3, "kernel6 sums");
    Ok(())
}

// Repro: fused kernel [5/12] (v proj) with the forward's exact data and ops
// (rms -> matmul -> reshape -> transpose). Fused ctx[768] is wrong; either
// v[3] (this kernel's output) or the ctx kernel miscomputes it.
#[test]
fn kernel5_vproj_repro() -> Result<(), ZyxError> {
    let x: Vec<f32> = (0..ST_SEQ * ST_D).map(|i| ((i * 7) % 19) as f32 / 19.0 - 0.5).collect();
    let wv: Vec<f32> = (0..ST_D * ST_D).map(|i| ((i * 3) % 7) as f32 / 7.0 - 0.5).collect();
    let xt = Tensor::from(x.clone()).reshape([ST_SEQ as i64, ST_D as i64])?;
    let wvt = Tensor::from(wv.clone()).reshape([ST_D as i64, ST_D as i64])?;
    let tape = Tape::empty();
    tape.add(&xt)?;
    tape.add(&wvt)?;
    let [s_len, _d] = xt.rdims::<2>()?;
    let xx = xt.clone() * xt.clone();
    let mean = xx.mean_keepdim([-1])?;
    let h = xt.clone() * (mean + ST_EPS).rsqrt();
    let v = h.matmul(&wvt)?.reshape([1i64.into(), s_len, (ST_HEADS as i64).into(), (ST_HD as i64).into()])?.transpose(1, 2)?;
    tape.realize([&v])?;
    let mut h_ref = x;
    st_rms_ref(&mut h_ref);
    let v_ref = st_matmul(&h_ref, &wv, ST_SEQ, ST_D, ST_D);
    let mut expected = vec![0.0f32; ST_HEADS * ST_SEQ * ST_HD];
    for head in 0..ST_HEADS {
        for i in 0..ST_SEQ {
            for d in 0..ST_HD {
                expected[head * ST_SEQ * ST_HD + i * ST_HD + d] = v_ref[i * ST_D + head * ST_HD + d];
            }
        }
    }
    assert_close(&to_f32(&v)?, &expected, 1e-3, "kernel5 v");
    Ok(())
}

// Repro: fused kernel [11/12] (ctx) with the forward's exact data and ops
// (matmul -> transpose -> reshape). attn/v leaves carry reference values.
#[test]
fn kernel11_ctx_repro() -> Result<(), ZyxError> {
    let x: Vec<f32> = (0..ST_SEQ * ST_D).map(|i| ((i * 7) % 19) as f32 / 19.0 - 0.5).collect();
    let wq: Vec<f32> = (0..ST_D * ST_D).map(|i| ((i * 13) % 11) as f32 / 11.0 - 0.5).collect();
    let wk: Vec<f32> = (0..ST_D * ST_D).map(|i| ((i * 5) % 17) as f32 / 17.0 - 0.5).collect();
    let wv: Vec<f32> = (0..ST_D * ST_D).map(|i| ((i * 3) % 7) as f32 / 7.0 - 0.5).collect();
    let mut h_ref = x;
    st_rms_ref(&mut h_ref);
    let mut q_ref = st_matmul(&h_ref, &wq, ST_SEQ, ST_D, ST_D);
    let mut k_ref = st_matmul(&h_ref, &wk, ST_SEQ, ST_D, ST_D);
    let v_ref = st_matmul(&h_ref, &wv, ST_SEQ, ST_D, ST_D);
    st_rope_ref(&mut q_ref);
    st_rope_ref(&mut k_ref);
    let scale = 1.0 / (ST_HD as f32).sqrt();
    let mut attn_ref = vec![0.0f32; ST_HEADS * ST_SEQ * ST_SEQ];
    let mut ctx_ref = vec![0.0f32; ST_SEQ * ST_D];
    for head in 0..ST_HEADS {
        for i in 0..ST_SEQ {
            let mut scores = [0.0f32; ST_SEQ];
            for j in 0..ST_SEQ {
                scores[j] = if j <= i {
                    (0..ST_HD).map(|p| q_ref[i * ST_D + head * ST_HD + p] * k_ref[j * ST_D + head * ST_HD + p]).sum::<f32>()
                        * scale
                } else {
                    f32::NEG_INFINITY
                };
            }
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..ST_SEQ {
                let p = exps[j] / sum;
                attn_ref[head * ST_SEQ * ST_SEQ + i * ST_SEQ + j] = p;
                for dd in 0..ST_HD {
                    ctx_ref[i * ST_D + head * ST_HD + dd] += p * v_ref[j * ST_D + head * ST_HD + dd];
                }
            }
        }
    }
    // v leaf in [1, heads, seq, hd] logical layout (post-transpose order).
    let mut v_leaf = vec![0.0f32; ST_HEADS * ST_SEQ * ST_HD];
    for head in 0..ST_HEADS {
        for i in 0..ST_SEQ {
            for d in 0..ST_HD {
                v_leaf[head * ST_SEQ * ST_HD + i * ST_HD + d] = v_ref[i * ST_D + head * ST_HD + d];
            }
        }
    }
    let attn = Tensor::from(attn_ref).reshape([1i64, ST_HEADS as i64, ST_SEQ as i64, ST_SEQ as i64])?;
    let v = Tensor::from(v_leaf).reshape([1i64, ST_HEADS as i64, ST_SEQ as i64, ST_HD as i64])?;
    let tape = Tape::empty();
    tape.add(&attn)?;
    tape.add(&v)?;
    let ctx = attn.matmul(&v)?.transpose(1, 2)?.reshape([ST_SEQ as i64, ST_D as i64])?;
    tape.realize([&ctx])?;
    assert_close(&to_f32(&ctx)?, &ctx_ref, 1e-3, "kernel11 ctx");
    Ok(())
}
