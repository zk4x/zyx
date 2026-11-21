// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Ternary-Bonsai-2-27B norms + Prism FWHT + SwiGLU band, at the
//! checkpoint's own geometry (`hidden 5120`, `ffn 17408`, Hadamard block
//! 1024, `eps 1e-6`), as `examples/bonsai-27b/src/model.rs` writes it.
//!
//! The band is the elementwise/normalization work around the projections:
//!
//! - `RMSNorm::forward` (`zyx-nn/src/rms_norm.rs`, `model.rs:600`):
//!   `x * rsqrt(mean(x^2) + eps) * w`, width 5120 (pre-attention / pre-FFN),
//!   width 256 per head for the full-attention q/k norms;
//! - `Prism::forward` (`prism.rs:151-183`): `H (s * x)` blockwise over the
//!   last 1024 columns, the fold every folded projection applies to its
//!   activation;
//! - the SwiGLU activation (`model.rs:607`): `silu(gate) * up` over 17408.
//!
//! The rotation is currently a dense `[rows, 1024] x [1024, 1024]` f32
//! matmul against the normalized Sylvester-Hadamard matrix. The normalized
//! Hadamard product is a 10-stage butterfly on the same 1024 elements, which
//! is what llama.cpp's `fwht.cu` runs (`fwht_cuda_block<1024, 256>`), so the
//! test writes both forms as custom kernel IR and compares them against the
//! same host reference `Prism::forward` defines.

use zyx::kernel::autotune::BeamSearch;
use zyx::kernel::{Dev, Kernel, MemScope, OpId};
use zyx::{DType, Tape, Tensor, ZyxError, f16};

/// Residual stream width.
const HIDDEN: i64 = 5_120;
/// SwiGLU inner width.
const FFN: i64 = 17_408;
/// Hadamard block width (`prism.hadamard.block_size`).
const BLOCK: i64 = 1_024;
/// RMSNorm epsilon.
const EPS: f32 = 1e-6;
/// Prefill sequence length.
const SEQ: i64 = 128;

/// Deterministic values in `[-1, 1)`.
fn seq_vals(n: i64, seed: i64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = (i * 17 + seed * 29) % 23;
            (v as f32 - 11.0) / 11.0
        })
        .collect()
}

/// The checkpoint's folded-weight sign vector, at an arbitrary but fixed
/// pattern (the kernel reads it as a `Tape` input, so only its absolute
/// values matter).
fn signs_for(width: i64) -> Vec<f32> {
    (0..width).map(|i| if i % 3 == 1 { -1.0f32 } else { 1.0 }).collect()
}

/// Normalized Sylvester-Walsh-Hadamard matrix, the same construction as
/// `prism.rs:187-200`.
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

/// The first non-host device, or `None` when the box has no GPU.
fn gpu() -> Option<Dev> {
    Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host))
}

/// Device-measured launch time, in microseconds: the best of `n` one-launch
/// measurements on the search's own timing path (`launch_timed`), so a cold
/// GPU that is still boosting does not decide the comparison. Each run is a
/// warmup launch plus a timed launch.
fn timed(kernel: Kernel, args: &[&Tensor], n: usize) -> Result<f64, ZyxError> {
    let opts: &[zyx::kernel::autotune::MakeOpt] = &[];
    let mut best = u64::MAX;
    for _ in 0..n {
        let (_winner, nanos) = BeamSearch::new().run([kernel.clone()], args, opts, |_| {}, |_| 0)?;
        best = best.min(nanos);
    }
    Ok(best as f64 * 1e-3)
}

/// One search over `default_optimizations`, as the runtime drives it: the same
/// epilogue and the same learned cost the plan uses, returning the winner so
/// its IR can be inspected beside the seed's.
fn searched_winner(kernel: Kernel, args: &[&Tensor], budget: Option<(usize, usize)>) -> Result<(Kernel, f64), ZyxError> {
    let mut search = BeamSearch::new();
    if let Some((opts, seeds)) = budget {
        search.n_total_opts = opts;
        search.n_seeds = seeds;
    }
    let (winner, nanos) =
        search.run([kernel], args, &Kernel::default_optimizations(), Kernel::default_epilogue, Kernel::base_cost)?;
    Ok((winner, nanos as f64 * 1e-3))
}

fn searched(kernel: Kernel, args: &[&Tensor], budget: Option<(usize, usize)>) -> Result<f64, ZyxError> {
    Ok(searched_winner(kernel, args, budget)?.1)
}

/// The tensor-op butterfly `prism.rs::rotate` spells, verbatim: block reshape,
/// the two halves of bit `h` via `narrow`, add/sub, one `1/sqrt(block)` scale.
fn butterfly(x: &Tensor, block: i64) -> Result<Tensor, ZyxError> {
    let shape = x.shape();
    let rows = x.numel() / block;
    let block_t = Tensor::from(block);
    let mut t = x.reshape([rows.clone(), block_t.clone()])?;
    let mut h = 1i64;
    while h < block {
        let view = t.reshape([
            rows.clone(),
            Tensor::from(block / (2 * h)),
            Tensor::from(2i64),
            Tensor::from(h),
        ])?;
        let low = view.narrow(-2, 0i64, 1i64)?;
        let high = view.narrow(-2, 1i64, 1i64)?;
        let sum = &low + &high;
        let diff = &low - &high;
        t = Tensor::cat([&sum, &diff], -2)?.reshape([rows.clone(), block_t.clone()])?;
        h *= 2;
    }
    (t * Tensor::from(1.0f32 / (block as f32).sqrt())).reshape(shape)
}

/// The `prism.rs::rotate` form with tinygrad's `.contiguous()` closing each
/// stage: a `STAGE`/store boundary between stages, so the kernelizer emits one
/// kernel per stage instead of inlining the ten stage views into one tree.
fn butterfly_contiguous(x: &Tensor, block: i64) -> Result<Tensor, ZyxError> {
    let shape = x.shape();
    let rows = x.numel() / block;
    let block_t = Tensor::from(block);
    let mut t = x.reshape([rows.clone(), block_t.clone()])?;
    let mut h = 1i64;
    while h < block {
        let view = t.reshape([
            rows.clone(),
            Tensor::from(block / (2 * h)),
            Tensor::from(2i64),
            Tensor::from(h),
        ])?;
        let low = view.narrow(-2, 0i64, 1i64)?;
        let high = view.narrow(-2, 1i64, 1i64)?;
        let sum = &low + &high;
        let diff = &low - &high;
        t = Tensor::cat([&sum, &diff], -2)?.reshape([rows.clone(), block_t.clone()])?.contiguous()?;
        h *= 2;
    }
    (t * Tensor::from(1.0f32 / (block as f32).sqrt())).reshape(shape)
}

/// Wall-clock time of one frozen-tape replay of `f`, in microseconds: the
/// expression is traced once, frozen (so the plan is compiled once and every
/// later replay is the plan's own launches), then replayed `reps` times with a
/// read of the output at the end to drain the queue.
fn frozen_us(f: impl Fn(&Tensor) -> Result<Tensor, ZyxError>, x: &Tensor, reps: usize) -> Result<f64, ZyxError> {
    let tape = Tape::new([x])?;
    let out = f(x)?;
    let frozen = tape.freeze([x], [&out])?;
    let _ = frozen.replay([x])?;
    let started = std::time::Instant::now();
    let mut last = Vec::new();
    for _ in 0..reps {
        last = frozen.replay([x])?;
    }
    let _: Vec<f32> = last.into_iter().next().expect("one output").to_vec()?;
    Ok(started.elapsed().as_secs_f64() * 1e6 / reps as f64)
}

/// The naive rotation kernel the graph lowers `matmul(&rot)` + sign flip +
/// norm scale to (`ZYX_DEBUG=264` dump of `band_tape`): one thread per output
/// element, the whole 1024-long dot in one register accumulator.
fn naive_rotation_kernel(dev: Dev, block: i64, nblk: i64, blocks: i64) -> Kernel {
    let mut k = Kernel::new(dev);
    let x = k.param(DType::F32);
    let inv = k.param(DType::F32);
    let norm = k.param(DType::F32);
    let signs = k.param(DType::F32);
    let rot = k.param(DType::F32);
    let out = k.param_mut(DType::F32);
    // One thread per output element of the `blocks` 1024-wide rows.
    let g = k.group_range(0, blocks * block);
    let shift = k.const_idx(block.trailing_zeros() as i64);
    let mask = k.const_idx(block - 1);
    let bidx = k.bit_shift_right(g, shift);
    let j = k.bit_and(g, mask);
    let inv_idx = k.div(bidx, nblk);
    let inv_v = k.load(inv, inv_idx);
    let block_in_row = k.mod_(bidx, nblk);
    let sn_base = k.mul(block_in_row, block);
    let acc = k.storage(DType::F32, MemScope::Register, 1);
    let zero = k.const_val(0.0f32);
    k.store(acc, zero, 0i64);
    k.loop_over(block, |k, c| {
        let bidx_off = k.mul(bidx, block);
        let idx = k.add(bidx_off, c);
        let xv = k.load(x, idx);
        let sn_idx = k.add(sn_base, c);
        let nv = k.load(norm, sn_idx);
        let sv = k.load(signs, sn_idx);
        let c_off = k.mul(c, block);
        let ridx = k.add(j, c_off);
        let rv = k.load(rot, ridx);
        let a = k.mul(xv, inv_v);
        let b = k.mul(a, nv);
        let c2 = k.mul(b, sv);
        let prod = k.mul(c2, rv);
        let prev = k.load(acc, 0i64);
        let sum = k.add(prev, prod);
        k.store(acc, sum, 0i64);
    });
    let res = k.load(acc, 0i64);
    k.store(out, res, g);
    k
}

/// The graph's `mean(x^2)` reduce kernel: one group per activation row, one
/// thread per group, one accumulator.
fn rms_reduce_kernel(dev: Dev, n: i64, rows: i64) -> Kernel {
    let mut k = Kernel::new(dev);
    let x = k.param(DType::F32);
    let out = k.param_mut(DType::F32);
    let r = k.group_range(0, rows);
    let base = k.mul(r, n);
    let acc = k.storage(DType::F32, MemScope::Register, 1);
    let zero = k.const_val(0.0f32);
    k.store(acc, zero, 0i64);
    k.loop_over(n, |k, i| {
        let idx = k.add(base, i);
        let v = k.load(x, idx);
        let sq = k.mul(v, v);
        let prev = k.load(acc, 0i64);
        let sum = k.add(prev, sq);
        k.store(acc, sum, 0i64);
    });
    let total = k.load(acc, 0i64);
    let nf = k.const_val(n as f32);
    let mean = k.div(total, nf);
    let eps = k.const_val(EPS);
    let shifted = k.add(mean, eps);
    let inv = k.rsqrt(shifted);
    k.store(out, inv, r);
    k
}

/// The graph's SwiGLU kernel: one thread per element, the SiLU written as
/// `exp2(-1.442695 x)` (`ZYX_DEBUG=264` dump of `band_tape`).
fn swiglu_kernel(dev: Dev, n: i64) -> Kernel {
    let mut k = Kernel::new(dev);
    let gate = k.param(DType::F32);
    let up = k.param(DType::F32);
    let out = k.param_mut(DType::F32);
    let g = k.group_range(0, n);
    let gv = k.load(gate, g);
    let uv = k.load(up, g);
    let lg = k.const_val(1.442_695f32);
    let neg = k.neg(gv);
    let scaled = k.mul(neg, lg);
    let e = k.exp2(scaled);
    let one = k.const_val(1.0f32);
    let denom = k.add(e, one);
    let recip = k.div(one, denom);
    let silu = k.mul(recip, gv);
    let res = k.mul(silu, uv);
    k.store(out, res, g);
    k
}

/// The band as the graph lowers it today, measured per kernel at the
/// checkpoint's full sizes: the dense Hadamard rotation (`[rows, 1024] x
/// [1024, 1024]`), the RMSNorm reduce and the SwiGLU elementwise.
#[test]
fn band_measure() -> Result<(), ZyxError> {
    let Some(dev) = gpu() else {
        return Ok(());
    };
    let opts: &[zyx::kernel::autotune::MakeOpt] = &[];

    for (label, blocks, nblk) in [
        ("decode", HIDDEN / BLOCK, HIDDEN / BLOCK),
        ("prefill", SEQ * HIDDEN / BLOCK, HIDDEN / BLOCK),
    ] {
        let k = naive_rotation_kernel(dev, BLOCK, nblk, blocks);
        let x = Tensor::from_vec(seq_vals(blocks * BLOCK, 1), [blocks * BLOCK])?.to(dev)?;
        let inv = Tensor::from_vec(vec![1.0f32; usize::try_from(blocks / nblk).unwrap()], [blocks / nblk])?.to(dev)?;
        let norm =
            Tensor::from_vec(signs_for(nblk * BLOCK).iter().map(|s| 0.5 + 0.25 * s).collect::<Vec<f32>>(), [nblk * BLOCK])?
                .to(dev)?;
        let signs = Tensor::from_vec(signs_for(nblk * BLOCK), [nblk * BLOCK])?.to(dev)?;
        let rot = Tensor::from_vec(hadamard(BLOCK), [BLOCK * BLOCK])?.to(dev)?;
        let scratch = Tensor::zeros([blocks * BLOCK], DType::F32).to(dev)?;
        let us = timed(k, &[&x, &inv, &norm, &signs, &rot, &scratch], 3)?;
        println!("band {label}: naive dense rotation [{blocks}x{BLOCK}]: {us:.2} us");
    }

    for (label, seq) in [("decode", 1i64), ("prefill", SEQ)] {
        let n = seq * HIDDEN;
        let x = Tensor::from_vec(seq_vals(n, 2), [n])?.to(dev)?;
        let scratch = Tensor::zeros([seq], DType::F32).to(dev)?;
        let k = rms_reduce_kernel(dev, HIDDEN, seq);
        let us = timed(k, &[&x, &scratch], 3)?;
        println!("band {label}: rms reduce [{HIDDEN}] x {seq} rows: {us:.2} us");
    }

    for (label, seq) in [("decode", 1i64), ("prefill", SEQ)] {
        let n = seq * FFN;
        let gate = Tensor::from_vec(seq_vals(n, 3), [n])?.to(dev)?;
        let up = Tensor::from_vec(seq_vals(n, 4), [n])?.to(dev)?;
        let scratch = Tensor::zeros([n], DType::F32).to(dev)?;
        let k = swiglu_kernel(dev, n);
        let us = timed(k, &[&gate, &up, &scratch], 3)?;
        println!("band {label}: silu(gate)*up [{n}]: {us:.2} us");
    }
    let _ = opts;
    Ok(())
}

/// The band as the model writes it, traced on a `Tape` so the kernelizer and
/// the passes see the model's own graph. Run with `ZYX_DEBUG=264` for the
/// no-search naive IR of every kernel in the band.
#[test]
fn band_tape() -> Result<(), ZyxError> {
    let (x_v, norm_v, sign_v, rot_v) =
        (seq_vals(HIDDEN, 8), vec![0.02f32; usize::try_from(HIDDEN).unwrap()], signs_for(HIDDEN), hadamard(BLOCK));
    let (gate_v, up_v) = (seq_vals(FFN, 9), seq_vals(FFN, 10));
    let x = Tensor::from_vec(x_v, [1, HIDDEN])?;
    let norm = Tensor::from_vec(norm_v, [HIDDEN])?;
    let signs = Tensor::from_vec(sign_v, [HIDDEN])?;
    let rot = Tensor::from_vec(rot_v, [BLOCK, BLOCK])?;
    let gate = Tensor::from_vec(gate_v, [1, FFN])?;
    let up = Tensor::from_vec(up_v, [1, FFN])?;

    let tape = Tape::new([&rot, &signs])?;
    tape.add(&x)?;
    tape.add(&norm)?;
    tape.add(&gate)?;
    tape.add(&up)?;

    // `RMSNorm::forward`.
    let eps = Tensor::from(EPS).cast(DType::F32);
    let h = &x * ((&x * &x).mean_keepdim([-1])? + eps).rsqrt() * &norm;
    // `Prism::forward`: `H (s * x)` over each 1024-wide block.
    let signed = &h * &signs;
    let rows = HIDDEN / BLOCK;
    let rotated = signed.reshape([rows, BLOCK])?.matmul(&rot)?.reshape([1, HIDDEN])?;
    // `silu(gate) * up`.
    let mid = gate.swish() * &up;
    tape.realize([&rotated, &mid])?;
    Ok(())
}

/// Host reference for the fused band kernel: `H (s * (x * inv * norm))` per
/// 1024-wide block, in f32 — the exact arithmetic `Prism::forward` and the
/// graph's fused kernel perform.
#[allow(clippy::too_many_arguments)]
fn fused_ref(x: &[f32], inv: &[f32], norm: &[f32], signs: &[f32], rot: &[f32], rows: usize, nblk: usize) -> Vec<f32> {
    let block = usize::try_from(BLOCK).expect("block fits usize");
    let width = nblk * block;
    let mut out = vec![0f32; rows * width];
    for m in 0..rows {
        for b in 0..nblk {
            for j in 0..block {
                let mut acc = 0f32;
                for c in 0..block {
                    let src = m * width + b * block + c;
                    let sn = b * block + c;
                    acc += x[src] * inv[m] * norm[sn] * signs[sn] * rot[j * block + c];
                }
                out[m * width + b * block + j] = acc;
            }
        }
    }
    out
}

/// Host reference for the pure rotation `H x` (no sign/norm/inv) at any block
/// width, in f64: the Sylvester-Walsh product the dense form's matrix carries,
/// normalized by `1/sqrt(block)` like the matrix is.
fn rotate_ref(x: &[f32], block: i64, rows: i64) -> Vec<f32> {
    let n = usize::try_from(block).expect("block fits usize");
    let r = usize::try_from(rows).expect("rows fit usize");
    let scale = 1.0f64 / (n as f64).sqrt();
    let mut out = vec![0f32; r * n];
    for m in 0..r {
        for j in 0..n {
            let mut acc = 0f64;
            for c in 0..n {
                let sign = if (j & c).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
                acc += sign * f64::from(x[m * n + c]);
            }
            out[m * n + j] = (acc * scale) as f32;
        }
    }
    out
}

/// The Prism rotation at the checkpoint's real widths: the naive dense-dot
/// kernel the graph lowers today against the FWHT block kernel, both checked
/// against the host reference `Prism::forward` defines, measured on the
/// search's one-launch path.
#[test]
fn prism_rotation_fwht() -> Result<(), ZyxError> {
    let Some(dev) = gpu() else {
        return Ok(());
    };

    // The decode shapes get the search's full default budget (they are what the
    // step pays for); the prefill seeds are 90 ms launches, so their search is
    // capped to keep the run inside the test budget and reported as a seed.
    for (label, rows, nblk, full_search) in [
        ("decode gate/up", 1i64, HIDDEN / BLOCK, true),
        ("decode down", 1i64, FFN / BLOCK, true),
        ("prefill gate/up", SEQ, HIDDEN / BLOCK, false),
        ("prefill down", SEQ, FFN / BLOCK, false),
    ] {
        let blocks = rows * nblk;
        let width = nblk * BLOCK;
        let x_v = seq_vals(blocks * BLOCK, 7);
        let inv_v: Vec<f32> = (0..rows).map(|m| 0.5 + 0.01 * m as f32).collect();
        let norm_v: Vec<f32> = (0..width).map(|i| 0.5 + 0.25 * (i % 5) as f32).collect();
        let sign_v = signs_for(width);
        let rot_v = hadamard(BLOCK);
        let want = fused_ref(
            &x_v,
            &inv_v,
            &norm_v,
            &sign_v,
            &rot_v,
            usize::try_from(rows).expect("rows fit usize"),
            usize::try_from(nblk).expect("nblk fit usize"),
        );

        let x_t = Tensor::from_vec(x_v, [blocks * BLOCK])?.to(dev)?;
        let inv_t = Tensor::from_vec(inv_v, [rows])?.to(dev)?;
        let norm_t = Tensor::from_vec(norm_v, [width])?.to(dev)?;
        let sign_t = Tensor::from_vec(sign_v, [width])?.to(dev)?;
        let rot_t = Tensor::from_vec(rot_v, [BLOCK * BLOCK])?.to(dev)?;

        // ---- the naive dense-dot kernel ----
        let naive = naive_rotation_kernel(dev, BLOCK, nblk, blocks);
        naive.clone().verify();
        let naive_out = Tensor::zeros([blocks * BLOCK], DType::F32).to(dev)?;
        // Prefill launches are 90+ ms; one measurement is enough for the seed.
        let reps = if rows > 1 { 1 } else { 3 };
        let naive_us = timed(naive.clone(), &[&x_t, &inv_t, &norm_t, &sign_t, &rot_t, &naive_out], reps)?;
        let got_naive: Vec<f32> = naive
            .clone()
            .compile()?
            .forward(&[&x_t, &inv_t, &norm_t, &sign_t, &rot_t], vec![[blocks * BLOCK]])?
            .into_iter()
            .next()
            .expect("one output")
            .try_into()?;

        // ---- the FWHT block kernel ----
        let fast = fwht_kernel(dev, BLOCK, 256, nblk, blocks);
        fast.clone().verify();
        let fast_out = Tensor::zeros([blocks * BLOCK], DType::F32).to(dev)?;
        let fast_us = timed(fast.clone(), &[&x_t, &inv_t, &norm_t, &sign_t, &fast_out], 3)?;
        let got_fast: Vec<f32> = fast
            .clone()
            .compile()?
            .forward(&[&x_t, &inv_t, &norm_t, &sign_t], vec![[blocks * BLOCK]])?
            .into_iter()
            .next()
            .expect("one output")
            .try_into()?;

        // ---- the graph's current form: the same seed through the search ----
        // What the model runs today: `default_optimizations` over the naive
        // dense seed, measured on the winner.
        let budget = if full_search { None } else { Some((8, 8)) };
        let searched_us = searched(naive.clone(), &[&x_t, &inv_t, &norm_t, &sign_t, &rot_t, &naive_out], budget)?;

        let mut worst_naive = (0f32, 0usize);
        let mut worst_fast = (0f32, 0usize);
        for (i, (g, w)) in got_naive.iter().zip(want.iter()).enumerate() {
            let e = (g - w).abs();
            if e > worst_naive.0 {
                worst_naive = (e, i);
            }
        }
        for (i, (g, w)) in got_fast.iter().zip(want.iter()).enumerate() {
            let e = (g - w).abs();
            if e > worst_fast.0 {
                worst_fast = (e, i);
            }
        }
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
        // The F16 the folded projection consumes: the model's own cast, applied
        // to both forms. A butterfly reorders the sum, so a value sitting on an
        // F16 tie may round the other way; count how often.
        let mut f16_diff = 0usize;
        let mut f16_ulp = 0f32;
        for (g, n) in got_fast.iter().zip(got_naive.iter()) {
            let (gf, nf) = (round16(*g), round16(*n));
            if gf != nf {
                f16_diff += 1;
                // One F16 ulp at this magnitude: 2^-10 of the leading bit,
                // floored at a thousandth of the row's own scale (below that a
                // value is negligible in the projection and F16 is dense).
                let mag = nf.abs().max(scale * 1e-3);
                let ulp = 2f32.powf(mag.log2().floor() - 10.0);
                f16_ulp = f16_ulp.max((gf - nf).abs() / ulp.max(1e-12));
            }
        }
        println!(
            "prism rotation {label}: naive {naive_us:.2} us (err {:.2e}), searched {searched_us:.2} us, fwht {fast_us:.2} us (err {:.2e}), speedup {:.1}x vs naive / {:.1}x vs searched, f16 diff {f16_diff}/{} (<= {f16_ulp:.0} ulp), |out|max {scale:.2}",
            worst_naive.0,
            worst_fast.0,
            naive_us / fast_us,
            searched_us / fast_us,
            got_fast.len()
        );
        assert!(
            f16_ulp <= 2.0,
            "{label}: the F16 the projection reads must not move by more than a tie-break ulp, got {f16_ulp}"
        );
        // The FWHT is the same Hadamard product in butterfly order: f32 error is
        // one ulp of the sum, far below the F16 the projection reads.
        let tol = 1e-3 * scale + 1e-4;
        assert!(
            worst_naive.0 <= tol,
            "{label}: the dense kernel must reproduce Prism::forward: {:.3e} at {}",
            worst_naive.0,
            worst_naive.1
        );
        assert!(
            worst_fast.0 <= 1e-5 * scale + 1e-5,
            "{label}: the FWHT kernel must reproduce Prism::forward: {:.3e} at {}",
            worst_fast.0,
            worst_fast.1
        );
    }
    Ok(())
}

/// The reference FWHT block kernel: one block per `block`-wide row, `threads`
/// threads holding `block/threads` elements each, every stage a butterfly in
/// the block's shared memory — llama.cpp's `fwht_cuda_smem` (`fwht.cu`) with
/// the `NE`-register row tiling of `fwht_cuda_block` so the smem traffic is
/// `1/NE` per stage.
///
/// The uniform stage form is the same arithmetic for all `log2(block)` stages:
/// with the row staged as `s[i*threads + tid]`, the partner of element `e` is
/// `e ^ h`, and `(e & h) == 0` selects the low element of the pair, which takes
/// `x + y` while the high one takes `x - y`. For `h < threads` that is a
/// lane exchange, for `h >= threads` it is another of the thread's own
/// registers, which is why one address rule covers both of the reference's
/// three stage loops.
///
/// The pre-scale is folded in exactly as the graph's fused kernel does it:
/// `x * inv_rms * norm * signs`, then the normalized Hadamard's `1/sqrt(N)` —
/// the same scalar the dense form's matrix carries. The output is F32, the
/// dtype the graph's rotation produces (the model's explicit cast to F16
/// before the folded projection is a separate op and stays one).
fn fwht_kernel(dev: Dev, block: i64, threads: i64, nblk: i64, rows: i64) -> Kernel {
    let ne = block / threads;
    assert!(block % threads == 0, "fwht: block {block} is not a multiple of {threads} threads");
    assert!(ne <= 4, "fwht: {ne} registers per thread is not the block form");
    let mut k = Kernel::new(dev);
    let x = k.param(DType::F32);
    let inv = k.param(DType::F32);
    let norm = k.param(DType::F32);
    let signs = k.param(DType::F32);
    let out = k.param_mut(DType::F32);

    let r = k.group_range(0, rows);
    let tid = k.local_range(0, u32::try_from(threads).expect("threads fit u32"));
    // `1/sqrt(N)`, the normalization the Sylvester matrix carries.
    let scale = k.const_val(1.0f32 / (block as f32).sqrt());

    // Element base of this row, and the row's own sign/norm block.
    let base = k.mul(r, block);
    let block_in_row = k.mod_(r, nblk);
    let sn_base = k.mul(block_in_row, block);
    // The RMSNorm inverse is one scalar per activation row.
    let inv_idx = k.div(r, nblk);
    let inv_v = k.load(inv, inv_idx);

    // One register per `NE`th of the row: `reg[i] = x[i*threads + tid]`.
    let mut reg = [OpId::NULL; 4];
    // `i*threads + tid`, the thread's element index inside its `NE`th slice.
    let mut lane_off = [OpId::NULL; 4];
    let mut i = 0;
    while i < ne {
        let slot = usize::try_from(i).expect("ne fits usize");
        let ic = k.const_idx(i);
        let off = k.mul(ic, threads);
        let at = k.add(off, tid);
        lane_off[slot] = at;
        let idx = k.add(base, at);
        let xv = k.load(x, idx);
        let sn_idx = k.add(sn_base, at);
        let nv = k.load(norm, sn_idx);
        let sv = k.load(signs, sn_idx);
        let a = k.mul(xv, inv_v);
        let b = k.mul(a, nv);
        let c = k.mul(b, sv);
        let d = k.mul(c, scale);
        reg[slot] = d;
        i += 1;
    }

    // All `log2(block)` stages, each pair `(e, e ^ h)` with `e & h == 0` low.
    let smem = k.storage(DType::F32, MemScope::Local, block);
    let mut h = 1i64;
    while h < block {
        let mut i = 0;
        while i < ne {
            k.store(smem, reg[usize::try_from(i).expect("ne fits usize")], lane_off[usize::try_from(i).expect("ne fits usize")]);
            i += 1;
        }
        k.barrier();
        let hm = k.const_idx(h);
        let zero = k.const_idx(0i64);
        let mut i = 0;
        while i < ne {
            let slot = usize::try_from(i).expect("ne fits usize");
            let addr = lane_off[slot];
            let partner = k.bit_xor(addr, hm);
            let val2 = k.load(smem, partner);
            let bit = k.bit_and(addr, hm);
            let low = k.equal(bit, zero);
            let val = reg[slot];
            let sum = k.add(val, val2);
            let diff = k.sub(val2, val);
            reg[slot] = k.select(low, sum, diff);
            i += 1;
        }
        k.barrier();
        h *= 2;
    }

    let mut i = 0;
    while i < ne {
        let slot = usize::try_from(i).expect("ne fits usize");
        let idx = k.add(base, lane_off[slot]);
        k.store(out, reg[slot], idx);
        i += 1;
    }
    k
}

/// F16 rounding, because the model casts the rotated activation to F16 before
/// the folded projection (`model.rs:585-589`).
fn round16(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

/// The rotation as `prism.rs::rotate` now writes it, as tensor ops, against the
/// dense `matmul(rotation)` form it replaces: the same Sylvester-Walsh product
/// and the same `1/sqrt(block)` normalization, checked against one f64 host
/// reference, then timed beside the hand-written FWHT kernel.
#[test]
fn prism_rotation_butterfly() -> Result<(), ZyxError> {
    let Some(dev) = gpu() else {
        return Ok(());
    };
    let max_err = |got: &[f32], want: &[f32]| {
        got.iter().zip(want.iter()).enumerate().fold((0f32, 0usize), |(m, mi), (i, (g, w))| {
            let e = (g - w).abs();
            if e > m { (e, i) } else { (m, mi) }
        })
    };
    // The butterfly and the dense dot sum the same `block` terms in different
    // associations, so they agree to f32 rounding, not bit-for-bit; report the
    // distance in f32 ulps of the transform's own scale (`|out|max`).
    let max_ulp = |a: &[f32], b: &[f32], scale: f32| {
        let one_ulp = 2f32.powf(scale.log2().floor() - 23.0);
        a.iter().zip(b.iter()).fold(0f32, |u, (x, y)| u.max((x - y).abs() / one_ulp))
    };

    // ---- the sequence itself, at blocks the kernelizer fuses whole ----
    for (block, rows, nblk) in [(4i64, 2i64, 2i64), (8, 2, 2)] {
        let width = nblk * block;
        let x_v = seq_vals(rows * width, 12);
        let rot_v = hadamard(block);
        let want = rotate_ref(&x_v, block, rows * nblk);

        let x = Tensor::from_vec(x_v, [rows, width])?.to(dev)?;
        let rot = Tensor::from_vec(rot_v, [block, block])?.to(dev)?;
        let tape = Tape::new([&x])?;
        let dense = x.reshape([rows * nblk, block])?.matmul(&rot)?.reshape([rows, width])?;
        let bf = butterfly(&x, block)?;
        let frozen = tape.freeze([&x], [&dense, &bf])?;
        let outs = frozen.replay([&x])?;
        let got_dense: Vec<f32> = outs[0].clone().to_vec()?;
        let got_bf: Vec<f32> = outs[1].clone().to_vec()?;
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
        let (dense_err, _) = max_err(&got_dense, &want);
        let (bf_err, bf_at) = max_err(&got_bf, &want);
        let ulp = max_ulp(&got_bf, &got_dense, scale);
        println!(
            "butterfly rotation block {block} [{rows}x{width}]: dense err {dense_err:.2e}, butterfly err {bf_err:.2e}@{bf_at}, butterfly-vs-dense {ulp:.1} ulp"
        );
        assert!(
            bf_err <= 1e-5 * scale + 1e-5,
            "block {block}: the butterfly must reproduce the Hadamard product: {bf_err:.3e} at {bf_at}"
        );
        assert!(ulp <= 2.0, "block {block}: the butterfly must agree with the dense form to f32 rounding, got {ulp:.1} ulp");
    }

    // ---- the per-token rotation set and llama.cpp's 257 fwht launches/token
    //      (`64 * 4 + 1`): four hidden-wide activations per block and one
    //      ffn-wide (`down`) one. Same convention as the board's ~150 ms/step.
    const BLOCKS_PER_STEP: f64 = 64.0;
    const HIDDEN_WIDE_PER_BLOCK: f64 = 4.0;
    let mut step = [0f64; 3];
    for (label, width) in [("gate/up", HIDDEN), ("down", FFN)] {
        let nblk = width / BLOCK;
        let x_v = seq_vals(width, 11);
        let rot_v = hadamard(BLOCK);
        let want = rotate_ref(&x_v, BLOCK, nblk);
        let x = Tensor::from_vec(x_v, [1, width])?.to(dev)?;
        let rot = Tensor::from_vec(rot_v, [BLOCK, BLOCK])?.to(dev)?;

        // The checkpoint block: the fused plan does not compile (see
        // `butterfly_compile`), so the same stages run one realize each and
        // the transform is still checked against the dense form.
        let tape = Tape::new([&x])?;
        let dense = x.reshape([nblk, BLOCK])?.matmul(&rot)?.reshape([1, width])?;
        let frozen = tape.freeze([&x], [&dense])?;
        let got_dense: Vec<f32> = frozen.replay([&x])?.into_iter().next().expect("one output").to_vec()?;
        let got_bf: Vec<f32> = butterfly_staged(&x, BLOCK)?.to_vec()?;
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
        let (dense_err, _) = max_err(&got_dense, &want);
        let (bf_err, bf_at) = max_err(&got_bf, &want);
        let ulp = max_ulp(&got_bf, &got_dense, scale);

        // ---- timings, all on the model's own path (frozen plan replay,
        //      wall clock, one read at the end to drain the queue) ----
        let reps = 10;
        let dense_plan_us = frozen_us(|t| t.reshape([nblk, BLOCK])?.matmul(&rot)?.reshape([1, width]), &x, reps)?;
        let dense_plan_us_long = frozen_us(|t| t.reshape([nblk, BLOCK])?.matmul(&rot)?.reshape([1, width]), &x, 10 * reps)?;
        let started = std::time::Instant::now();
        let mut last = None;
        for _ in 0..reps {
            last = Some(butterfly_staged(&x, BLOCK)?);
        }
        let _: Vec<f32> = last.expect("a rotation ran").to_vec()?;
        let bf_us = started.elapsed().as_secs_f64() * 1e6 / reps as f64;

        let inv = Tensor::from_vec(vec![1.0f32], [1])?.to(dev)?;
        let norm = Tensor::from_vec(vec![0.5f32; usize::try_from(width).expect("width fits usize")], [width])?.to(dev)?;
        let signs = Tensor::from_vec(vec![1.0f32; usize::try_from(width).expect("width fits usize")], [width])?.to(dev)?;
        let flat = x.reshape([width])?;
        let scratch = Tensor::zeros([width], DType::F32).to(dev)?;
        let fast = fwht_kernel(dev, BLOCK, 256, nblk, nblk);
        fast.clone().verify();
        let fast_us = timed(fast, &[&flat, &inv, &norm, &signs, &scratch], 3)?;
        let dense_k = naive_rotation_kernel(dev, BLOCK, nblk, nblk);
        let dense_us = searched(dense_k, &[&flat, &inv, &norm, &signs, &rot, &scratch], None)?;

        println!(
            "butterfly rotation {label} [{width}]: dense plan {dense_plan_us:.2} us (x10 reps {dense_plan_us_long:.2} us; searched kernel {dense_us:.2} us), tensor-op staged {bf_us:.2} us, fwht {fast_us:.2} us (dense err {dense_err:.2e}, butterfly err {bf_err:.2e}@{bf_at}, butterfly-vs-dense {ulp:.1} ulp of |out|max, |x|max {scale:.2})"
        );
        assert!(
            bf_err <= 1e-5 * scale + 1e-5,
            "{label}: the butterfly must reproduce the Hadamard product: {bf_err:.3e} at {bf_at}"
        );
        assert!(
            ulp <= 32.0,
            "{label}: the butterfly must agree with the dense form to f32 rounding, got {ulp:.1} ulp of |out|max"
        );

        step[0] += if width == HIDDEN { HIDDEN_WIDE_PER_BLOCK } else { 1.0 } * dense_plan_us;
        step[1] += if width == HIDDEN { HIDDEN_WIDE_PER_BLOCK } else { 1.0 } * bf_us;
        step[2] += if width == HIDDEN { HIDDEN_WIDE_PER_BLOCK } else { 1.0 } * fast_us;
    }
    println!(
        "butterfly rotation per step (64 blocks x (4 x 5120 + 1 x 17408)): dense plan {:.2} ms, tensor-op staged {:.2} ms, fwht {:.2} ms (llama.cpp 257 launches = 0.47 ms)",
        BLOCKS_PER_STEP * step[0] * 1e-3,
        BLOCKS_PER_STEP * step[1] * 1e-3,
        BLOCKS_PER_STEP * step[2] * 1e-3
    );
    Ok(())
}

/// The same butterfly stages with a plan boundary between them: the smallest
/// change that lets `block 1024` reach the device at all.
fn butterfly_staged(x: &Tensor, block: i64) -> Result<Tensor, ZyxError> {
    let shape = x.shape();
    let rows = x.numel() / block;
    let block_t = Tensor::from(block);
    let mut t = x.reshape([rows.clone(), block_t.clone()])?;
    let mut h = 1i64;
    while h < block {
        let tape = Tape::new([&t])?;
        let view = t.reshape([
            rows.clone(),
            Tensor::from(block / (2 * h)),
            Tensor::from(2i64),
            Tensor::from(h),
        ])?;
        let low = view.narrow(-2, 0i64, 1i64)?;
        let high = view.narrow(-2, 1i64, 1i64)?;
        let sum = &low + &high;
        let diff = &low - &high;
        let next = Tensor::cat([&sum, &diff], -2)?.reshape([rows.clone(), block_t.clone()])?;
        let frozen = tape.freeze([&t], [&next])?;
        t = frozen.replay([&t])?.into_iter().next().expect("one output");
        h *= 2;
    }
    (t * Tensor::from(1.0f32 / (block as f32).sqrt())).reshape(shape)
}

/// Op growth of the fused butterfly, one plan per block size, built exactly
/// as `prism.rs::rotate` builds it.
///
/// The growth is the defect: the traced expression is `O(log2(block))` ops
/// (seven per stage), but the body `Graph::kernelize` emits is `4^stages` view
/// ops — every edge into a `Reshape`/`Narrow`/`Permute` pushes a fresh view op
/// and replays its shape expression with `replay_shape_into_kernel`, whose
/// dedupe map is per call, so no two views are CSE-equal. Measured, one
/// careful plan of `4 x block` elements: 106 ops at block 2, 451 at 4, 1896
/// at 8, 7677 at 16 (x4.2 per stage; `reshape(x)` alone is exactly `4^stages`:
/// 4, 16, 64, 256). At block 1024 that is ~4.3e7 ops and `4^10 = 1048576`
/// load entries. The search is skipped (`NoSearch`) because it would time
/// those bodies.
#[test]
fn butterfly_growth() -> Result<(), ZyxError> {
    let Some(dev) = gpu() else {
        return Ok(());
    };
    let _no_search = Tensor::with_debug(zyx::DebugMask::new(256));
    for block in [2i64, 4, 8, 16] {
        let width = 4 * block;
        let x = Tensor::from_vec(seq_vals(width, 14), [1, width])?.to(dev)?;
        let tape = Tape::new([&x])?;
        let out = butterfly(&x, block)?;
        let frozen = tape.freeze([&x], [&out])?;
        let _ = frozen.replay([&x])?;
        println!("butterfly growth block {block} [{width}]: plan ran");
    }
    Ok(())
}

/// What the generic kernelizer does with the whole butterfly in one plan, by
/// block size. Two invariants of `Graph::kernelize` catch the `4^stages`
/// body: the load-entry bookkeeping (`n_params == loads.len()`, 1024 entries
/// already at block 32, 4096 at 64) and `is_preceded_by_reduce`'s 250k-op
/// walk (block 128 and up).
#[test]
fn butterfly_compile() -> Result<(), ZyxError> {
    let Some(dev) = gpu() else {
        return Ok(());
    };
    for block in [32i64, 64, 128, 256, BLOCK] {
        let width = 4 * block;
        let x = Tensor::from_vec(seq_vals(width, 13), [1, width])?.to(dev)?;
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<bool, ZyxError> {
            let tape = Tape::new([&x])?;
            let out = butterfly(&x, block)?;
            let frozen = tape.freeze([&x], [&out])?;
            let _ = frozen.replay([&x])?;
            Ok(true)
        }));
        match attempt {
            Ok(Ok(_)) => println!("butterfly plan block {block} [{width}]: one fused plan compiles"),
            Ok(Err(e)) => println!("butterfly plan block {block} [{width}]: error {e}"),
            Err(_) => println!("butterfly plan block {block} [{width}]: kernelizer panics (fused 4^stages view body)"),
        }
    }
    Ok(())
}

/// Measured cost of the ordered form: the butterfly with `.contiguous()`
/// closing each stage, at the checkpoint's real widths, steady state (the
/// kernels are warm from the correctness pass before the timing replay).
#[test]
fn butterfly_contiguous_ms() -> Result<(), ZyxError> {
    let Some(dev) = gpu() else {
        return Ok(());
    };
    let block = BLOCK;
    let mut hidden_us = 0.0;
    let mut ffn_us = 0.0;
    for width in [HIDDEN, FFN] {
        let vals = seq_vals(width, 41);
        let x = Tensor::from_vec(vals.clone(), [1, width])?.to(dev)?;
        let got: Vec<f32> = butterfly_contiguous(&x, block)?.try_into()?;
        let want = rotate_ref(&vals, block, width / block);
        let peak = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        for (i, (g, e)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - e).abs() <= 2e-5 + 1e-5 * peak, "contiguous butterfly[{i}]: {g} != {e}");
        }
        // Two replays: the first covers any remaining cold cache, the second is
        // the per-rotation cost the model pays once per token.
        let _ = frozen_us(|t| butterfly_contiguous(t, block), &x, 2)?;
        let us = frozen_us(|t| butterfly_contiguous(t, block), &x, 4)?;
        if width == HIDDEN {
            hidden_us = us;
        } else {
            ffn_us = us;
        }
        println!("butterfly+contiguous [{width}]: {us:.1} us per rotation");
    }
    // 64 linear blocks per step, each rotating 4 hidden-wide activations
    // (gate, up, q, k) and one ffn-wide (down).
    let ms = 64.0 * (4.0 * hidden_us + ffn_us) * 1e-3;
    println!(
        "butterfly+contiguous per step: 64 x (4 x {hidden_us:.1} + {ffn_us:.1}) us = {ms:.1} ms \
         (hand FWHT 1.9 ms, llama.cpp 0.47 ms)"
    );
    Ok(())
}

/// Where the norm and SwiGLU gaps sit: the seed the graph lowers (one thread,
/// one serial accumulator / one element per thread) against the search's
/// winner at the decode sizes, with both IRs printed.
#[test]
fn band_search_gap() -> Result<(), ZyxError> {
    let Some(dev) = gpu() else {
        return Ok(());
    };
    let n = HIDDEN;
    let x = Tensor::from_vec(seq_vals(n, 2), [n])?.to(dev)?;
    let scratch = Tensor::zeros([1], DType::F32).to(dev)?;
    let seed = rms_reduce_kernel(dev, n, 1);
    let seed_us = timed(seed.clone(), &[&x, &scratch], 3)?;
    let (winner, win_us) = searched_winner(seed.clone(), &[&x, &scratch], None)?;
    println!("rms decode [{n}]: seed {seed_us:.2} us, winner {win_us:.2} us, llama 4.74 us");
    println!("--- rms seed IR ---");
    seed.debug();
    println!("--- rms winner IR ---");
    winner.debug();

    let m = FFN;
    let gate = Tensor::from_vec(seq_vals(m, 3), [m])?.to(dev)?;
    let up = Tensor::from_vec(seq_vals(m, 4), [m])?.to(dev)?;
    let sout = Tensor::zeros([m], DType::F32).to(dev)?;
    let seed = swiglu_kernel(dev, m);
    let seed_us = timed(seed.clone(), &[&gate, &up, &sout], 3)?;
    let (winner, win_us) = searched_winner(seed.clone(), &[&gate, &up, &sout], None)?;
    println!("swiglu decode [{m}]: seed {seed_us:.2} us, winner {win_us:.2} us, llama 1.84 us");
    println!("--- swiglu seed IR ---");
    seed.debug();
    println!("--- swiglu winner IR ---");
    winner.debug();
    Ok(())
}
