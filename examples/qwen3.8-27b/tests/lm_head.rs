// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! LM head test (runs on CUDA, fp16 weights with fp32 accumulation).
//!
//! Golden: `examples/data/qwen3_8b_lm_head.safetensors` from
//! `tests/lm_head_ref.py`. Run the dump first:
//! `cd tests && python3.12 lm_head_ref.py`.

use std::time::Instant;

use zyx::kernel::{Dev, Kernel, MMADims, MMADType, MMALayout, MemScope};
use zyx::DType;
use zyx::{Tensor, ZyxError};

const VOCAB: usize = 256;
const HIDDEN: usize = 64;
const TOKENS: usize = 8;
const ROWS_PER_BLOCK: usize = 32;
const MMA_N: usize = 8;

#[test]
fn lm_head() -> Result<(), ZyxError> {
    let goldens = Tensor::load("../data/qwen3_8b_lm_head.safetensors")?;
    let weight = goldens["weight"].to(Dev::Cuda(0))?;
    let input = goldens["input"].to(Dev::Cuda(0))?;
    let expected = goldens["output"].to_vec::<f32>()?;

    let out = input.dot_dtype(weight.t(), DType::F32)?.to_vec::<f32>()?;

    assert_eq!(out.len(), expected.len());
    for (i, (&v, &e)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((v - e).abs() < 1e-4, "out[{i}] = {v}, expected {e}");
    }
    Ok(())
}

/// Custom CUDA kernel for the LM head: fp16 tensor-core matmul mirroring
/// llama.cpp's Turing `mul_mat_f` (mmf.cu) structure.
///
/// One warp (32 threads) per block computes a 32 (vocab rows) x 8 (tokens)
/// output tile as two m16n8k8 mma subtiles with fp32 accumulation. K (hidden)
/// is iterated in chunks of 8. A = weight (row-major [vocab, hidden]), B =
/// input (column-major [hidden, tokens] view of the [tokens, hidden] input),
/// C = logits ([vocab, tokens]). `hidden`, `tokens` and the grid sizes are
/// runtime args, like llama.cpp's runtime kernel parameters. Divisibility
/// (vocab % 32, tokens % 8, hidden % 8) is assumed, as in mmf's GGML_ASSERTs.
#[test]
fn lm_head_cuda() -> Result<(), ZyxError> {
    let goldens = Tensor::load("../data/qwen3_8b_lm_head.safetensors")?;
    let weight = goldens["weight"].to(Dev::Cuda(0))?;
    let input = goldens["input"].reshape([TOKENS as i64, HIDDEN as i64])?.to(Dev::Cuda(0))?;
    let expected = goldens["output"].to_vec::<f32>()?;

    let mut kernel = Kernel::new(Dev::Cuda(0));

    // Runtime args (llama.cpp passes these as kernel parameters).
    let hidden = kernel.variable(DType::I64);
    let tokens = kernel.variable(DType::I64);
    let glen_x = kernel.variable(DType::I64); // vocab / rows_per_block
    let glen_y = kernel.variable(DType::I64); // tokens / mma_n

    let w = kernel.param(DType::F16);
    let x = kernel.param(DType::F16);
    let out = kernel.param_mut(DType::F32);

    let gidx = kernel.group_range(0, glen_x);
    let gidy = kernel.group_range(1, glen_y);
    let wid = kernel.local_range(0, 32);

    let [c0, c1, c2, c4, c8, c16, c32] = kernel.const_idxs([0u32, 1, 2, 4, 8, 16, 32]);

    let gid = kernel.div(wid, c4);
    let tig = kernel.mod_(wid, c4);
    let col2 = kernel.mul(tig, c2);

    // Two register accumulators (one per m16n8k8 subtile).
    let acc0 = kernel.storage(DType::F32, MemScope::Register, 4);
    let acc1 = kernel.storage(DType::F32, MemScope::Register, 4);
    let zf = kernel.const_val(0.0f32);
    let zero4 = kernel.stack(&[zf, zf, zf, zf]);
    kernel.store_vector(acc0, zero4, c0, 4);
    kernel.store_vector(acc1, zero4, c0, 4);

    // A row bases: r0 = gidx*32 + gid, r1 = r0 + 16. Fragment rows are
    // r_s and r_s + 8.
    let r0 = kernel.mad(gidx, c32, gid);
    let r1 = kernel.add(r0, c16);
    // B token: n0 = gidy*8 + gid.
    let n0 = kernel.mad(gidy, c8, gid);

    let k_tiles = kernel.div(hidden, c8);
    kernel.loop_over(k_tiles, |k, k_loop| {
        let k0 = k.mul(k_loop, c8);
        let a_col = k.add(k0, col2);

        // A fragment addresses: row*hidden + k0 + tig*2 (+1), +8*hidden for the
        // upper 8 rows of the fragment.
        let a_base0 = k.mad(r0, hidden, a_col);
        let a_base0_hi = k.mad(c8, hidden, a_base0);
        let a_base0_p1 = k.add(a_base0, c1);
        let a_base0_hi_p1 = k.add(a_base0_hi, c1);
        let a00 = k.load(w, a_base0);
        let a01 = k.load(w, a_base0_p1);
        let a02 = k.load(w, a_base0_hi);
        let a03 = k.load(w, a_base0_hi_p1);
        let a_frag0 = k.stack(&[a00, a01, a02, a03]);

        let a_base1 = k.mad(r1, hidden, a_col);
        let a_base1_hi = k.mad(c8, hidden, a_base1);
        let a_base1_p1 = k.add(a_base1, c1);
        let a_base1_hi_p1 = k.add(a_base1_hi, c1);
        let a10 = k.load(w, a_base1);
        let a11 = k.load(w, a_base1_p1);
        let a12 = k.load(w, a_base1_hi);
        let a13 = k.load(w, a_base1_hi_p1);
        let a_frag1 = k.stack(&[a10, a11, a12, a13]);

        // B fragment addresses: token*hidden + k0 + tig*2 (+1) — consecutive k.
        let b_base = k.mad(n0, hidden, a_col);
        let b_base_p1 = k.add(b_base, c1);
        let b0 = k.load(x, b_base);
        let b1 = k.load(x, b_base_p1);
        let b_frag = k.stack(&[b0, b1]);

        let acc_old0 = k.load_vector(acc0, c0, 4);
        let acc_new0 = k.wmma(MMADims::m16n8k8, MMALayout::row_col, MMADType::f16_f16_f16_f32, a_frag0, b_frag, acc_old0);
        k.store_vector(acc0, acc_new0, c0, 4);

        let acc_old1 = k.load_vector(acc1, c0, 4);
        let acc_new1 = k.wmma(MMADims::m16n8k8, MMALayout::row_col, MMADType::f16_f16_f16_f32, a_frag1, b_frag, acc_old1);
        k.store_vector(acc1, acc_new1, c0, 4);
    });

    // C stores: out[row*tokens + col], col = gidy*8 + tig*2.
    let col = kernel.mad(gidy, c8, col2);
    let o0 = kernel.mad(r0, tokens, col);
    let o0_hi = kernel.mad(c8, tokens, o0);
    let o1 = kernel.mad(r1, tokens, col);
    let o1_hi = kernel.mad(c8, tokens, o1);

    let acc0_final = kernel.load_vector(acc0, c0, 4);
    let [d00, d01, d02, d03] = kernel.devectorize(acc0_final);
    let o0_p1 = kernel.add(o0, c1);
    let o0_hi_p1 = kernel.add(o0_hi, c1);
    kernel.store(out, d00, o0);
    kernel.store(out, d01, o0_p1);
    kernel.store(out, d02, o0_hi);
    kernel.store(out, d03, o0_hi_p1);

    let acc1_final = kernel.load_vector(acc1, c0, 4);
    let [d10, d11, d12, d13] = kernel.devectorize(acc1_final);
    let o1_p1 = kernel.add(o1, c1);
    let o1_hi_p1 = kernel.add(o1_hi, c1);
    kernel.store(out, d10, o1);
    kernel.store(out, d11, o1_p1);
    kernel.store(out, d12, o1_hi);
    kernel.store(out, d13, o1_hi_p1);

    let compiled = kernel.compile()?;

    let hidden_t = Tensor::from(HIDDEN as i64);
    let tokens_t = Tensor::from(TOKENS as i64);
    let glen_x_t = Tensor::from((VOCAB / ROWS_PER_BLOCK) as i64);
    let glen_y_t = Tensor::from((TOKENS / MMA_N) as i64);

    let launch = || -> Result<Vec<Tensor>, ZyxError> {
        compiled.forward(
            &[&hidden_t, &tokens_t, &glen_x_t, &glen_y_t, &weight, &input],
            vec![[VOCAB as i64, TOKENS as i64]],
        )
    };

    // Correctness: C is [vocab, tokens], golden is [tokens, vocab].
    let out = launch()?.remove(0).to_vec::<f32>()?;
    for t in 0..TOKENS {
        for v in 0..VOCAB {
            let got = out[v * TOKENS + t];
            let exp = expected[t * VOCAB + v];
            assert!((got - exp).abs() < 1e-4, "out[{v}, {t}] = {got}, expected {exp}");
        }
    }

    // Timing.
    let iters = 1000;
    let start = Instant::now();
    for _ in 0..iters {
        launch()?;
    }
    let _ = launch()?.remove(0).to_vec::<f32>()?;
    let us_per_iter = start.elapsed().as_secs_f64() * 1e6 / (iters + 1) as f64;
    let flops = 2.0 * VOCAB as f64 * TOKENS as f64 * HIDDEN as f64;
    println!(
        "lm_head_cuda: {us_per_iter:.2} µs/iter, {:.2} GFLOP/s",
        flops / us_per_iter / 1e3
    );
    Ok(())
}
