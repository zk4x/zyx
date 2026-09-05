// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! LM head test (runs on CUDA, fp16 weights with fp32 accumulation).
//!
//! Golden: `examples/data/qwen3_8b_lm_head.safetensors` from
//! `tests/lm_head_ref.py`. Run the dump first:
//! `cd tests && python3.12 lm_head_ref.py`.

use std::time::Instant;

use zyx::kernel::{Dev, Kernel};
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
    let input = goldens["input"]
        .reshape([TOKENS as i64, HIDDEN as i64])?
        .to(Dev::Cuda(0))?;
    let expected = goldens["output"].to_vec::<f32>()?;

    let mut kernel = Kernel::new(Dev::Cuda(0));

    // Runtime args (llama.cpp passes these as kernel parameters).
    let vocab = kernel.variable(DType::I64);
    let hidden = kernel.variable(DType::I64);
    let tokens = kernel.variable(DType::I64);
    let glen_x = kernel.variable(DType::I64); // vocab / rows_per_block
    let glen_y = kernel.variable(DType::I64); // tokens / mma_n

    let w = kernel.param(DType::F16);
    let x = kernel.param(DType::F16);
    let out = kernel.param_mut(DType::F32);

    let gidx = kernel.group_range(0, glen_x);
    let gidy = kernel.group_range(1, glen_y);
    // The ONLY thread-machinery line: every partition method finds the warp
    // op via open_warp (the IR is the state).
    let lidx = kernel.local_range(0, 32);
    kernel.warp(lidx);

    // Views: fully symbolic iteration shapes, row-major strides derived.
    let wp = kernel.partition(w, [vocab, hidden]); // A: [vocab, hidden]
    let xp = kernel.partition(x, [tokens, hidden]); // B: [tokens, hidden], consecutive K
    let cp = kernel.partition(out, [vocab, tokens]); // C: [vocab, tokens]

    let [c8, c16, c32] = kernel.const_idxs([8u32, 16, 32]);

    // Chunk coords: one [32, 8] block tile = two m16n8k8 subtiles per warp.
    let r0 = kernel.mul(gidx, c32);
    let r1 = kernel.add(r0, c16);
    let n0 = kernel.mul(gidy, c8);

    let mut acc0 = kernel.acc([c16, c8], DType::F32);
    let mut acc1 = kernel.acc([c16, c8], DType::F32);

    // Loop length (hidden / 8) is derived and patched by the first mma.
    kernel.loop_partition(|kernel, k| {
        kernel.mma(&mut acc0, &wp, &xp, &[r0, n0, k]);
        kernel.mma(&mut acc1, &wp, &xp, &[r1, n0, k]);
    });

    kernel.store_partition(&cp, &acc0);
    kernel.store_partition(&cp, &acc1);

    let compiled = kernel.compile()?;

    let vocab_t = Tensor::from(VOCAB as i64);
    let hidden_t = Tensor::from(HIDDEN as i64);
    let tokens_t = Tensor::from(TOKENS as i64);
    let glen_x_t = Tensor::from((VOCAB / ROWS_PER_BLOCK) as i64);
    let glen_y_t = Tensor::from((TOKENS / MMA_N) as i64);

    // Correctness: C is [vocab, tokens], golden is [tokens, vocab].
    let mut out = compiled.forward(
        &[
            &vocab_t, &hidden_t, &tokens_t, &glen_x_t, &glen_y_t, &weight, &input,
        ],
        vec![[VOCAB as i64, TOKENS as i64]],
    )?;
    let out = out.pop().unwrap().to_vec::<f32>()?;
    for t in 0..TOKENS {
        for v in 0..VOCAB {
            let got = out[v * TOKENS + t];
            let exp = expected[t * VOCAB + v];
            assert!(
                (got - exp).abs() < 1e-4,
                "out[{v}, {t}] = {got}, expected {exp}"
            );
        }
    }

    // Timing with real Qwen3-8B lm_head dims: vocab 151936, hidden 2048,
    // tokens 8 (single-token generation). The kernel structure is unchanged —
    // rows-per-block (32), MMA_N (8) and the warp size are the only constants
    // baked in, and 151936 % 32 == 0, 2048 % 8 == 0.
    let weight_r = Tensor::rand([151936i64, 2048i64], DType::F16)?.to(Dev::Cuda(0))?;
    let input_r = Tensor::rand([TOKENS as i64, 2048i64], DType::F16)?.to(Dev::Cuda(0))?;
    let vocab_r = Tensor::from(151936i64);
    let hidden_r = Tensor::from(2048i64);
    let tokens_r = Tensor::from(TOKENS as i64);
    let glen_x_r = Tensor::from((151936 / ROWS_PER_BLOCK) as i64);
    let glen_y_r = Tensor::from((TOKENS / MMA_N) as i64);

    let launch_r = || -> Result<Vec<Tensor>, ZyxError> {
        compiled.forward(
            &[
                &vocab_r, &hidden_r, &tokens_r, &glen_x_r, &glen_y_r, &weight_r, &input_r,
            ],
            vec![[151936i64, TOKENS as i64]],
        )
    };

    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        launch_r()?;
    }
    let _ = launch_r()?.remove(0).to_vec::<f32>()?;
    let us_per_iter = start.elapsed().as_secs_f64() * 1e6 / (iters + 1) as f64;
    let flops = 2.0 * 151936.0 * TOKENS as f64 * 2048.0;
    println!(
        "lm_head_cuda: {us_per_iter:.2} µs/iter, {:.2} GFLOP/s",
        flops / us_per_iter / 1e3
    );
    Ok(())
}
