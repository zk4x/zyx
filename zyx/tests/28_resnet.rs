// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `resnet50-v2-7` (the `onnx/models` zoo export that
//! `examples/onnx-resnet` runs) as one tape in tensor ops: every op and every
//! layer of the network, with the same shapes and the same op order the ONNX
//! graph has, so every kernel the model schedules exists here and can be
//! worked on in isolation.
//!
//! The graph, read off `resnet50-v2-7.onnx`:
//!
//! ```text
//! BN(3) -> Conv 7x7/2 pad 3 -> BN -> Relu -> MaxPool 3x3/2 pad 1
//! for each stage (inner, out, blocks, stride):
//!     (64, 256, 3, 1), (128, 512, 4, 2), (256, 1024, 6, 2), (512, 2048, 3, 2)
//!   for each block:
//!     a = Relu(BN(h))                 // bn0
//!     t = Conv 1x1/stride(a)          // conv0
//!     t = Relu(BN(t))                 // bn1, conv1
//!     t = Conv 3x3/1 pad 1(t)
//!     t = Relu(BN(t))                 // bn2, conv2
//!     t = Conv 1x1/1(t)
//!     h = t + (Conv 1x1/stride(a) if first block of the stage else h)
//! BN -> Relu -> GlobalAveragePool -> Reshape [1, 2048] -> Gemm [1000, 2048]
//! ```
//!
//! This is the pre-activation (`v2`) order: the BN that follows an `Add`
//! belongs to the next block, and the shortcut of a stage's first block is a
//! strided 1x1 conv of the block's own activation (no BN on it). The
//! classifier `Gemm` is included; the example's top-1 argmax is not part of
//! the model.
//!
//! Weights are deterministic values in `[-1, 1]`, so the test needs no
//! downloaded model. The kernels are the point, not the numbers: the numeric
//! agreement with ONNX Runtime lives in `examples/onnx-resnet`. The tail of
//! this file is the fused kernel measured at 322.6 ms in
//! `ZYX_NO_CACHE=1 ZYX_DEBUG=1` on `Cuda(0)` (`examples/onnx-resnet`), which
//! is the one that motivated this harness.

use zyx::{DType, Dev, Tape, Tensor, ZyxError};

const EPS: f32 = 1e-5;
const IMAGE: i64 = 224;

/// Deterministic operands in `[-1, 1]`, so the test needs no RNG.
fn values(n: usize, seed: u64) -> Vec<f32> {
    (0..n).map(|i| ((i as u64).wrapping_mul(2654435761).wrapping_add(seed) % 2000) as f32 / 1000.0 - 1.0).collect()
}

#[test]
fn resnet50() -> Result<(), ZyxError> {
    // Without a GPU the same ops run on `C`; the sequence and the kernel set
    // are the same, only the device code differs.
    let dev = Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)).unwrap_or(Dev::C);

    // Every initializer is a distinct tensor, exactly as the ONNX graph has
    // them; `seed` advances once per tensor.
    let tensor = |v: &[f32], shape: &[i64]| -> Result<Tensor, ZyxError> { Tensor::from_vec(v.to_vec(), shape.to_vec()) };
    let mut seed = 0u64;

    let x = tensor(&values(3 * (IMAGE * IMAGE) as usize, seed), &[1, 3, IMAGE, IMAGE])?;
    seed += 1;

    // Built under a tape: this is the path `zyx-onnx` replays, and the one
    // whose kernelizer fuses each BN/Relu/conv chain.
    let tape = Tape::empty();
    tape.add(&x)?;

    // ---- stem ------------------------------------------------------------
    // BatchNormalization on the 3-channel input.
    let cs = [Tensor::from(1i64), Tensor::from(3i64), Tensor::from(1i64), Tensor::from(1i64)];
    let gamma = tensor(&values(3, seed), &[3])?;
    seed += 1;
    let beta = tensor(&values(3, seed), &[3])?;
    seed += 1;
    let mean = tensor(&values(3, seed), &[3])?;
    seed += 1;
    let var = tensor(&values(3, seed).iter().map(|v| 1.0 + v.abs()).collect::<Vec<f32>>(), &[3])?;
    seed += 1;
    let mut h = (x.clone() - mean.reshape(cs.to_vec())?) / (var.reshape(cs.to_vec())? + EPS).sqrt()
        * gamma.reshape(cs.to_vec())?
        + beta.reshape(cs.to_vec())?;

    // Conv 7x7/2 pad 3 -> 64 channels, then BN, Relu, MaxPool 3x3/2 pad 1.
    let w = tensor(&values(64 * 3 * 7 * 7, seed), &[64, 3, 7, 7])?;
    seed += 1;
    h = h.conv(&w, None, 1, [2i64, 2], [1i64, 1], [3i64, 3, 3, 3])?;

    let cs = [
        Tensor::from(1i64),
        Tensor::from(64i64),
        Tensor::from(1i64),
        Tensor::from(1i64),
    ];
    let gamma = tensor(&values(64, seed), &[64])?;
    seed += 1;
    let beta = tensor(&values(64, seed), &[64])?;
    seed += 1;
    let mean = tensor(&values(64, seed), &[64])?;
    seed += 1;
    let var = tensor(&values(64, seed).iter().map(|v| 1.0 + v.abs()).collect::<Vec<f32>>(), &[64])?;
    seed += 1;
    h = (h - mean.reshape(cs.to_vec())?) / (var.reshape(cs.to_vec())? + EPS).sqrt() * gamma.reshape(cs.to_vec())?
        + beta.reshape(cs.to_vec())?;
    h = h.relu();
    h = h.max_pool([3i64, 3], [2i64, 2], [1i64, 1], [(0, 0), (0, 0), (1, 1), (1, 1)], false, false)?;

    // ---- the four stages -------------------------------------------------
    // (inner channels, output channels, blocks, stride of the first block)
    let stages: [(i64, i64, usize, i64); 4] = [(64, 256, 3, 1), (128, 512, 4, 2), (256, 1024, 6, 2), (512, 2048, 3, 2)];
    let mut in_ch = 64i64;
    for (inner, cout, nblocks, stage_stride) in stages {
        for block in 0..nblocks {
            // The first block of a stage strided-convs its shortcut and
            // downsamples; every other block is the identity. Only the first
            // block takes the previous stage's width, the rest take the
            // stage's output width.
            let stride = if block == 0 { stage_stride } else { 1 };
            let block_in = if block == 0 { in_ch } else { cout };

            // bn0 -> relu
            let cs = [
                Tensor::from(1i64),
                Tensor::from(block_in),
                Tensor::from(1i64),
                Tensor::from(1i64),
            ];
            let gamma = tensor(&values(block_in as usize, seed), &[block_in])?;
            seed += 1;
            let beta = tensor(&values(block_in as usize, seed), &[block_in])?;
            seed += 1;
            let mean = tensor(&values(block_in as usize, seed), &[block_in])?;
            seed += 1;
            let var = tensor(&values(block_in as usize, seed).iter().map(|v| 1.0 + v.abs()).collect::<Vec<f32>>(), &[block_in])?;
            seed += 1;
            let a = ((h.clone() - mean.reshape(cs.to_vec())?) / (var.reshape(cs.to_vec())? + EPS).sqrt()
                * gamma.reshape(cs.to_vec())?
                + beta.reshape(cs.to_vec())?)
            .relu();

            // conv0 (1x1, strided on the stage's first block)
            let w0 = tensor(&values((inner * block_in) as usize, seed), &[inner, block_in, 1, 1])?;
            seed += 1;
            let mut t = a.conv(&w0, None, 1, [stride, stride], [1i64, 1], [0i64, 0, 0, 0])?;

            // bn1 -> relu
            let cs = [
                Tensor::from(1i64),
                Tensor::from(inner),
                Tensor::from(1i64),
                Tensor::from(1i64),
            ];
            let gamma = tensor(&values(inner as usize, seed), &[inner])?;
            seed += 1;
            let beta = tensor(&values(inner as usize, seed), &[inner])?;
            seed += 1;
            let mean = tensor(&values(inner as usize, seed), &[inner])?;
            seed += 1;
            let var = tensor(&values(inner as usize, seed).iter().map(|v| 1.0 + v.abs()).collect::<Vec<f32>>(), &[inner])?;
            seed += 1;
            t = ((t - mean.reshape(cs.to_vec())?) / (var.reshape(cs.to_vec())? + EPS).sqrt() * gamma.reshape(cs.to_vec())?
                + beta.reshape(cs.to_vec())?)
            .relu();

            // conv1 (3x3, stride 1, pad 1)
            let w1 = tensor(&values((inner * inner * 9) as usize, seed), &[inner, inner, 3, 3])?;
            seed += 1;
            t = t.conv(&w1, None, 1, [1i64, 1], [1i64, 1], [1i64, 1, 1, 1])?;

            // bn2 -> relu
            let cs = [
                Tensor::from(1i64),
                Tensor::from(inner),
                Tensor::from(1i64),
                Tensor::from(1i64),
            ];
            let gamma = tensor(&values(inner as usize, seed), &[inner])?;
            seed += 1;
            let beta = tensor(&values(inner as usize, seed), &[inner])?;
            seed += 1;
            let mean = tensor(&values(inner as usize, seed), &[inner])?;
            seed += 1;
            let var = tensor(&values(inner as usize, seed).iter().map(|v| 1.0 + v.abs()).collect::<Vec<f32>>(), &[inner])?;
            seed += 1;
            t = ((t - mean.reshape(cs.to_vec())?) / (var.reshape(cs.to_vec())? + EPS).sqrt() * gamma.reshape(cs.to_vec())?
                + beta.reshape(cs.to_vec())?)
            .relu();

            // conv2 (1x1, to the block's output width)
            let w2 = tensor(&values((cout * inner) as usize, seed), &[cout, inner, 1, 1])?;
            seed += 1;
            t = t.conv(&w2, None, 1, [1i64, 1], [1i64, 1], [0i64, 0, 0, 0])?;

            // shortcut: the stage's first block convolves its own activation,
            // every other block adds its input directly.
            let res = if block == 0 {
                let ws = tensor(&values((cout * block_in) as usize, seed), &[cout, block_in, 1, 1])?;
                seed += 1;
                a.conv(&ws, None, 1, [stride, stride], [1i64, 1], [0i64, 0, 0, 0])?
            } else {
                h
            };
            h = t + res;
        }
        in_ch = cout;
    }

    // ---- head ------------------------------------------------------------
    // BN on the last stage's output, Relu, GlobalAveragePool, Reshape, Gemm.
    let cs = [
        Tensor::from(1i64),
        Tensor::from(2048i64),
        Tensor::from(1i64),
        Tensor::from(1i64),
    ];
    let gamma = tensor(&values(2048, seed), &[2048])?;
    seed += 1;
    let beta = tensor(&values(2048, seed), &[2048])?;
    seed += 1;
    let mean = tensor(&values(2048, seed), &[2048])?;
    seed += 1;
    let var = tensor(&values(2048, seed).iter().map(|v| 1.0 + v.abs()).collect::<Vec<f32>>(), &[2048])?;
    seed += 1;
    h = (h - mean.reshape(cs.to_vec())?) / (var.reshape(cs.to_vec())? + EPS).sqrt() * gamma.reshape(cs.to_vec())?
        + beta.reshape(cs.to_vec())?;
    h = h.relu();
    h = h.mean_keepdim([2i32, 3])?;
    h = h.reshape([1i64, 2048])?;
    let w = tensor(&values(1000 * 2048, seed), &[1000, 2048])?;
    seed += 1;
    let bias = tensor(&values(1000, seed), &[1000])?;
    let logits = h.matmul(&w.t())? + bias;

    let frozen = tape.freeze([&x], [&logits])?;
    let logits = frozen.replay([&x.to(dev)?])?.remove(0);

    assert_eq!(logits.resolve_shape(), [1i64, 1000], "classifier output shape");
    let got: Vec<f32> = logits.cast(DType::F32).try_into()?;
    assert!(got.iter().all(|v| v.is_finite()), "resnet50 produced non-finite logits");
    let min = got.iter().copied().fold(f32::INFINITY, f32::min);
    let max = got.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(max > min, "resnet50 logits are constant ({min}), the graph did not compute");
    Ok(())
}
