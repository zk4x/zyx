// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `dequant_q4k_tt(4)` on the P100A: 4 F32 tilized tiles from plane-interleaved
//! U16 + full-tile BF16 scales/mins (first 4 tiles of blk.1.attn_qkv).
//! Golden is host-computed F32 from the same slices (face math below);
//! compared tilized-flat at 1e-2 abs like the GEMM tests.

use qwen3_8_27b::dequant_q4k_tt;
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

#[test]
fn dequant_q4k_tt_run() -> Result<(), ZyxError> {
    let dev = Dev::TT(0);
    let repacked = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_attn_qkv_q4k_repacked.safetensors")?;
    let packed = repacked["packed"].narrow(0, 0i64, 1024i64)?.to(dev)?;
    // Test expands the dense [4, 32] sidecar to full [4, 1024] tiles
    // host-side (each slot replicated across its row); production keeps
    // the dense file with on-device broadcast.
    let sv0: Vec<zyx::bf16> = repacked["scales"].narrow(0, 0i64, 4i64)?.to_vec()?;
    let mv0: Vec<zyx::bf16> = repacked["mins"].narrow(0, 0i64, 4i64)?.to_vec()?;
    let mut sc_full = Vec::with_capacity(4 * 1024);
    let mut mn_full = Vec::with_capacity(4 * 1024);
    for t in 0..4 {
        for i in 0..32 {
            sc_full.extend(std::iter::repeat(sv0[t * 32 + i]).take(32));
            mn_full.extend(std::iter::repeat(mv0[t * 32 + i]).take(32));
        }
    }
    let scales = Tensor::from_vec(sc_full, [4i64, 1024])?.to(dev)?;
    let mins = Tensor::from_vec(mn_full, [4i64, 1024])?.to(dev)?;

    let kk = dequant_q4k_tt(4);
    let k = kk.compile()?;
    // REVIEW-THEN-LAUNCH (AGENTS.md): with ZYX_TT_DUMP_ONLY=1, stop after
    // compile so generated sources (ZYX_DEBUG=16) can be compared against
    // the official tt-metal kernels before anything executes on the board.
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        return Ok(());
    }
    let out = k.forward(&[&packed, &scales, &mins], vec![[4 * 1024]])?;
    let v: Vec<f32> = out[0].to_vec()?;

    // Host golden from llama.cpp itself (gguf python dequantize), tilized-flat.
    let gold: Vec<f32> = Tensor::load("/tmp/opengen/golden_q4k_first4tiles.safetensors")?
        .remove("x")
        .expect("golden x")
        .to_vec()?;
    assert_eq!(v.len(), 4096);
    let mut max_err = 0f32;
    for (s, (&got, &exp)) in v.iter().zip(gold.iter()).enumerate() {
        max_err = max_err.max((got - exp).abs());
        assert!((got - exp).abs() < 1e-2, "tile {s}/1024: got {got}, expected {exp}");
    }
    eprintln!("dequant_q4k_tt max_err {max_err}");
    Ok(())
}
