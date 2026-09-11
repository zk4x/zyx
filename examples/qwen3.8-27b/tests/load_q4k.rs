// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `load_gguf` Q4_K arm: raw super-blocks as [num_blocks, 144] U8.
//! Smoke on the real file: header parse is lazy, only 64 blocks (9KB)
//! materialize through `repack_q4k`. Placement is automatic.

use qwen3_8_27b::repack_q4k;
use zyx::{DType, Tensor, ZyxError};

#[test]
fn load_q4k_raw() -> Result<(), ZyxError> {
    let (_meta, tensors) =
        Tensor::load_gguf("/home/x/Dev/rust/zyx/examples/models/Qwen3.8-27B-UD-Q4_K_XL.gguf")?;
    let w = &tensors["blk.1.attn_qkv.weight"];
    assert_eq!(w.dtype(), DType::U8);
    // 5120*10240 weights / 256 per block = 204800 blocks of 144B.
    let blocks: Vec<u8> = w.narrow(0, 0i64, 1i64)?.contiguous()?.to_vec()?;
    assert_eq!(blocks.len(), 144);
    // First 64 blocks = 16384 weights, repacked as a 64x256 slice.
    let first = w.narrow(0, 0i64, 64i64)?;
    let (packed, scales, mins) = repack_q4k(&first, 64, 256)?;
    assert_eq!(packed.dtype(), DType::U16);
    let pv: Vec<u16> = packed.to_vec()?;
    assert_eq!(pv.len(), 64 * 256 / 4);
    let sv: Vec<zyx::bf16> = scales.to_vec()?;
    let mv: Vec<zyx::bf16> = mins.to_vec()?;
    assert_eq!(sv.len(), 16 * 32);
    assert_eq!(mv.len(), 16 * 32);
    // Sanity: dequant of block 0 weight 0 matches d0 * sc - dmin * m symbolically;
    // exact values are pinned by the repack round-trip test, here only liveness.
    assert!(sv.iter().any(|&x| x.to_f32() != 0.0));
    Ok(())
}
