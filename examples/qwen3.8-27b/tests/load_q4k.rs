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
    // Real-data values: host reference straight from block-0 bytes,
    // independent of the tensor pipeline above. 64x256 slice has block b
    // = row b with c32 = 8, so tile (0,tc) row 0 = block-0 sub-block tc
    // and scales[tc*32] pins it.
    let d0 = zyx::f16::from_bits(blocks[0] as u16 | ((blocks[1] as u16) << 8)).to_f32();
    let dmin0 = zyx::f16::from_bits(blocks[2] as u16 | ((blocks[3] as u16) << 8)).to_f32();
    let sb0: &[u8] = &blocks[4..16];
    for tc in 0..8usize {
        let (esc, em) = if tc < 4 {
            (sb0[tc] & 63, sb0[tc + 4] & 63)
        } else {
            (
                (sb0[tc + 4] & 15) | ((sb0[tc - 4] >> 6) << 4),
                (sb0[tc + 4] >> 4) | ((sb0[tc] >> 6) << 4),
            )
        };
        let (exp_sc, exp_mn) = (d0 * esc as f32, dmin0 * em as f32);
        let (got_sc, got_mn) = (sv[tc * 32].to_f32(), mv[tc * 32].to_f32());
        // Pipeline-consistency level only (BF16-relative tol): the real
        // verification is device dequant vs original weight on the board.
        assert!(got_sc.is_finite(), "scales non-finite at sub-block {tc}");
        assert!(got_mn.is_finite(), "mins non-finite at sub-block {tc}");
        assert!(
            (got_sc - exp_sc).abs() < 1e-4 + 0.01 * exp_sc.abs(),
            "scales mismatch at sub-block {tc}: got {got_sc}, expected {exp_sc}"
        );
        assert!(
            (got_mn - exp_mn).abs() < 1e-4 + 0.01 * exp_mn.abs(),
            "mins mismatch at sub-block {tc}: got {got_mn}, expected {exp_mn}"
        );
    }
    // Sanity: dequant of block 0 weight 0 matches d0 * sc - dmin * m symbolically;
    // exact values are pinned by the repack round-trip test, here only liveness.
    assert!(sv.iter().any(|&x| x.to_f32() != 0.0));
    Ok(())
}
