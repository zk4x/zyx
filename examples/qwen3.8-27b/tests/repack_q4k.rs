// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `repack_q4k`: gguf Q4_K raw super-blocks -> device-ready dequant layout.
//! Placement is automatic (fastest available backend, no `.to(dev)`).
//! Byte-exact round-trip: nibbles via untilize inversion, scales via
//! forward re-packing of the recovered 6-bit values.

use qwen3_8_27b::repack_q4k;
use zyx::{DType, Tensor, ZyxError};

#[test]
fn repack_q4k_roundtrip() -> Result<(), ZyxError> {
    let (rows, cols) = (64i64, 512i64);
    let n = (rows * cols / 256) as usize; // 128 super-blocks
    // Deterministic pattern; fixed d = 1.0 (0x3C00), dmin = 0.5 (0x3800)
    // so the F16 scale math (d*sc, dmin*m) inverts exactly via round().
    let mut raw = vec![0u8; n * 144];
    for b in 0..n {
        for i in 0..144 {
            raw[b * 144 + i] = ((b * 144 + i) * 7 + 3) as u8;
        }
        raw[b * 144] = 0x00;
        raw[b * 144 + 1] = 0x3c; // d = 1.0
        raw[b * 144 + 2] = 0x00;
        raw[b * 144 + 3] = 0x38; // dmin = 0.5
    }
    let raw_t = Tensor::from_vec(raw.clone(), [n as i64, 144])?;
    let (packed, scales, mins) = repack_q4k(&raw_t, rows, cols)?;

    assert_eq!(packed.dtype(), DType::U16);
    assert_eq!(scales.dtype(), DType::BF16);
    assert_eq!(mins.dtype(), DType::BF16);
    let pv: Vec<u16> = packed.to_vec()?;
    assert_eq!(pv.len(), (rows * cols / 4) as usize);
    let sv: Vec<zyx::bf16> = scales.to_vec()?;
    let mv: Vec<zyx::bf16> = mins.to_vec()?;
    let ntiles = (rows / 32 * (cols / 32)) as usize;
    assert_eq!(sv.len(), ntiles * 32);
    assert_eq!(mv.len(), ntiles * 32);

    // Spot check: tilized slot 0 is weight (0,0) = super-block 0, qs byte 0 low.
    assert_eq!(pv[0] & 15, u16::from(raw[16] & 15));

    // Full qs round-trip: packed words -> strided pages -> tilized flat
    // (word 1024p+s = tiles 4p+k slot s) -> untilize -> row-major.
    let l = (rows * cols) as usize;
    let mut til = vec![0u16; l];
    for (i, &w) in pv.iter().enumerate() {
        let (p, s) = (i / 1024, i % 1024);
        for k in 0..4 {
            til[(p * 4 + k) * 1024 + s] = (w >> (4 * k)) & 15;
        }
    }
    let nib: Vec<u16> = Tensor::from_vec(til, [rows, cols])?.untilize(rows, cols)?.to_vec()?;
    for b in 0..n {
        for w in 0..256 {
            let g = w / 64;
            let p = w % 64;
            let (byte, high) = if p < 32 { (32 * g + p, false) } else { (32 * g + p - 32, true) };
            let v = nib[b * 256 + w];
            let got = raw[b * 144 + 16 + byte];
            let exp = if high { (got >> 4) & 15 } else { got & 15 };
            assert_eq!(v, u16::from(exp), "super-block {b} weight {w}");
        }
    }

    // Full scales round-trip: recover 6-bit (sc, m) per super-block
    // sub-block, forward-pack per llama's quantize layout, compare bytes.
    let ntc = (cols / 32) as usize;
    let c256 = (cols / 256) as usize;
    let mut sb_sc = vec![[0u8; 8]; n];
    let mut sb_m = vec![[0u8; 8]; n];
    for t in 0..ntiles {
        let (tr, tc) = (t / ntc, t % ntc);
        for i in 0..32 {
            let r = tr * 32 + i;
            let q = 32 * tc / 256;
            let s = (32 * tc % 256) / 32;
            let sb = r * c256 + q;
            let sc = sv[t * 32 + i].to_f32().round() as u8;
            let m = (mv[t * 32 + i].to_f32() * 2.0).round() as u8;
            assert!(sc < 64 && m < 64, "tile {t} row {i}: sc={sc} m={m}");
            sb_sc[sb][s] = sc;
            sb_m[sb][s] = m;
        }
    }
    for sb in 0..n {
        let mut s = [0u8; 12];
        for j in 0..4 {
            s[j] = sb_sc[sb][j];
            s[j + 4] = sb_m[sb][j];
        }
        for j in 4..8 {
            s[j + 4] = (sb_sc[sb][j] & 15) | ((sb_m[sb][j] & 15) << 4);
            s[j - 4] |= (sb_sc[sb][j] >> 4) << 6;
            s[j] |= (sb_m[sb][j] >> 4) << 6;
        }
        assert_eq!(&raw[sb * 144 + 4..sb * 144 + 16], &s, "super-block {sb} scales");
    }
    Ok(())
}
