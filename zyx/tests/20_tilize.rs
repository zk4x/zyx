// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `tilize`/`untilize` roundtrips through the public tensor API.

use zyx::{Tensor, ZyxError, f16};

#[test]
fn tilize_roundtrip_odd_shape() -> Result<(), ZyxError> {
    // Odd, non-tile-multiple shape exercises padding + face order.
    let rows = 37i64;
    let cols = 45i64;
    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.5 - 100.0).collect();
    let t = Tensor::from_vec(data.clone(), [rows, cols])?;
    let til = Tensor::tilize(&t)?;
    let shape: Vec<i64> = til.shape().iter().map(|d| d.item::<i64>()).collect();
    assert_eq!(shape, vec![64, 64]);
    // First tile, first face must hold rows 0..16, cols 0..16 in order.
    let flat: Vec<f32> = til.to_vec()?;
    for lr in 0..16 {
        for lc in 0..16 {
            assert_eq!(flat[(lr * 16 + lc) as usize], (lr * cols + lc) as f32 * 0.5 - 100.0);
        }
    }
    // Padded region reads back as zero.
    assert_eq!(flat[((63 * 64 + 63) % (64 * 64)) as usize], 0.0);
    let back = Tensor::untilize(&til, rows, cols)?;
    let shape: Vec<i64> = back.shape().iter().map(|d| d.item::<i64>()).collect();
    assert_eq!(shape, vec![rows, cols]);
    let rt: Vec<f32> = back.to_vec()?;
    assert_eq!(rt, data);
    Ok(())
}

#[test]
fn tilize_matches_golden_face_order() -> Result<(), ZyxError> {
    // Conformance with the `14_tenstorrent` golden kernel's `lin()`
    // mapping: face slot -> linear index within a tile.
    let lin = |s: usize| {
        let (face, local) = (s / 256, s % 256);
        let (fr0, fc0) = (face / 2, face % 2);
        fr0 * 16 * 32 + fc0 * 16 + (local / 16) * 32 + local % 16
    };
    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let t = Tensor::from_vec(data, [32, 32])?;
    let til = Tensor::tilize(&t)?;
    let flat: Vec<f32> = til.to_vec()?;
    for p in 0..1024 {
        assert_eq!(flat[p], lin(p) as f32, "face slot {p}");
    }
    Ok(())
}

#[test]
fn tilize_batched_f16() -> Result<(), ZyxError> {
    let data: Vec<f16> = (0..2 * 6 * 40).map(|i| f16::from_f32(i as f32)).collect();
    let t = Tensor::from_vec(data.clone(), [2, 6, 40])?;
    let til = Tensor::tilize(&t)?;
    let shape: Vec<i64> = til.shape().iter().map(|d| d.item::<i64>()).collect();
    assert_eq!(shape, vec![2, 32, 64]);
    let back = Tensor::untilize(&til, 6, 40)?;
    let rt: Vec<f16> = back.to_vec()?;
    assert_eq!(rt, data);
    Ok(())
}
