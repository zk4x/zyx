// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Real shapes that reach a tensor core, through the public `Tensor` API: a
//! batched matmul, a convolution (`Tensor::conv`) and RoPE (`Tensor::rope`).
//!
//! Each test builds the tensors on the host, moves them to a tensor-core
//! device, and compares the realized result against a host reference on the
//! same (f16-rounded) operands. The whole pipeline runs — linearize, split,
//! tensorize, codegen — so a matcher that only knows the 2-D, two-range
//! `x @ W.t()` shape shows up here as wrong values or as a panic, not as a
//! missing optimization.

use zyx::{DType, Tensor, ZyxError, kernel::Dev};

/// The tensor-core device, or `None` where there is none (C and Host have no
/// instructions to tile to).
fn tensor_core_dev() -> Option<Dev> {
    Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host))
}

/// f16 operands, the way the pipeline rounds them: the reference multiplies
/// exactly the values the kernel reads.
fn f16_round(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&x| zyx::f16::from_f32(x).to_f32()).collect()
}

fn host_tensor(values: &[f32], shape: impl IntoIterator<Item = i64>) -> Result<Tensor, ZyxError> {
    let data: Vec<zyx::f16> = values.iter().map(|&x| zyx::f16::from_f32(x)).collect();
    Tensor::from_vec(data, shape)
}

/// `x[b, m, k] @ w[b, n, k].T -> out[b, m, n]`: three output ranges (batch, m,
/// n) over the one contraction, which is what a batched projection is. The
/// batch axis is in every view, so it is neither m nor n — the matcher has to
/// carry it as a fixed term.
#[test]
fn batched_matmul_values() -> Result<(), ZyxError> {
    const B: i64 = 2;
    const M: i64 = 32;
    const N: i64 = 64;
    const K: i64 = 32;

    let Some(dev) = tensor_core_dev() else {
        return Ok(());
    };
    let x: Vec<f32> = (0..(B * M * K) as usize).map(|i| ((i as u64 * 2654435761) % 1000) as f32 / 1000.0 - 0.5).collect();
    let w: Vec<f32> = (0..(B * N * K) as usize).map(|i| ((i as u64 * 40503 + 7) % 1000) as f32 / 1000.0 - 0.5).collect();
    let xh = f16_round(&x);
    let wh = f16_round(&w);

    let a = host_tensor(&xh, [B, M, K])?.to(dev)?;
    let b = host_tensor(&wh, [B, N, K])?.to(dev)?;
    let out: Vec<f32> = a.matmul(b.t())?.cast(DType::F32).try_into()?;

    let mut worst = 0.0f32;
    for bi in 0..B as usize {
        for i in 0..M as usize {
            for j in 0..N as usize {
                let mut sum = 0.0f32;
                for t in 0..K as usize {
                    sum += xh[bi * (M * K) as usize + i * K as usize + t] * wh[bi * (N * K) as usize + j * K as usize + t];
                }
                let got = out[bi * (M * N) as usize + i * N as usize + j];
                let tol = 1e-2f32.max(2e-3 * sum.abs());
                let err = (got - sum).abs();
                assert!(
                    err <= tol,
                    "batched matmul disagrees with the host at (b={bi}, m={i}, n={j}): {got} vs {sum}, tol {tol}"
                );
                worst = worst.max(err);
            }
        }
    }
    println!("batched matmul {B}x{M}x{K}x{N} matches the host, max |err| = {worst:.3e}");
    Ok(())
}

/// `Tensor::conv`: zero-pad, slide the kernel, broadcast-multiply and reduce
/// over (cin, kh, kw). The output has four ranges (b, cout, oh, ow) and the
/// contraction is three axes, so this is only tensorizable as a contraction
/// whose output axes are carried one-per-operand and everything else is a
/// fixed index term.
#[test]
fn conv_values() -> Result<(), ZyxError> {
    const B: i64 = 1;
    const CIN: i64 = 4;
    const COUT: i64 = 8;
    const H: i64 = 9;
    const W: i64 = 9;
    const KH: i64 = 2;
    const KW: i64 = 2;

    let Some(dev) = tensor_core_dev() else {
        return Ok(());
    };
    let x: Vec<f32> = (0..(B * CIN * H * W) as usize).map(|i| ((i as u64 * 2654435761) % 1000) as f32 / 1000.0 - 0.5).collect();
    let w: Vec<f32> =
        (0..(COUT * CIN * KH * KW) as usize).map(|i| ((i as u64 * 40503 + 7) % 1000) as f32 / 1000.0 - 0.5).collect();
    let xh = f16_round(&x);
    let wh = f16_round(&w);

    let a = host_tensor(&xh, [B, CIN, H, W])?.to(dev)?;
    let weight =
        Tensor::from_vec(wh.iter().map(|&x| zyx::f16::from_f32(x)).collect::<Vec<zyx::f16>>(), [COUT, CIN, KH, KW])?.to(dev)?;
    let out: Vec<f32> = a.conv(&weight, None, 1, [1, 1], [1, 1], [0, 0])?.cast(DType::F32).try_into()?;

    let (oh, ow) = (H - KH + 1, W - KW + 1);
    let mut worst = 0.0f32;
    for bi in 0..B as usize {
        for co in 0..COUT as usize {
            for y in 0..oh as usize {
                for x0 in 0..ow as usize {
                    let mut sum = 0.0f32;
                    for ci in 0..CIN as usize {
                        for ky in 0..KH as usize {
                            for kx in 0..KW as usize {
                                let xv = xh[((bi * CIN as usize + ci) * H as usize + y + ky) * W as usize + x0 + kx];
                                let wv = wh[((co * CIN as usize + ci) * KH as usize + ky) * KW as usize + kx];
                                sum += xv * wv;
                            }
                        }
                    }
                    let got = out[((bi * COUT as usize + co) * oh as usize + y) * ow as usize + x0];
                    let tol = 1e-2f32.max(2e-3 * sum.abs());
                    let err = (got - sum).abs();
                    assert!(
                        err <= tol,
                        "conv disagrees with the host at (b={bi}, co={co}, y={y}, x={x0}): {got} vs {sum}, tol {tol}"
                    );
                    worst = worst.max(err);
                }
            }
        }
    }
    println!("conv {B}x{CIN}x{H}x{W} x {COUT}x{CIN}x{KH}x{KW} matches the host, max |err| = {worst:.3e}");
    Ok(())
}

/// `Tensor::rope`: `out[..., :half] = x[:half]*cos - x[half:]*sin`,
/// `out[..., half:] = x[:half]*sin + x[half:]*cos`. No contraction, so no
/// tensor core is involved — the point of the test is that a real op built
/// out of narrow/cat/mul/add still runs with the whole optimization list
/// (including tensorize) applied, and matches the reference.
#[test]
fn rope_values() -> Result<(), ZyxError> {
    const S: i64 = 8;
    const E: i64 = 8;

    let Some(dev) = tensor_core_dev() else {
        return Ok(());
    };
    let x: Vec<f32> = (0..(2 * S * E) as usize).map(|i| ((i as u64 * 2654435761) % 1000) as f32 / 1000.0 - 0.5).collect();
    let cos: Vec<f32> = (0..(S * E / 2) as usize).map(|i| ((i as u64 * 40503 + 3) % 100) as f32 / 100.0 - 0.5).collect();
    let sin: Vec<f32> = (0..(S * E / 2) as usize).map(|i| ((i as u64 * 7919 + 11) % 100) as f32 / 100.0 - 0.5).collect();

    let a = Tensor::from_vec(x.clone(), [2, S, E])?.to(dev)?;
    let c = Tensor::from_vec(cos.clone(), [S, E / 2])?.to(dev)?;
    let s = Tensor::from_vec(sin.clone(), [S, E / 2])?.to(dev)?;
    let out: Vec<f32> = a.rope(c, s)?.try_into()?;

    let mut worst = 0.0f32;
    for b in 0..2usize {
        for t in 0..S as usize {
            for j in 0..(E / 2) as usize {
                let lo = x[(b * S as usize + t) * E as usize + j];
                let hi = x[(b * S as usize + t) * E as usize + j + (E / 2) as usize];
                let freq = t * (E / 2) as usize + j;
                let want_lo = lo * cos[freq] - hi * sin[freq];
                let want_hi = lo * sin[freq] + hi * cos[freq];
                let got_lo = out[(b * S as usize + t) * E as usize + j];
                let got_hi = out[(b * S as usize + t) * E as usize + j + (E / 2) as usize];
                let tol = 1e-3f32.max(2e-3 * want_lo.abs());
                assert!((got_lo - want_lo).abs() <= tol, "rope (b={b}, s={t}, j={j}) low half: {got_lo} vs {want_lo}");
                assert!((got_hi - want_hi).abs() <= tol, "rope (b={b}, s={t}, j={j}) high half: {got_hi} vs {want_hi}");
                worst = worst.max((got_lo - want_lo).abs()).max((got_hi - want_hi).abs());
            }
        }
    }
    println!("rope [2x{S}x{E}] matches the host, max |err| = {worst:.3e}");
    Ok(())
}
