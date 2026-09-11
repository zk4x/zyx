// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `pad_kernel(s, m, d)`: pad [s, d] F32 input to [m, d] F16, zero-fill rows s..m.
//! Two uses in qwen3.8-27b: (s=S=6, m=M_PAD=16, d=HIDDEN=5120) for input;
//! (s=S=6, m=M_PAD=16, d=VAL_DIM=6144) for normed.

use qwen3_8_27b::{
    pad_cast_tt, pad_copy_tt, pad_kernel, pad_mul_tt, HIDDEN, M_PAD, S, VAL_DIM,
};
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

#[test]
fn pad_input() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_pad_input.safetensors")?;
    let input = goldens["input"].to(dev)?;
    let expected = &goldens["output"];
    let kk = pad_kernel(S, M_PAD, HIDDEN);
    let (flops, read, write) = kk.flop_mem_rw();
    let k = kk.compile()?;
    let t0 = std::time::Instant::now();
    let out = k.forward(&[&input], vec![[M_PAD, HIDDEN]])?;
    out[0].sync()?;
    let total_us = t0.elapsed().as_micros() as f64;
    let tflops = if total_us > 0.0 {
        flops as f64 / total_us / 1e3
    } else {
        0.0
    };
    let gbs = if total_us > 0.0 {
        (read + write) as f64 / total_us / 1e3
    } else {
        0.0
    };
    eprintln!("pad_input forward+sync {total_us:.0}us, {tflops:.2} TFLOPS, {gbs:.1} GB/s");
    let v: Vec<zyx::f16> = out[0].to_vec()?;
    let v: Vec<f32> = v.iter().map(|&x| x.to_f32()).collect();
    let exp: Vec<zyx::f16> = expected.to_vec()?;
    let exp: Vec<f32> = exp.iter().map(|&x| x.to_f32()).collect();
    assert_eq!(v.len(), exp.len());
    for (i, (&a, &b)) in v.iter().zip(exp.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "pad_input[{i}] = {a}, expected {b}");
    }
    Ok(())
}

#[test]
fn pad_normed() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_pad_normed.safetensors")?;
    let input = goldens["input"].to(dev)?;
    let expected = &goldens["output"];
    let kk = pad_kernel(S, M_PAD, VAL_DIM);
    let (flops, read, write) = kk.flop_mem_rw();
    let k = kk.compile()?;
    let t0 = std::time::Instant::now();
    let out = k.forward(&[&input], vec![[M_PAD, VAL_DIM]])?;
    out[0].sync()?;
    let total_us = t0.elapsed().as_micros() as f64;
    let tflops = if total_us > 0.0 {
        flops as f64 / total_us / 1e3
    } else {
        0.0
    };
    let gbs = if total_us > 0.0 {
        (read + write) as f64 / total_us / 1e3
    } else {
        0.0
    };
    eprintln!("pad_normed forward+sync {total_us:.0}us, {tflops:.2} TFLOPS, {gbs:.1} GB/s");
    let v: Vec<zyx::f16> = out[0].to_vec()?;
    let v: Vec<f32> = v.iter().map(|&x| x.to_f32()).collect();
    let exp: Vec<zyx::f16> = expected.to_vec()?;
    let exp: Vec<f32> = exp.iter().map(|&x| x.to_f32()).collect();
    assert_eq!(v.len(), exp.len());
    for (i, (&a, &b)) in v.iter().zip(exp.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "pad_normed[{i}] = {a}, expected {b}");
    }
    Ok(())
}

/// `pad_kernel_tt` on the P100A: F32 tilized padded input + F32 tilized
/// mask -> F16 tilized out, chunked launches (64 tiles each, one tile per
/// group). Output chunks are concatenated in tile order, untilized, and
/// compared against the same torch golden as `pad_input`.
#[test]
fn pad_input_tt() -> Result<(), ZyxError> {
    let dev = Dev::TT(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_pad_input.safetensors")?;
    let input: Vec<f32> = goldens["input"].to_vec()?;
    let expected = &goldens["output"];
    // Host-pad rows s..m with zeros (tilize pads the 16 rows to 32).
    let mut padded = vec![0.0f32; (M_PAD * HIDDEN) as usize];
    for r in 0..S as usize {
        padded[r * HIDDEN as usize..(r + 1) * HIDDEN as usize]
            .copy_from_slice(&input[r * HIDDEN as usize..(r + 1) * HIDDEN as usize]);
    }
    // Mask: 1.0 on kept rows, 0.0 on padded rows.
    let mut mask = vec![0.0f32; (M_PAD * HIDDEN) as usize];
    for r in 0..S as usize {
        for c in 0..HIDDEN as usize {
            mask[r * HIDDEN as usize + c] = 1.0;
        }
    }
    let data_t = Tensor::tilize(&Tensor::from_vec(padded, [M_PAD, HIDDEN])?)?.to(dev)?;
    let mask_t = Tensor::tilize(&Tensor::from_vec(mask, [M_PAD, HIDDEN])?)?.to(dev)?;

    // Split (A): F32 mask-mul kernel (32-bit DST) then standalone F32->F16
    // cast kernel (16-bit DST). Fused mixed-format SFPU is off the tt-metal
    // supported path (mode-unaware typecast addressing).
    let kk1 = pad_mul_tt(S, M_PAD, HIDDEN);
    let kk2 = pad_cast_tt(S, M_PAD, HIDDEN);
    let (flops1, read1, write1) = kk1.flop_mem_rw();
    let (flops2, read2, write2) = kk2.flop_mem_rw();
    let (flops, read, write) = (flops1 + flops2, read1 + read2, write1 + write2);
    let k1 = kk1.compile()?;
    let k2 = kk2.compile()?;
    // Tilized [32, 5120] = 160 tiles in a single launch.
    const TILES: i64 = 160;
    let t0 = std::time::Instant::now();
    let mid = k1.forward(&[&data_t, &mask_t], vec![[TILES * 1024]])?;
    let out = k2.forward(&[&mid[0]], vec![[TILES * 1024]])?;
    out[0].sync()?;
    let chunks: Vec<zyx::f16> = out[0].to_vec()?;
    let total_us = t0.elapsed().as_micros() as f64;
    let tflops = if total_us > 0.0 {
        flops as f64 / total_us / 1e3
    } else {
        0.0
    };
    let gbs = if total_us > 0.0 {
        (read + write) as f64 / total_us / 1e3
    } else {
        0.0
    };
    eprintln!("pad_input_tt forward+sync {total_us:.0}us, {tflops:.2} TFLOPS, {gbs:.1} GB/s");
    let til = Tensor::from_vec(chunks, [32, HIDDEN])?;
    let back = Tensor::untilize(&til, M_PAD, HIDDEN)?;
    let v: Vec<zyx::f16> = back.to_vec()?;
    let exp: Vec<zyx::f16> = expected.to_vec()?;
    assert_eq!(v.len(), exp.len());
    let mut bad = 0;
    for (&a, &b) in v.iter().zip(exp.iter()) {
        // FTZ carve-out: the device flushes F16 subnormals to zero;
        // accept ±0.0 for subnormal-range goldens.
        let be = b.to_bits();
        if be & 0x7c00 == 0 && be & 0x03ff != 0 {
            let ga = a.to_bits();
            if ga == 0x0000 || ga == 0x8000 {
                continue;
            }
        }
        // Device truncates F32->F16b on unpack while the host rounds:
        // allow exactly 1 ulp (±0.0 equal).
        if a.to_f32() == b.to_f32() {
            continue;
        }
        if (a.to_bits() as i32 - b.to_bits() as i32).abs() > 1 {
            bad += 1;
        }
    }
    eprintln!("pad_input_tt bad: {bad} / {}", v.len());
    assert_eq!(bad, 0);
    Ok(())
}

/// Bring-up diagnostic: F32 tilized input -> F16 tilized out, no mask, no
/// binary op. Passes iff reader/tilize/cast/pack/writer are all correct;
/// isolates binary-op unpack configuration as the suspect when the full
/// `pad_kernel_tt` returns wrong values.
#[test]
fn pad_passthrough_tt_run() -> Result<(), ZyxError> {
    let dev = Dev::TT(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_pad_input.safetensors")?;
    let input: Vec<f32> = goldens["input"].to_vec()?;
    let mut padded = vec![0.0f32; (M_PAD * HIDDEN) as usize];
    for r in 0..S as usize {
        padded[r * HIDDEN as usize..(r + 1) * HIDDEN as usize]
            .copy_from_slice(&input[r * HIDDEN as usize..(r + 1) * HIDDEN as usize]);
    }
    let data_t = Tensor::tilize(&Tensor::from_vec(padded.clone(), [M_PAD, HIDDEN])?)?.to(dev)?;

    // Split (A): F32 copy kernel (32-bit DST) then standalone F32->F16
    // cast kernel (16-bit DST).
    let kk1 = pad_copy_tt(S, M_PAD, HIDDEN);
    let kk2 = pad_cast_tt(S, M_PAD, HIDDEN);
    let k1 = kk1.compile()?;
    let k2 = kk2.compile()?;
    const TILES: i64 = 160;
    let mid = k1.forward(&[&data_t], vec![[TILES * 1024]])?;
    let out = k2.forward(&[&mid[0]], vec![[TILES * 1024]])?;
    out[0].sync()?;
    let chunks: Vec<zyx::f16> = out[0].to_vec()?;
    let til = Tensor::from_vec(chunks, [32, HIDDEN])?;
    let back = Tensor::untilize(&til, M_PAD, HIDDEN)?;
    let v: Vec<zyx::f16> = back.to_vec()?;
    // Expected: padded input cast to F16 (host rounds; the device
    // truncates F32->F16b on unpack, so allow exactly 1 ulp).
    let exp: Vec<zyx::f16> = padded.iter().map(|&x| zyx::f16::from_f32(x)).collect();
    assert_eq!(v.len(), exp.len());
    let mut bad = 0;
    for (&a, &b) in v.iter().zip(exp.iter()) {
        // FTZ carve-out: the device flushes F16 subnormals to zero;
        // accept ±0.0 for subnormal-range goldens.
        let be = b.to_bits();
        if be & 0x7c00 == 0 && be & 0x03ff != 0 {
            let ga = a.to_bits();
            if ga == 0x0000 || ga == 0x8000 {
                continue;
            }
        }
        if a.to_f32() == b.to_f32() {
            continue;
        }
        if (a.to_bits() as i32 - b.to_bits() as i32).abs() > 1 {
            bad += 1;
        }
    }
    eprintln!("passthrough bad: {bad} / {}", v.len());
    assert_eq!(bad, 0);
    Ok(())
}

/// Bring-up diagnostic: pure F32 movement, empty compute section. Passes
/// iff reader/writer/CB/DRAM paths are correct; isolates compute
/// math/pack as the suspect when values come back wrong.
#[test]
fn pad_move_tt_run() -> Result<(), ZyxError> {
    use qwen3_8_27b::pad_move_tt;
    let dev = Dev::TT(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_pad_input.safetensors")?;
    let input: Vec<f32> = goldens["input"].to_vec()?;
    let mut padded = vec![0.0f32; (M_PAD * HIDDEN) as usize];
    for r in 0..S as usize {
        padded[r * HIDDEN as usize..(r + 1) * HIDDEN as usize]
            .copy_from_slice(&input[r * HIDDEN as usize..(r + 1) * HIDDEN as usize]);
    }
    let data_t = Tensor::tilize(&Tensor::from_vec(padded.clone(), [M_PAD, HIDDEN])?)?.to(dev)?;

    let kk = pad_move_tt(S, M_PAD, HIDDEN);
    let k = kk.compile()?;
    const TILES: i64 = 160;
    let out = k.forward(&[&data_t], vec![[TILES * 1024]])?;
    out[0].sync()?;
    let moved: Vec<f32> = out[0].to_vec()?;
    let til = Tensor::from_vec(moved, [32, HIDDEN])?;
    let back = Tensor::untilize(&til, M_PAD, HIDDEN)?;
    let v: Vec<f32> = back.to_vec()?;
    let mut bad = 0;
    for (i, (&a, &b)) in v.iter().zip(padded.iter()).enumerate() {
        if (a - b).abs() > 1e-6 {
            if bad < 10 {
                eprintln!("move[{i}] = {a}, expected {b}");
            }
            bad += 1;
        }
    }
    eprintln!("move bad: {bad} / {}", v.len());
    assert_eq!(bad, 0);
    Ok(())
}

/// Bring-up diagnostic: F32 copy in compute (copy_tile + pack, no cast).
/// Passes iff copy/pack are correct; isolates the F32->F16 typecast when
/// values come back wrong.
#[test]
fn pad_copy_tt_run() -> Result<(), ZyxError> {
    let dev = Dev::TT(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_pad_input.safetensors")?;
    let input: Vec<f32> = goldens["input"].to_vec()?;
    let mut padded = vec![0.0f32; (M_PAD * HIDDEN) as usize];
    for r in 0..S as usize {
        padded[r * HIDDEN as usize..(r + 1) * HIDDEN as usize]
            .copy_from_slice(&input[r * HIDDEN as usize..(r + 1) * HIDDEN as usize]);
    }
    let data_t = Tensor::tilize(&Tensor::from_vec(padded.clone(), [M_PAD, HIDDEN])?)?.to(dev)?;

    let kk = pad_copy_tt(S, M_PAD, HIDDEN);
    let k = kk.compile()?;
    const TILES: i64 = 160;
    let out = k.forward(&[&data_t], vec![[TILES * 1024]])?;
    out[0].sync()?;
    let moved: Vec<f32> = out[0].to_vec()?;
    let til = Tensor::from_vec(moved, [32, HIDDEN])?;
    let back = Tensor::untilize(&til, M_PAD, HIDDEN)?;
    let v: Vec<f32> = back.to_vec()?;
    let mut bad = 0;
    for (i, (&a, &b)) in v.iter().zip(padded.iter()).enumerate() {
        if (a - b).abs() > 1e-6 {
            if bad < 10 {
                eprintln!("copy[{i}] = {a}, expected {b}");
            }
            bad += 1;
        }
    }
    eprintln!("copy bad: {bad} / {}", v.len());
    assert_eq!(bad, 0);
    Ok(())
}
