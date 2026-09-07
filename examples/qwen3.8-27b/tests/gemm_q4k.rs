// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Q4_K fused gemm reference vs pytorch

use zyx::kernel::Dev;
use zyx::{DType, Tensor, ZyxError};

fn quant_q4_0(b: &[f32], k: usize, n: usize) -> (Vec<u32>, Vec<zyx::f16>) {
    // 8 per U32, 4 U32 per 32 block
    let mut qs = vec![0u32; n * k / 8];
    let mut scales = Vec::with_capacity(n * k / 32);
    for row in 0..n {
        for blk in 0..k / 32 {
            let base = row * k + blk * 32;
            let mut max = 0f32;
            for i in 0..32 {
                max = max.max(b[base + i].abs());
            }
            let d = if max == 0.0 { 1.0 } else { max / 7.0 };
            scales.push(zyx::f16::from_f32(d));
            for i in 0..32 {
                let v = b[base + i];
                let q = ((v / d).round() + 8.0).clamp(0.0, 15.0) as u32;
                let u32_idx = base / 8 + i / 8;
                let shift = (i % 8) * 4;
                qs[u32_idx] |= (q & 0xF) << shift;
            }
        }
    }
    (qs, scales)
}

#[test]
fn test_u8_roundtrip() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let qs = vec![255u8, 0, 1, 0x12, 0xAB];
    let t = Tensor::from(qs.clone()).to(dev)?;
    let v: Vec<u8> = t.to_vec()?;
    eprintln!("u8 roundtrip {:?} -> {:?}", qs, v);
    assert_eq!(qs, v);
    Ok(())
}

#[test]
fn test_u8_load_cast() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let qs = vec![255u8, 0, 1];
    let mut k = zyx::kernel::Kernel::new(dev);
    let qs_p = k.param(DType::U8);
    let out = k.param_mut(DType::F32);
    let idx0 = k.const_idx(0);
    let v = k.load(qs_p, idx0);
    let vf = k.cast(v, DType::F32);
    k.store(out, vf, idx0);
    let idx1 = k.const_idx(1);
    let v1 = k.load(qs_p, idx1);
    let vf1 = k.cast(v1, DType::F32);
    k.store(out, vf1, idx1);
    k.default_epilogue();
    let ck = k.compile()?;
    let qs_t = Tensor::from(qs.clone()).to(dev)?;
    let outs = ck.forward(&[&qs_t], vec![vec![Tensor::from(2i64)]])?;
    let out_vec: Vec<f32> = outs[0].to_vec()?;
    eprintln!("u8 load cast {:?} -> {:?}", qs, out_vec);
    assert_eq!(out_vec[0] as u8, 255);
    Ok(())
}

#[test]
fn dequant_q4k_simple() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let k = 32;
    let n = 8;
    let b_host: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.02).cos() * 0.5).collect();
    let (qs, scales) = quant_q4_0(&b_host, k, n);
    eprintln!(
        "scales {:?}",
        scales.iter().map(|x| x.to_f32()).collect::<Vec<_>>()
    );
    eprintln!("qs first 2 {:08x?}", &qs[0..2]);
    eprintln!("b_host first 8 {:?}", &b_host[0..8]);
    let mut b_deq = vec![0f32; n * k];
    for row in 0..n {
        for col in 0..k {
            let blk = col / 32;
            let d = scales[row * k / 32 + blk].to_f32();
            let u32_idx = (row * k + col) / 8;
            let shift = (col % 8) * 4;
            let q = ((qs[u32_idx] >> shift) & 0xF) as i32;
            b_deq[row * k + col] = (q - 8) as f32 * d;
        }
    }
    eprintln!("b_deq first 8 {:?}", &b_deq[0..8]);
    let mut kernel = zyx::kernel::Kernel::new(dev);
    let qs_p = kernel.param(DType::U32);
    let sc_p = kernel.param(DType::F16);
    let out = kernel.param_mut(DType::F32);
    let n_k = (n * k) as i64;
    kernel.loop_over(n_k, |kernel, idx| {
        let v = kernel.dequant_q4_k(qs_p, sc_p, idx);
        let vf = kernel.cast(v, DType::F32);
        kernel.store(out, vf, idx);
    });
    kernel.default_epilogue();
    let ck = kernel.compile()?;
    let qs_t = Tensor::from(qs.clone()).to(dev)?;
    let sc_t = Tensor::from(scales.clone()).to(dev)?;
    let outs = ck.forward(&[&qs_t, &sc_t], vec![vec![Tensor::from(n_k)]])?;
    let out_vec: Vec<f32> = outs[0].to_vec()?;
    for (a, b) in out_vec.iter().zip(b_deq.iter()) {
        let d = (a - b).abs();
        assert!(d < 1e-3, "d {d} {a} {b}");
    }
    Ok(())
}

#[test]
fn gemm_q4k_small() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let r = 16;
    let k = 32;
    let n = 8;
    let a_host: Vec<zyx::f16> = (0..r * k)
        .map(|i| zyx::f16::from_f32((i as f32 * 0.01).sin()))
        .collect();
    let b_host: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.02).cos() * 0.5).collect();
    let (qs, scales) = quant_q4_0(&b_host, k, n);
    let mut b_deq = vec![0f32; n * k];
    for row in 0..n {
        for col in 0..k {
            let blk = col / 32;
            let d = scales[row * k / 32 + blk].to_f32();
            let u32_idx = (row * k + col) / 8;
            let shift = (col % 8) * 4;
            let q = ((qs[u32_idx] >> shift) & 0xF) as i32;
            b_deq[row * k + col] = (q - 8) as f32 * d;
        }
    }
    let mut c_ref = vec![0f32; r * n];
    for i in 0..r {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += a_host[i * k + kk].to_f32() * b_deq[j * k + kk];
            }
            c_ref[i * n + j] = acc;
        }
    }
    let a_t = Tensor::from(a_host.clone()).to(dev)?;
    let qs_t = Tensor::from(qs.clone()).to(dev)?;
    let sc_t = Tensor::from(scales.clone()).to(dev)?;
    let k = qwen3_8_27b::gemm_cuda_q4_k(r as i64, k as i64, n as i64);
    let ck = k.compile()?;
    let out = ck.forward(
        &[&a_t, &qs_t, &sc_t],
        vec![vec![Tensor::from(r as i64), Tensor::from(n as i64)]],
    )?;
    let out_vec: Vec<f32> = out[0].to_vec()?;
    for (a, b) in out_vec.iter().zip(c_ref.iter()) {
        let diff = (a - b).abs();
        assert!(diff < 5e-2, "diff {diff} a {a} b {b}");
    }
    Ok(())
}

#[test]
fn gemm_q4k_exact_small() -> Result<(), ZyxError> {
    // Exact path: q*scale-min per 32, k=256 to cover a full Q4_K super-block.
    let dev = Dev::Cuda(0);
    let r = 16;
    let k = 256;
    let n = 8;
    let a_host: Vec<zyx::f16> = (0..r * k)
        .map(|i| zyx::f16::from_f32((i as f32 * 0.01).sin()))
        .collect();
    let b_host: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.02).cos() * 0.5).collect();
    let mut qs = vec![0u32; n * k / 8];
    let mut scales = Vec::with_capacity(n * k / 32);
    let mut mins = Vec::with_capacity(n * k / 32);
    for row in 0..n {
        for blk in 0..k / 32 {
            let base = row * k + blk * 32;
            let scale = 0.03 + 0.005 * ((row + blk) % 3) as f32;
            let min = 0.2 + 0.05 * ((row * 2 + blk) % 2) as f32;
            scales.push(zyx::f16::from_f32(scale));
            mins.push(zyx::f16::from_f32(min));
            for i in 0..32 {
                let q = ((b_host[base + i] + min) / scale).round().clamp(0.0, 15.0) as u32;
                let u32_idx = base / 8 + i / 8;
                qs[u32_idx] |= (q & 0xF) << ((i % 8) * 4);
            }
        }
    }
    let mut b_deq = vec![0f32; n * k];
    for row in 0..n {
        for col in 0..k {
            let blk = col / 32;
            let sc = scales[row * k / 32 + blk].to_f32();
            let mn = mins[row * k / 32 + blk].to_f32();
            let u32_idx = (row * k + col) / 8;
            let q = ((qs[u32_idx] >> ((col % 8) * 4)) & 0xF) as f32;
            b_deq[row * k + col] = q * sc - mn;
        }
    }
    let mut c_ref = vec![0f32; r * n];
    for i in 0..r {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += a_host[i * k + kk].to_f32() * b_deq[j * k + kk];
            }
            c_ref[i * n + j] = acc;
        }
    }
    let a_t = Tensor::from(a_host.clone()).to(dev)?;
    let qs_t = Tensor::from(qs.clone()).to(dev)?;
    let sc_t = Tensor::from(scales.clone()).to(dev)?;
    let mn_t = Tensor::from(mins.clone()).to(dev)?;
    let kk = qwen3_8_27b::gemm_cuda_q4_k_exact(r as i64, k as i64, n as i64);
    let ck = kk.compile()?;
    let out = ck.forward(
        &[&a_t, &qs_t, &sc_t, &mn_t],
        vec![vec![Tensor::from(r as i64), Tensor::from(n as i64)]],
    )?;
    for o in &out {
        o.sync()?;
    }
    let out_vec: Vec<f32> = out[0].to_vec()?;
    for (a, b) in out_vec.iter().zip(c_ref.iter()) {
        let diff = (a - b).abs();
        assert!(diff < 8e-2, "diff {diff} a {a} b {b}");
    }
    Ok(())
}
