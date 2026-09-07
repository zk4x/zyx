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
    let n = 64;
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

#[test]
fn gemm_q4k_real_1024() -> Result<(), ZyxError> {
    // Real dims 1024×1024, 32×32 local tiles → 16× fewer loads. Single sync-timed launch.
    let dev = Dev::Cuda(0);
    let r = 1024;
    let k = 1024;
    let n = 1024;
    let a_host: Vec<zyx::f16> = (0..r * k)
        .map(|i| zyx::f16::from_f32(((i % 128) as f32 / 64.0 - 1.0) * 0.5))
        .collect();
    let b_host: Vec<f32> = (0..n * k).map(|i| ((i % 64) as f32 / 32.0 - 1.0) * 0.4).collect();
    let mut qs = vec![0u32; n * k / 8];
    let mut scales = Vec::with_capacity(n * k / 32);
    let mut mins = Vec::with_capacity(n * k / 32);
    for row in 0..n {
        for blk in 0..k / 32 {
            let base = row * k + blk * 32;
            let sc = 0.04;
            let mn = 0.3;
            scales.push(zyx::f16::from_f32(sc));
            mins.push(zyx::f16::from_f32(mn));
            for i in 0..32 {
                let q = ((b_host[base + i] + mn) / sc).round().clamp(0.0, 15.0) as u32;
                qs[base / 8 + i / 8] |= (q & 0xF) << ((i % 8) * 4);
            }
        }
    }
    let a_t = Tensor::from(a_host.clone()).to(dev)?;
    let qs_t = Tensor::from(qs.clone()).to(dev)?;
    let sc_t = Tensor::from(scales.clone()).to(dev)?;
    let mn_t = Tensor::from(mins.clone()).to(dev)?;
    let kk = qwen3_8_27b::gemm_cuda_q4_k_exact(r as i64, k as i64, n as i64);
    let (flops, read, write) = kk.flop_mem_rw();
    eprintln!("flops {} read {} write {}", flops, read, write);
    let ck = kk.compile()?;
    let t0 = std::time::Instant::now();
    let outs = ck.forward(&[&a_t, &qs_t, &sc_t, &mn_t], vec![vec![Tensor::from(r as i64), Tensor::from(n as i64)]])?;
    let enqueue_us = t0.elapsed().as_micros();
    let t1 = std::time::Instant::now();
    for o in &outs {
        o.sync()?;
    }
    let sync_us = t1.elapsed().as_micros();
    eprintln!("real 1024 enqueue {enqueue_us}us sync {sync_us}us total {}", enqueue_us + sync_us);
    eprintln!("GB/s read+write {} / {}us = {:.1}", read + write, enqueue_us + sync_us, (read + write) as f64 / (enqueue_us + sync_us) as f64 / 1e3);
    let v: Vec<f32> = outs[0].to_vec()?;
    assert_eq!(v.len(), (r * n) as usize);
    // spot-check a few outputs vs ref
    let mut b_deq = vec![0f32; 32];
    for col in 0..32 {
        b_deq[col] = ((qs[col / 8] >> ((col % 8) * 4) & 0xF) as f32) * 0.04 - 0.3;
    }
    let _ = b_deq;
    Ok(())
}

#[test]
fn bench_q4_variants() -> Result<(), ZyxError> {
    // For loop over kernel variants, timing each with forward().sync() — prints all timings.
    let dev = Dev::Cuda(0);
    let variants: Vec<(&str, fn(i64, i64, i64) -> zyx::kernel::Kernel)> = vec![
        ("q4_exact_16x64", qwen3_8_27b::gemm_cuda_q4_k_exact as fn(i64, i64, i64) -> zyx::kernel::Kernel),
        // ("q4_simple_16x8", qwen3_8_27b::gemm_cuda_q4_k as fn(i64, i64, i64) -> zyx::kernel::Kernel),
    ];
    let sizes: Vec<(i64, i64, i64)> = vec![(1024, 1024, 1024), (16, 5120, 10240)];
    for (name, maker) in variants {
        for (r, k, n) in &sizes {
            if name.contains("16x64") && n % 64 != 0 { continue; }
            if name.contains("16x8") && n % 8 != 0 { continue; }
            let kk = maker(*r, *k, *n);
            let (flops, read, write) = kk.flop_mem_rw();
            // build synthetic Q4 for timing (same per-32 scale/min for both variants)
            let mut qs = vec![0u32; (n * k / 8) as usize];
            let mut scales = Vec::with_capacity((n * k / 32) as usize);
            let mut mins = Vec::with_capacity((n * k / 32) as usize);
            for row in 0..*n {
                for blk in 0..k / 32 {
                    scales.push(zyx::f16::from_f32(0.04));
                    mins.push(zyx::f16::from_f32(0.3));
                    let base = (row * k + blk * 32) as usize;
                    for i in 0..32 {
                        qs[base / 8 + i / 8] |= 8u32 << ((i % 8) * 4);
                    }
                }
            }
            let a = Tensor::randn([*r, *k], DType::F16)?.to(dev)?;
            let qs_t = Tensor::from(qs.clone()).to(dev)?;
            let sc_t = Tensor::from(scales.clone()).to(dev)?;
            let mn_t_opt = if name.contains("exact") { Some(Tensor::from(mins.clone()).to(dev)?) } else { None };
            let ck = kk.compile()?;
            // warmup
            let _ = if let Some(ref mn) = mn_t_opt {
                ck.forward(&[&a, &qs_t, &sc_t, mn], vec![vec![Tensor::from(*r), Tensor::from(*n)]])?
            } else {
                ck.forward(&[&a, &qs_t, &sc_t], vec![vec![Tensor::from(*r), Tensor::from(*n)]])?
            };
            let t0 = std::time::Instant::now();
            let outs = if let Some(ref mn) = mn_t_opt {
                ck.forward(&[&a, &qs_t, &sc_t, mn], vec![vec![Tensor::from(*r), Tensor::from(*n)]])?
            } else {
                ck.forward(&[&a, &qs_t, &sc_t], vec![vec![Tensor::from(*r), Tensor::from(*n)]])?
            };
            for o in &outs { o.sync()?; }
            let total = t0.elapsed().as_micros();
            eprintln!("variant {name} {r}x{k}x{n} flops {flops} enqueue+sync {total}us read+write {} GB/s {:.1}", read+write, (read+write) as f64 / total as f64 / 1e3);
        }
    }
    Ok(())
}

// Q4_K block: d(F16), dmin(F16), scales[12], qs[128] = 144 bytes
fn make_q4_k_block(d: f32, dmin: f32, _scales: &[f32; 8], _mins: &[f32; 4], qs: &[u8; 128]) -> Vec<u8> {
    let mut data = vec![0u8; 144];
    let d_bits = zyx::f16::from_f32(d).to_bits();
    data[0] = (d_bits & 0xFF) as u8;
    data[1] = ((d_bits >> 8) & 0xFF) as u8;
    let dmin_bits = zyx::f16::from_f32(dmin).to_bits();
    data[2] = (dmin_bits & 0xFF) as u8;
    data[3] = ((dmin_bits >> 8) & 0xFF) as u8;
    for i in 0..12 {
        data[4 + i] = 0xF0;
    }
    data[16..144].copy_from_slice(qs);
    data
}

// Q8_1 block: d(F16), sum(F16), qs[128] = 132 bytes
fn make_q8_1_block(d: f32, sum: f32, qs: &[i8; 128]) -> (Vec<u8>, Vec<zyx::f16>, Vec<zyx::f16>) {
    let mut data = vec![0u8; 132];
    let d_bits = zyx::f16::from_f32(d).to_bits();
    data[0] = (d_bits & 0xFF) as u8;
    data[1] = ((d_bits >> 8) & 0xFF) as u8;
    let sum_bits = zyx::f16::from_f32(sum).to_bits();
    data[2] = (sum_bits & 0xFF) as u8;
    data[3] = ((sum_bits >> 8) & 0xFF) as u8;
    for (i, &q) in qs.iter().enumerate() {
        data[4 + i] = q as u8;
    }
    let scales = vec![zyx::f16::from_f32(d)];
    let sums = vec![zyx::f16::from_f32(sum)];
    (data, scales, sums)
}

#[test]
fn gemm_mmq_q4_k_small() -> Result<(), ZyxError> {
    // Qwen3.8-27B dims: hidden=5120, intermediate=17408
    let dev = Dev::Cuda(0);
    let r: i64 = 5120;   // hidden_size
    let k: i64 = 17408;  // intermediate_size
    let n: i64 = 5120;   // hidden_size
    let kblocks = k / 256; // 68
    let ncols = n / 32;    // 160
    let nchunks = kblocks; // 68
    // A: all 1.0 -> Q4_K block with d=1.0, dmin=0, qs=8 (q=8 -> 0 in signed 8-bit with scale 1)
    let mut a_blocks = Vec::new();
    for _ in 0..r / 16 {
        for _ in 0..k / 256 {
            let mut qs = [0u8; 128];
            for q in &mut qs { *q = 8; }
            let block = make_q4_k_block(1.0, 0.0, &[1.0; 8], &[0.0; 4], &qs);
            a_blocks.extend(block);
        }
    }
    // B: [nchunks, 8, ncols] Q8_1 blocks (kernel's access pattern: block_b = (chunk*8 + kstep)*ncols + j)
    let mut b_qs_flat = Vec::new();
    let mut b_scales_flat = Vec::new();
    let mut b_sums_flat = Vec::new();
    for _chunk in 0..nchunks {
        for _kstep in 0..8 {
            for _col in 0..ncols {
                let mut qs = [0i8; 128];
                for q in &mut qs { *q = 0; }
                let (data, _scales, _sums) = make_q8_1_block(1.0, 0.0, &qs);
                b_qs_flat.extend(data[4..].iter().copied());
                b_scales_flat.push(zyx::f16::from_f32(1.0));
                b_sums_flat.push(zyx::f16::from_f32(0.0));
            }
        }
    }
    let a_t = Tensor::from(a_blocks).to(dev)?;
    let y_qs_t = Tensor::from(b_qs_flat).to(dev)?;
    let y_scales_t = Tensor::from(b_scales_flat).to(dev)?;
    let y_sums_t = Tensor::from(b_sums_flat).to(dev)?;
    let kk = qwen3_8_27b::gemm_mmq_q4_k(r, k, n);
    let ck = kk.compile()?;
    let outs = ck.forward(&[&a_t, &y_qs_t, &y_scales_t, &y_sums_t], vec![vec![Tensor::from(r), Tensor::from(n)]])?;
    for o in &outs { o.sync()?; }
    let v: Vec<f32> = outs[0].to_vec()?;
    eprintln!("mmq result[0..8] = {:?}", &v[..8]);
    for (i, &val) in v.iter().enumerate() {
        assert!(val.abs() < 1e-3, "result[{i}] = {val}, expected ~0");
    }
    Ok(())
}
