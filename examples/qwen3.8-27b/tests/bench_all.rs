// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Unified bench for all qwen kernels + estimate tok/s.
//! Real GGUF dims: examples/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
//!  qwen35 block 65 embed 5120 ffn 17408 heads 24 kv 4 key 256 val 256 rope 64
//!  ssm inner 6144 conv 4 state 128 group 16 rank 48 interval 4
//!  blk ssm: qkv 5120x10240 out 6144x5120 ; full attn: q 5120x12288 k/v 5120x1024 out 6144x5120
//!  ffn: gate/up 5120x17408 down 17408x5120 ; lm_head 5120x248320 vocab 248320
//! Run: cargo test --release --test bench_all -- --nocapture --test-threads=1

use std::time::Instant;
use zyx::kernel::Dev;
use zyx::{DType, Tensor, ZyxError};

fn bench_kernel<F>(
    name: &str,
    build: F,
    inputs: Vec<Tensor>,
    out_shapes: Vec<Vec<i64>>,
) -> Result<(u128, u64, u64, u64), ZyxError>
where
    F: FnOnce() -> zyx::kernel::Kernel,
{
    let t0 = Instant::now();
    let k = build();
    let builder_us = t0.elapsed().as_micros();
    let (flops, read, write) = k.flop_mem_rw();
    let t1 = Instant::now();
    let ck = k.compile()?;
    let compile_us = t1.elapsed().as_micros();
    let total_bc = builder_us + compile_us;
    eprintln!(
        "[bench] {name} flops {flops} read {read} write {write} builder {builder_us}us compile {compile_us}us total {total_bc}us budget 50000us {}",
        if total_bc <= 50000 { "OK" } else { "OVER" }
    );
    let in_refs: Vec<&Tensor> = inputs.iter().collect();
    let out_shapes_ref: Vec<Vec<Tensor>> = out_shapes
        .iter()
        .map(|s| s.iter().map(|&d| Tensor::from(d)).collect())
        .collect();
    let _ = ck.forward(&in_refs, out_shapes_ref.clone())?;
    // Sync-timed launch: forward only enqueues (async), sync blocks until done.
    // tok/s must use enqueue+sync total, never async enqueue alone.
    let t2 = Instant::now();
    let outs = ck.forward(&in_refs, out_shapes_ref.clone())?;
    let enqueue_us = t2.elapsed().as_micros();
    let t3s = Instant::now();
    for o in &outs {
        o.sync()?;
    }
    let sync_only_us = t3s.elapsed().as_micros();
    let sync_us = enqueue_us + sync_only_us;
    eprintln!("[bench] {name} enqueue {enqueue_us}us sync {sync_only_us}us total {sync_us}us");
    let tflops = if sync_us > 0 {
        flops as f64 / sync_us as f64 / 1e6
    } else {
        0.0
    };
    let gbs = if sync_us > 0 {
        (read + write) as f64 / sync_us as f64 / 1e3
    } else {
        0.0
    };
    eprintln!(
        "[bench] {name} forward+sync {sync_us}us TFLOPS {tflops:.3} GB/s {gbs:.1} budget 1000us {}",
        if sync_us <= 1000 { "OK" } else { "OVER" }
    );
    let t3 = Instant::now();
    let outs2 = ck.forward(&in_refs, out_shapes_ref)?;
    let mut elems = 0usize;
    for o in &outs2 {
        let v: Vec<f32> = o.to_vec()?;
        elems += v.len();
    }
    let dtoh_us = t3.elapsed().as_micros();
    eprintln!(
        "[bench] {name} forward+DtoH {dtoh_us}us elems {elems} budget 1000us {}",
        if dtoh_us <= 1000 { "OK" } else { "OVER" }
    );
    Ok((sync_us, flops, read, write))
}

#[test]
fn bench_all() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let mut totals: Vec<(&str, u128, u64, u64, u64, usize)> = Vec::new();
    // mlp SwiGLU fused 6,17408 (ffn gate*up) -> still representative for FFN elementwise
    {
        let g = Tensor::randn([6, 17408], DType::F32)?.to(dev)?;
        let u = Tensor::randn([6, 17408], DType::F32)?.to(dev)?;
        let (us, flops, read, write) = bench_kernel(
            "mlp 6,17408",
            || qwen3_8_27b::mlp_kernel(6, 17408),
            vec![g, u],
            vec![vec![6, 17408]],
        )?;
        totals.push(("mlp", us, flops, read, write, 65));
    }
    // embed vocab 248320 dim 5120 seq 6 slice 8192 for VRAM
    {
        let vocab = 248320;
        let vocab_bench = 8192;
        let dim = 5120;
        let seq = 6;
        let w = Tensor::randn([vocab_bench, dim], DType::F32)?.to(dev)?;
        let ids = Tensor::randint::<i64>([seq], 0..vocab_bench as i64)?.to(dev)?;
        let (us, flops, read, write) = bench_kernel(
            "embed 8192,5120,6",
            || qwen3_8_27b::embed_kernel(vocab, dim, seq),
            vec![w, ids],
            vec![vec![seq, dim]],
        )?;
        totals.push(("embed", us, flops, read, write, 1));
    }
    // rope full-attn dims: heads 24 kv 4 key 256 val 256 rope 64 -> demo seq 6
    {
        let heads = 24;
        let seq = 6;
        let hd = 256;
        let rot = 64;
        let m = heads * seq;
        let x = Tensor::randn([m, hd], DType::F32)?.to(dev)?;
        let cos = Tensor::randn([seq, rot], DType::F32)?.to(dev)?;
        let sin = Tensor::randn([seq, rot], DType::F32)?.to(dev)?;
        let (us, flops, read, write) = bench_kernel(
            "rope 24,6,256,64",
            || qwen3_8_27b::rope_kernel(seq, heads, hd, rot),
            vec![x, cos, sin],
            vec![vec![m, hd]],
        )?;
        totals.push(("rope", us, flops, read, write, 16));
    }
    // attention full dims: seq 6 h 24 kv 4 d 256 (GQA)
    {
        let seq = 6;
        let h = 24;
        let kv = 4;
        let d = 256;
        let q = Tensor::randn([h, seq, d], DType::F32)?.to(dev)?;
        let k = Tensor::randn([kv, seq, d], DType::F32)?.to(dev)?;
        let v = Tensor::randn([kv, seq, d], DType::F32)?.to(dev)?;
        let gate = Tensor::randn([seq, h * d], DType::F32)?.to(dev)?;
        let (us, flops, read, write) = bench_kernel(
            "attn 6,24,4,256",
            || qwen3_8_27b::attention_kernel(seq, h, kv, d),
            vec![q, k, v, gate],
            vec![vec![seq, h * d]],
        )?;
        totals.push(("attn", us, flops, read, write, 16));
    }
    // gemm real sizes from GGUF, r pad 16
    let gemm_sizes: Vec<(&str, i64, i64, i64, usize)> = vec![
        ("gemm ssm_qkv 16,5120,10240", 16, 5120, 10240, 49),
        ("gemm ssm_out 16,6144,5120", 16, 6144, 5120, 49),
        ("gemm attn_q 16,5120,12288", 16, 5120, 12288, 16),
        ("gemm attn_k 16,5120,1024", 16, 5120, 1024, 16),
        ("gemm attn_v 16,5120,1024", 16, 5120, 1024, 16),
        ("gemm attn_o 16,6144,5120", 16, 6144, 5120, 16),
        ("gemm ffn_gate 16,5120,17408", 16, 5120, 17408, 65),
        ("gemm ffn_up 16,5120,17408", 16, 5120, 17408, 65),
        ("gemm ffn_down 16,17408,5120", 16, 17408, 5120, 65),
        ("gemm lm_head 16,5120,248320", 16, 5120, 248320, 1),
    ];
    for (name, r, k, n, cnt) in gemm_sizes {
        let n_bench = if n == 248320 { 8192 } else { n };
        let is_lm = n == 248320;
        let n_ = if is_lm { n_bench } else { n };
        let a = Tensor::randn([r, k], DType::F16)?.to(dev)?;
        let b = Tensor::randn([n_, k], DType::F16)?.to(dev)?;
        let (us, flops, read, write) = bench_kernel(
            name,
            {
                let r_ = r;
                let k_ = k;
                let n__ = n_;
                move || qwen3_8_27b::gemm_kernel(r_, k_, n__)
            },
            vec![a, b],
            vec![vec![r as i64, n_ as i64]],
        )?;
        totals.push((
            Box::leak(name.to_string().into_boxed_str()),
            us,
            flops,
            read,
            write,
            cnt,
        ));
    }
    // Exact Q4_K gemm benches: host converts GGUF Q4_K blocks (d/dmin + 8×6b sc/min
    // per 256) into per-32 (scale=d*sc, min=dmin*m) + 4b qs. Kernel is then
    // `q*scale-min` — exact for Q4_K and reusable for Q3_K/Q5_K/Q6_K/IQ4_XS/Q8_0
    // after host-side expansion to 4b. Covers Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_XS mix.
    let q4_sizes: Vec<(&str, i64, i64, i64, usize)> = vec![
        ("q4x ssm_qkv 16,5120,10240", 16, 5120, 10240, 49),
        ("q4x ffn_gate 16,5120,17408", 16, 5120, 17408, 65),
        ("q4x ffn_down 16,17408,5120", 16, 17408, 5120, 65),
        ("q4x lm_head 16,5120,248320", 16, 5120, 248320, 1),
    ];
    for (name, r, k, n, cnt) in q4_sizes {
        let n_bench = if n == 248320 { 8192 } else { n };
        let n_ = if n == 248320 { n_bench } else { n };
        // Mimic GGUF Q4_K super-block: per 256 rows share d/dmin, per 32 sub-block
        // 6b sc/min. Generate deterministic d/dmin/sc/m then quantize synthetic weights.
        let b_f32: Vec<f32> = (0..(n_ * k) as usize)
            .map(|i| (i as f32 * 0.02).cos() * 0.5)
            .collect();
        let mut qs = vec![0u32; (n_ * k / 8) as usize];
        let mut scales = Vec::with_capacity((n_ * k / 32) as usize);
        let mut mins = Vec::with_capacity((n_ * k / 32) as usize);
        for row in 0..n_ {
            for blk256 in 0..k / 256 {
                // super-block scales vary slowly like real GGUF
                let d = 0.02 + 0.01 * ((row * 7 + blk256 * 13) % 5) as f32;
                let dmin = 0.015 + 0.008 * ((row * 3 + blk256 * 11) % 4) as f32;
                for sub in 0..8 {
                    let blk = blk256 * 8 + sub;
                    let base = (row * k + blk * 32) as usize;
                    let sc6 = 20.0 + ((row + blk as i64 * 3) % 30) as f32; // 6b 0..63
                    let m6 = 18.0 + ((row * 2 + blk as i64) % 28) as f32;
                    let scale = d * sc6 / 32.0;
                    let min = dmin * m6 / 32.0;
                    scales.push(zyx::f16::from_f32(scale));
                    mins.push(zyx::f16::from_f32(min));
                    for i in 0..32 {
                        let q = ((b_f32[base + i] + min) / scale).round().clamp(0.0, 15.0) as u32;
                        let idx = base / 8 + i / 8;
                        let shift = (i % 8) * 4;
                        qs[idx as usize] |= (q & 0xF) << shift;
                    }
                }
            }
        }
        let a = Tensor::randn([r, k], DType::F16)?.to(dev)?;
        let qs_t = Tensor::from(qs.clone()).to(dev)?;
        let sc_t = Tensor::from(scales.clone()).to(dev)?;
        let mn_t = Tensor::from(mins.clone()).to(dev)?;
        let (us, flops, read, write) = bench_kernel(
            name,
            {
                let r_ = r;
                let k_ = k;
                let n__ = n_;
                move || qwen3_8_27b::gemm_cuda_q4_k(r_, k_, n__)
            },
            vec![a, qs_t, sc_t, mn_t],
            vec![vec![r as i64, n_ as i64]],
        )?;
        totals.push((
            Box::leak(name.to_string().into_boxed_str()),
            us,
            flops,
            read,
            write,
            cnt,
        ));
    }
    let total_sync: u128 = totals
        .iter()
        .map(|(_, us, _, _, _, cnt)| us * *cnt as u128)
        .sum();
    let total_flops: u128 = totals
        .iter()
        .map(|(_, _, f, _, _, cnt)| *f as u128 * *cnt as u128)
        .sum();
    let total_bytes: u128 = totals
        .iter()
        .map(|(_, _, _, r, w, cnt)| (*r as u128 + *w as u128) * *cnt as u128)
        .sum();
    eprintln!("[estimate] per-kernel forward+sync | TFLOPS GB/s:");
    for (name, us, flops, read, write, cnt) in &totals {
        let tf = if *us > 0 {
            *flops as f64 / *us as f64 / 1e6
        } else {
            0.0
        };
        let gb = if *us > 0 {
            (*read as f64 + *write as f64) / *us as f64 / 1e3
        } else {
            0.0
        };
        eprintln!(
            "  {name} {us}us TFLOPS {tf:.3} GB/s {gb:.1} x {cnt} = {}us",
            us * *cnt as u128
        );
    }
    eprintln!("[estimate] total forward+sync per forward (seq 6) {total_sync}us = {}ms flops {total_flops} bytes {total_bytes}", total_sync as f64/1000.0);
    if total_sync > 0 {
        let tflops = total_flops as f64 / total_sync as f64 / 1e6;
        let gbs = total_bytes as f64 / total_sync as f64 / 1e3;
        eprintln!("[estimate] avg TFLOPS {tflops:.3} GB/s {gbs:.1}");
        let tok_per_s = 1_000_000.0 / total_sync as f64 * 6.0;
        eprintln!("[estimate] tok/s (seq6) {tok_per_s:.2}");
        let tok_per_s1 = 1_000_000.0 / total_sync as f64;
        eprintln!("[estimate] forward/s {tok_per_s1:.2}");
    }
    Ok(())
}
