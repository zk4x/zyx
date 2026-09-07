// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Unified bench for all qwen kernels: builder+compile vs 50ms, forward async vs 50us, forward sync vs 1ms, forward+DtoH vs 1ms.
//! Run: cargo test --release --test bench_all -- --nocapture --test-threads=1

use std::time::Instant;
use zyx::kernel::Dev;
use zyx::{DType, Tensor, ZyxError};

fn bench_kernel<F>(name: &str, build: F, inputs: Vec<Tensor>, out_shapes: Vec<Vec<i64>>) -> Result<(), ZyxError>
where
    F: FnOnce() -> zyx::kernel::Kernel,
{
    let t0 = Instant::now();
    let k = build();
    let builder_us = t0.elapsed().as_micros();
    let t1 = Instant::now();
    let ck = k.compile()?;
    let compile_us = t1.elapsed().as_micros();
    let total_bc = builder_us + compile_us;
    eprintln!(
        "[bench] {name} builder {builder_us}us compile {compile_us}us total {total_bc}us budget 50000us {}",
        if total_bc <= 50000 { "OK" } else { "OVER" }
    );
    let in_refs: Vec<&Tensor> = inputs.iter().collect();
    let out_shapes_ref: Vec<Vec<Tensor>> = out_shapes
        .iter()
        .map(|s| s.iter().map(|&d| Tensor::from(d)).collect())
        .collect();
    // warmup async
    let _ = ck.forward(&in_refs, out_shapes_ref.clone())?;
    let t2 = Instant::now();
    let outs = ck.forward(&in_refs, out_shapes_ref.clone())?;
    let async_us = t2.elapsed().as_micros();
    eprintln!(
        "[bench] {name} forward async {async_us}us budget 50us {}",
        if async_us <= 50 { "OK" } else { "OVER" }
    );
    // sync without DtoH
    let t2s = Instant::now();
    let outs_s = ck.forward(&in_refs, out_shapes_ref.clone())?;
    for o in &outs_s {
        o.sync()?;
    }
    let sync_us = t2s.elapsed().as_micros();
    eprintln!(
        "[bench] {name} forward sync {sync_us}us budget 1000us {}",
        if sync_us <= 1000 { "OK" } else { "OVER" }
    );
    let t3 = Instant::now();
    let outs2 = ck.forward(&in_refs, out_shapes_ref)?;
    let mut total_elems = 0usize;
    for o in &outs2 {
        let v: Vec<f32> = o.to_vec()?;
        total_elems += v.len();
    }
    let forward_dtoh_us = t3.elapsed().as_micros();
    eprintln!(
        "[bench] {name} forward+DtoH {forward_dtoh_us}us elems {total_elems} budget 1000us {}",
        if forward_dtoh_us <= 1000 { "OK" } else { "OVER" }
    );
    let _ = outs;
    Ok(())
}

#[test]
fn bench_all() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    // mlp
    {
        let g = Tensor::randn([8, 128], DType::F32)?.to(dev)?;
        let u = Tensor::randn([8, 128], DType::F32)?.to(dev)?;
        bench_kernel("mlp 8,128", || qwen3_8_27b::mlp_kernel(8, 128), vec![g, u], vec![vec![8, 128]])?;
    }
    {
        let g = Tensor::randn([6, 17408], DType::F32)?.to(dev)?;
        let u = Tensor::randn([6, 17408], DType::F32)?.to(dev)?;
        bench_kernel("mlp 6,17408", || qwen3_8_27b::mlp_kernel(6, 17408), vec![g, u], vec![vec![6, 17408]])?;
    }
    // embed
    {
        let vocab = 256; let dim = 64; let seq = 8;
        let w = Tensor::randn([vocab, dim], DType::F32)?.to(dev)?;
        let ids = Tensor::randint::<i64>([seq], 0..vocab as i64)?.to(dev)?;
        bench_kernel("embed 256,64,8", || qwen3_8_27b::embed_kernel(vocab, dim, seq), vec![w, ids], vec![vec![seq, dim]])?;
    }
    {
        let vocab = 152064; let dim = 5120; let seq = 6; let vocab_bench = 8192;
        let w = Tensor::randn([vocab_bench, dim], DType::F32)?.to(dev)?;
        let ids = Tensor::randint::<i64>([seq], 0..vocab_bench as i64)?.to(dev)?;
        bench_kernel("embed 8192,5120,6 slice152064", || qwen3_8_27b::embed_kernel(vocab, dim, seq), vec![w, ids], vec![vec![seq, dim]])?;
    }
    // rope
    {
        let heads = 2; let seq = 4; let hd = 16; let rot = 4; let m = heads*seq;
        let x = Tensor::randn([m, hd], DType::F32)?.to(dev)?;
        let cos = Tensor::randn([seq, rot], DType::F32)?.to(dev)?;
        let sin = Tensor::randn([seq, rot], DType::F32)?.to(dev)?;
        bench_kernel("rope 2,4,16,4", || qwen3_8_27b::rope_kernel(seq, heads, hd, rot), vec![x, cos, sin], vec![vec![m, hd]])?;
    }
    {
        let heads = 8; let seq = 6; let hd = 128; let rot = 128; let m = heads*seq;
        let x = Tensor::randn([m, hd], DType::F32)?.to(dev)?;
        let cos = Tensor::randn([seq, rot], DType::F32)?.to(dev)?;
        let sin = Tensor::randn([seq, rot], DType::F32)?.to(dev)?;
        bench_kernel("rope 8,6,128,128", || qwen3_8_27b::rope_kernel(seq, heads, hd, rot), vec![x, cos, sin], vec![vec![m, hd]])?;
    }
    // attention (small golden)
    {
        let seq = 4; let h = 4; let kv = 2; let d = 8;
        let q = Tensor::randn([h, seq, d], DType::F32)?.to(dev)?;
        let k = Tensor::randn([kv, seq, d], DType::F32)?.to(dev)?;
        let v = Tensor::randn([kv, seq, d], DType::F32)?.to(dev)?;
        let gate = Tensor::randn([seq, h*d], DType::F32)?.to(dev)?;
        bench_kernel("attn 4,4,2,8", || qwen3_8_27b::attention_kernel(seq, h, kv, d), vec![q,k,v,gate], vec![vec![seq, h*d]])?;
    }
    // attention Qwen slice (seq6 h8 kv2 d128)
    {
        let seq = 6; let h = 8; let kv = 2; let d = 128;
        let q = Tensor::randn([h, seq, d], DType::F32)?.to(dev)?;
        let k = Tensor::randn([kv, seq, d], DType::F32)?.to(dev)?;
        let v = Tensor::randn([kv, seq, d], DType::F32)?.to(dev)?;
        let gate = Tensor::randn([seq, h*d], DType::F32)?.to(dev)?;
        bench_kernel("attn 6,8,2,128 slice", || qwen3_8_27b::attention_kernel(seq, h, kv, d), vec![q,k,v,gate], vec![vec![seq, h*d]])?;
    }
    // rmsnorm (M=6 D=5120)
    {
        let m = 6; let d = 5120;
        let x = Tensor::randn([m, d], DType::F32)?.to(dev)?;
        let w = Tensor::randn([d], DType::F32)?.to(dev)?;
        bench_kernel("rmsnorm 6,5120", || qwen3_8_27b::rmsnorm_kernel(m, d), vec![x,w], vec![vec![m,d]])?;
    }
    // gemm example (M=6 K=5120 N=17408 slice)
    {
        let m = 6; let k = 5120; let n = 1024;
        let a = Tensor::randn([m, k], DType::F32)?.to(dev)?;
        let b = Tensor::randn([k, n], DType::F32)?.to(dev)?;
        bench_kernel("gemm 6,5120,1024 slice", || qwen3_8_27b::gemm_kernel(m,k,n), vec![a,b], vec![vec![m,n]])?;
    }
    Ok(())
}
