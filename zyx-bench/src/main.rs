// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Pure matmul benchmark — dump the generated C kernel with ZYX_DEBUG=16.

use std::time::Instant;
use zyx::{DType, Tensor, ZyxError};

fn matmul_bench(m: u64, n: u64, k: u64, iters: u32) -> Result<(), ZyxError> {
    let a = Tensor::randn([m, k], DType::F32)?;
    let b = Tensor::randn([n, k], DType::F32)?;
    Tensor::realize([&a, &b])?;

    // Warmup + compile
    let c = a.matmul(b.t()).unwrap();
    c.realize_one()?;

    // Timed
    let total = (0..iters)
        .map(|_| {
            let a = Tensor::randn([m, k], DType::F32)?;
            let b = Tensor::randn([n, k], DType::F32)?;
            let start = Instant::now();
            let c = a.matmul(b.t()).unwrap();
            c.realize_one()?;
            Ok::<u128, ZyxError>(start.elapsed().as_micros())
        })
        .try_fold(0u128, |acc, t| t.map(|v| acc + v))?;

    let avg = total as f64 / iters as f64;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let gflops = flops / (avg * 1000.0); // avg μs → s
    println!("matmul [{m},{k}]x[{k},{n}]  {avg:.0} μs  {gflops:.1} GFLOP/s");

    Ok(())
}

fn main() -> Result<(), ZyxError> {
    println!("=== pure matmul benchmarks ===");

    // Small: fits in L1/L2
    matmul_bench(256, 256, 256, 100)?;

    // Medium: L2/L3
    matmul_bench(1024, 1024, 1024, 10)?;

    // Llama QKV sizes
    matmul_bench(3072, 3072, 3072, 5)?;  // Q or O projection
    matmul_bench(1024, 3072, 3072, 5)?;  // K or V projection

    // Attention score: [n_heads*seq, head_dim] @ [head_dim, seq]
    matmul_bench(24 * 128, 64, 128, 10)?;   // seq=64
    matmul_bench(24 * 4096, 4096, 128, 1)?; // seq=4096

    Ok(())
}
