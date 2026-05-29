// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Pure matmul benchmark — dump the generated kernel with ZYX_DEBUG=16.

use std::time::Instant;
use zyx::{DType, Scalar, Tensor, ZyxError};

fn matmul_bench(m: u64, n: u64, k: u64, iters: u32) -> Result<(), ZyxError> {
    let a = Tensor::rand([m, k], DType::F32)?;
    let b = Tensor::rand([n, k], DType::F32)?;
    Tensor::realize([&a, &b])?;

    let a_data: Vec<f32> = a.clone().try_into()?;
    let b_data: Vec<f32> = b.clone().try_into()?;

    // Warmup + compile + correctness
    let c = a.matmul(b.t()).unwrap();
    let z_data: Vec<f32> = c.try_into()?;
    let expected = matmul_cpu(&a_data, &b_data, m as usize, k as usize, n as usize);
    for (x, y) in z_data.into_iter().zip(expected) {
        if !x.is_equal(y) {
            panic!("matmul wrong! {x} != {y}");
        }
    }
    println!("correctness OK");

    // Timed (force sync by reading data back)
    let total = (0..iters)
        .map(|_| {
            let a = Tensor::rand([m, k], DType::F32)?;
            let b = Tensor::rand([n, k], DType::F32)?;
            let start = Instant::now();
            let c = a.matmul(b.t()).unwrap();
            let _data: Vec<f32> = c.try_into()?;
            Ok::<u128, ZyxError>(start.elapsed().as_micros())
        })
        .try_fold(0u128, |acc, t| t.map(|v| acc + v))?;

    let avg = total as f64 / iters as f64;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let gflops = flops / (avg * 1000.0);
    println!("matmul [{m},{k}]x[{k},{n}]  {avg:.0} μs  {gflops:.1} GFLOP/s");

    Ok(())
}

fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[j * k + l];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn main() -> Result<(), ZyxError> {
    matmul_bench(1024, 1024, 1024, 3)?;
    Ok(())
}
