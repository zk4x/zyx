// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Benchmarks for zyx tensor library.
//!
//! Run with: `cargo run --release`

use std::time::Instant;
use zyx::{DType, Module, Tensor, ZyxError};

fn matmul_bench() -> Result<(), ZyxError> {
    println!("=== MatMul ===");

    let sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)];

    for (m, n, k) in sizes {
        let flops = 2.0 * m as f64 * n as f64 * k as f64;

        let a = Tensor::rand([m, k], DType::F32)?;
        let b = Tensor::rand([n, k], DType::F32)?;
        Tensor::realize([&a, &b])?;

        // Warmup (discard timing)
        let c = a.matmul(b.t())?;
        c.realize()?;

        let total = (0..3)
            .map(|_| {
                let a = Tensor::rand([m, k], DType::F32)?;
                let b = Tensor::rand([n, k], DType::F32)?;
                let start = Instant::now();
                let c = a.matmul(b.t())?;
                c.realize_one()?;
                Ok::<u128, ZyxError>(start.elapsed().as_micros())
            })
            .try_fold(0u128, |acc, t| t.map(|v| acc + v))?;

        let avg = total as f64 / 3.0;
        let gflops_val = flops / (avg * 1000.0);
        //println!("  [{m},{k}]x[{k},{n}]  {avg:.0} μs  {gflops_val:.1} GFLOP/s");
    }

    println!();
    Ok(())
}

fn reduce_bench() -> Result<(), ZyxError> {
    println!("=== Reduce ===");

    let shapes: &[(&str, &[u64])] = &[
        ("[4096]", &[4096]),
        ("[16384]", &[16384]),
        ("[4096, 4096]", &[4096, 4096]),
        ("[256, 1024, 512]", &[256, 1024, 512]),
    ];

    for (label, dims) in shapes {
        let shape: Vec<u64> = dims.to_vec();
        let elem_count = shape.iter().product::<u64>();
        let x = Tensor::rand(&shape, DType::F32)?;
        Tensor::realize([&x])?;

        // Warmup
        let out = x.sum_all();
        out.realize_one()?;
        let out = x.mean_all();
        out.realize_one()?;
        let out = x.max_all();
        out.realize_one()?;
        let out = x.var_all();
        out.realize_one()?;
        let out = x.std_all();
        out.realize_one()?;

        // Sum all
        let total = (0..5)
            .map(|_| {
                let x = Tensor::rand(&shape, DType::F32)?;
                let start = Instant::now();
                let out = x.sum_all();
                out.realize_one()?;
                Ok::<u128, ZyxError>(start.elapsed().as_micros())
            })
            .try_fold(0u128, |acc, t| t.map(|v| acc + v))?;
        let avg = total as f64 / 5.0;
        let throughput = elem_count as f64 / (avg * 1000.0);
        //println!("  {label} sum_all  {avg:.0} μs  {throughput:.1}M elem/s");

        // Mean all
        let total = (0..5)
            .map(|_| {
                let x = Tensor::rand(&shape, DType::F32)?;
                let start = Instant::now();
                let out = x.mean_all();
                out.realize_one()?;
                Ok::<u128, ZyxError>(start.elapsed().as_micros())
            })
            .try_fold(0u128, |acc, t| t.map(|v| acc + v))?;
        let avg = total as f64 / 5.0;
        let throughput = elem_count as f64 / (avg * 1000.0);
        //println!("  {label} mean_all  {avg:.0} μs  {throughput:.1}M elem/s");

        // Max all
        let total = (0..5)
            .map(|_| {
                let x = Tensor::rand(&shape, DType::F32)?;
                let start = Instant::now();
                let out = x.max_all();
                out.realize_one()?;
                Ok::<u128, ZyxError>(start.elapsed().as_micros())
            })
            .try_fold(0u128, |acc, t| t.map(|v| acc + v))?;
        let avg = total as f64 / 5.0;
        let throughput = elem_count as f64 / (avg * 1000.0);
        //println!("  {label} max_all  {avg:.0} μs  {throughput:.1}M elem/s");

        // Var along last axis
        let total = (0..5)
            .map(|_| {
                let x = Tensor::rand(&shape, DType::F32)?;
                let start = Instant::now();
                let out = x.var([i32::try_from(shape.len() - 1).unwrap()])?;
                out.realize_one()?;
                Ok::<u128, ZyxError>(start.elapsed().as_micros())
            })
            .try_fold(0u128, |acc, t| t.map(|v| acc + v))?;
        let avg = total as f64 / 5.0;
        //println!("  {label} var([-1])  {avg:.0} μs");
    }

    println!();
    Ok(())
}

fn softmax_bench() -> Result<(), ZyxError> {
    println!("=== Softmax ===");

    let cases: &[(&str, &[u64], &[i32])] = &[
        ("[4096]", &[4096], &[]),
        ("[128, 4096]", &[128, 4096], &[]),
        ("[512, 1024]", &[512, 1024], &[1]),
        ("[16, 768, 1024]", &[16, 768, 1024], &[1]),
    ];

    for (label, dims, axes) in cases {
        let shape: Vec<u64> = dims.to_vec();
        let x = Tensor::rand(&shape, DType::F32)?;
        Tensor::realize([&x])?;

        // Warmup
        let out = x.softmax(axes.iter().copied())?;
        out.realize_one()?;

        let total = (0..5)
            .map(|_| {
                let x = Tensor::rand(&shape, DType::F32)?;
                let start = Instant::now();
                let out = x.softmax(axes.iter().copied())?;
                out.realize_one()?;
                Ok::<u128, ZyxError>(start.elapsed().as_micros())
            })
            .try_fold(0u128, |acc, t| t.map(|v| acc + v))?;

        let avg = total as f64 / 5.0;
        //println!("  {label} softmax({axes:?})  {avg:.0} μs");
    }

    println!();
    Ok(())
}

fn embedding_bench() -> Result<(), ZyxError> {
    println!("=== Embedding (index_select) ===");

    let cases: &[(&str, u64, u64, u64)] = &[
        ("[10000, 768] x [64]", 10000, 768, 64),
        ("[50000, 1024] x [256]", 50000, 1024, 256),
        ("[100000, 768] x [128]", 100000, 768, 128),
    ];

    for (label, vocab_size, embed_dim, seq_len) in cases {
        let embedding = Tensor::rand([*vocab_size, *embed_dim], DType::F32)?;
        Tensor::realize([&embedding])?;

        // Warmup
        let indices = Tensor::from(vec![0i32; *seq_len as usize]);
        let out = embedding.index_select(0, indices)?;
        out.realize_one()?;

        let total = (0..5)
            .map(|_| {
                let embedding = Tensor::rand([*vocab_size, *embed_dim], DType::F32)?;
                let idx: Vec<i32> = (0..*seq_len).map(|i| (i % vocab_size) as i32).collect();
                let indices = Tensor::from(idx);
                let start = Instant::now();
                let out = embedding.index_select(0, indices)?;
                out.realize_one()?;
                Ok::<u128, ZyxError>(start.elapsed().as_micros())
            })
            .try_fold(0u128, |acc, t| t.map(|v| acc + v))?;

        let avg = total as f64 / 5.0;
        //println!("  {label}  {avg:.0} μs");
    }

    println!();
    Ok(())
}

fn gelu_bench() -> Result<(), ZyxError> {
    println!("=== GELU Activation ===");

    let shapes: &[(&str, &[u64])] = &[
        ("[4096]", &[4096]),
        ("[16384]", &[16384]),
        ("[1024, 4096]", &[1024, 4096]),
        ("[256, 1024, 512]", &[256, 1024, 512]),
    ];

    for (label, dims) in shapes {
        let shape: Vec<u64> = dims.to_vec();
        let x = Tensor::rand(&shape, DType::F32)?;
        Tensor::realize([&x])?;

        // Warmup
        let out = x.gelu();
        out.realize_one()?;

        let total = (0..5)
            .map(|_| {
                let x = Tensor::rand(&shape, DType::F32)?;
                let start = Instant::now();
                let out = x.gelu();
                out.realize_one()?;
                Ok::<u128, ZyxError>(start.elapsed().as_micros())
            })
            .try_fold(0u128, |acc, t| t.map(|v| acc + v))?;

        let avg = total as f64 / 5.0;
        //println!("  {label} gelu  {avg:.0} μs");
    }

    println!();
    Ok(())
}

fn activation_bench() -> Result<(), ZyxError> {
    println!("=== Other Activations ===");

    let shape: [u64; 1] = [16384];
    let x = Tensor::rand(shape, DType::F32)?;
    Tensor::realize([&x])?;

    // Warmup
    let _ = x.relu();
    let _ = x.sigmoid();
    let _ = x.tanh();
    let _ = x.exp();
    let _ = x.log(Tensor::from(2.0));

    let activations = [
        (
            "relu",
            Box::new(|t: &Tensor| -> Result<Tensor, ZyxError> { Ok(t.relu()) })
                as Box<dyn Fn(&Tensor) -> Result<Tensor, ZyxError> + Send>,
        ),
        (
            "sigmoid",
            Box::new(|t: &Tensor| -> Result<Tensor, ZyxError> { Ok(t.sigmoid()) })
                as Box<dyn Fn(&Tensor) -> Result<Tensor, ZyxError> + Send>,
        ),
        (
            "tanh",
            Box::new(|t: &Tensor| -> Result<Tensor, ZyxError> { Ok(t.tanh()) })
                as Box<dyn Fn(&Tensor) -> Result<Tensor, ZyxError> + Send>,
        ),
        (
            "exp",
            Box::new(|t: &Tensor| -> Result<Tensor, ZyxError> { Ok(t.exp()) })
                as Box<dyn Fn(&Tensor) -> Result<Tensor, ZyxError> + Send>,
        ),
        (
            "log",
            Box::new(|t: &Tensor| -> Result<Tensor, ZyxError> { Ok(t.log(2)) })
                as Box<dyn Fn(&Tensor) -> Result<Tensor, ZyxError> + Send>,
        ),
    ];

    for (name, op) in &activations {
        let total = (0..5)
            .map(|_| {
                let x = Tensor::rand(shape, DType::F32)?;
                let start = Instant::now();
                let out = op(&x)?;
                out.realize_one()?;
                Ok::<u128, ZyxError>(start.elapsed().as_micros())
            })
            .try_fold(0u128, |acc, t| t.map(|v| acc + v))?;

        let avg = total as f64 / 5.0;
        //println!("  {:?} {name}  {avg:.0} μs", shape);
    }

    println!();
    Ok(())
}

fn main() -> Result<(), ZyxError> {
    matmul_bench()?;
    reduce_bench()?;
    softmax_bench()?;
    embedding_bench()?;
    gelu_bench()?;
    activation_bench()?;
    Ok(())
}
