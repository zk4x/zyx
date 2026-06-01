// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Benchmarks for zyx tensor library.
//!
//! Run with: `cargo run --release`

use std::ops::{Add, Div, Mul, Sub};
use zyx::{DType, Module, Tensor, ZyxError};

fn matmul_bench() -> Result<(), ZyxError> {
    println!("=== MatMul ===");
    for &(m, n, k) in &[
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (128, 128, 128),
        (128, 256, 64),
        (256, 128, 512),
        (512, 256, 128),
        (1024, 512, 256),
        (256, 1024, 512),
        (512, 1024, 128),
        (128, 512, 1024),
        (2048, 2048, 2048),
    ] {
        let a = Tensor::rand([m, k], DType::F32)?;
        let b = Tensor::rand([n, k], DType::F32)?;
        a.matmul(b.t())?.realize()?;
    }
    println!();
    Ok(())
}

fn reduce_bench() -> Result<(), ZyxError> {
    println!("=== Reduce ===");
    for dims in [
        &[4096u64][..],
        &[16384],
        &[65536],
        &[4096, 4096],
        &[256, 1024, 512],
        &[64, 64, 64],
    ] {
        let x = Tensor::rand(dims, DType::F32)?;
        x.sum_all().realize_one()?;
        x.mean_all().realize_one()?;
        x.max_all().realize_one()?;
        x.min_all().realize_one()?;
        x.prod_all().realize_one()?;
        x.var_all().realize_one()?;
        x.std_all().realize_one()?;
    }
    println!();
    Ok(())
}

fn softmax_bench() -> Result<(), ZyxError> {
    println!("=== Softmax ===");
    for &(dims, ref axes) in &[
        (&[4096u64][..], &[0i32][..]),
        (&[128, 4096], &[0i32][..]),
        (&[512, 1024], &[1i32][..]),
        (&[16, 768, 1024], &[1i32][..]),
        (&[64, 64, 64, 64], &[3i32][..]),
        (&[64, 128, 4096], &[2i32][..]),
        (&[16384u64], &[0i32][..]),
        (&[256, 2048], &[1i32][..]),
        (&[32, 4096], &[0i32][..]),
        (&[8, 512, 2048], &[2i32][..]),
    ] {
        let x = Tensor::rand(dims, DType::F32)?;
        x.softmax(axes.iter().copied())?.realize_one()?;
    }
    println!();
    Ok(())
}

fn embedding_bench() -> Result<(), ZyxError> {
    println!("=== Embedding (index_select) ===");
    for &(vocab_size, embed_dim, seq_len) in
        &[(10000, 768, 64), (50000, 1024, 256), (100000, 768, 128)]
    {
        let embedding = Tensor::rand([vocab_size, embed_dim], DType::F32)?;
        let idx: Vec<i32> = (0..seq_len).map(|i| (i % vocab_size) as i32).collect();
        let indices = Tensor::from(idx);
        embedding.index_select(0, indices)?.realize_one()?;
    }
    println!();
    Ok(())
}

fn gelu_bench() -> Result<(), ZyxError> {
    println!("=== GELU Activation ===");
    for dims in [&[4096u64][..], &[16384], &[1024, 4096], &[256, 1024, 512]] {
        let x = Tensor::rand(dims, DType::F32)?;
        x.gelu().realize_one()?;
    }
    println!();
    Ok(())
}

fn activation_bench() -> Result<(), ZyxError> {
    println!("=== Other Activations ===");
    for dims in [
        &[16384u64][..],
        &[32768u64],
        &[65536u64],
        &[131072u64],
        &[524288u64],
        &[4096u64, 4096u64],
        &[512u64, 2048u64],
        &[128u64, 16384u64],
    ] {
        let x = Tensor::rand(dims, DType::F32)?;
        x.relu().realize_one()?;
        x.sigmoid().realize_one()?;
        x.tanh().realize_one()?;
        x.exp().realize_one()?;
        x.log(Tensor::from(2.0)).realize_one()?;
    }
    println!();
    Ok(())
}

fn ln_softmax_bench() -> Result<(), ZyxError> {
    println!("=== Ln Softmax ===");
    for &(dims, ref axes) in &[
        (&[4096u64][..], &[0i32][..]),
        (&[256, 4096], &[0i32][..]),
        (&[512, 1024], &[1i32][..]),
        (&[32, 768, 1024], &[1i32][..]),
        (&[64, 512], &[0i32][..]),
    ] {
        let x = Tensor::rand(dims, DType::F32)?;
        x.ln_softmax(axes.iter().copied())?.realize_one()?;
    }
    println!();
    Ok(())
}

fn reduce_axis_bench() -> Result<(), ZyxError> {
    println!("=== Reduce Axis ===");
    for dims in [
        &[512, 512][..],
        &[256, 1024],
        &[128, 2048],
        &[64, 4096],
        &[32, 128, 256],
        &[16, 512, 512],
        &[8, 1024, 1024],
        &[1024, 256],
        &[4096, 128],
        &[64, 16384],
        &[32, 32768],
        &[16, 65536],
        &[8, 128, 512],
        &[4, 256, 256, 256],
        &[1024, 1024],
        &[2048, 64],
        &[128, 128, 256],
        &[256, 128, 128],
        &[16, 256, 256],
        &[8, 64, 64, 64],
    ] {
        let x = Tensor::rand(dims, DType::F32)?;
        let last = dims.len() as i32 - 1;
        x.sum([last])?.realize_one()?;
        x.max([last])?.realize_one()?;
        x.min([last])?.realize_one()?;
        x.mean([last])?.realize_one()?;
        x.var([last])?.realize_one()?;
        // Reduce along axis 0 as well
        if dims.len() > 1 {
            x.sum([0])?.realize_one()?;
            x.max([0])?.realize_one()?;
            x.mean([0])?.realize_one()?;
        }
    }
    println!();
    Ok(())
}

fn silu_like_bench() -> Result<(), ZyxError> {
    println!("=== SiLU-like (x * sigmoid(x)) ===");
    for dims in [
        &[4096u64][..],
        &[16384],
        &[1024, 4096],
        &[512, 1024],
        &[256, 1024, 512],
        &[16, 768, 2048],
    ] {
        let x = Tensor::rand(dims, DType::F32)?;
        x.sigmoid().mul(&x).realize_one()?;
    }
    println!();
    Ok(())
}

fn layer_norm_like_bench() -> Result<(), ZyxError> {
    println!("=== Layer Norm-like (mean+var+normalize) ===");
    for dims in [&[128, 768][..], &[64, 1024], &[32, 2048], &[16, 768, 1024]] {
        let last = *dims.last().unwrap();
        let x = Tensor::rand(dims, DType::F32)?;
        let gamma = Tensor::rand([last], DType::F32)?;
        let beta = Tensor::rand([last], DType::F32)?;
        let mean = x.mean([-1])?.unsqueeze(-1)?;
        let centered = x.sub(&mean);
        let var = centered.var([-1])?.unsqueeze(-1)?;
        let std = var.add(Tensor::from(1e-5)).sqrt();
        let norm = centered.div(&std);
        norm.mul(&gamma).add(&beta).realize_one()?;
    }
    println!();
    Ok(())
}

fn reduce_multi_axis_bench() -> Result<(), ZyxError> {
    println!("=== Reduce Multi Axis ===");
    for dims in [
        &[64, 64, 64][..],
        &[32, 128, 256],
        &[16, 256, 128],
        &[8, 512, 64],
    ] {
        let x = Tensor::rand(dims, DType::F32)?;
        x.sum([0, 1])?.realize_one()?;
        x.max([0, 1])?.realize_one()?;
        x.mean([1, 2])?.realize_one()?;
        x.var([0, 2])?.realize_one()?;
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
    ln_softmax_bench()?;
    reduce_axis_bench()?;
    silu_like_bench()?;
    layer_norm_like_bench()?;
    reduce_multi_axis_bench()?;
    Ok(())
}
