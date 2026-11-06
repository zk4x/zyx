// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Llama forward-pass benchmark on random weights: no checkpoint is loaded, so
//! the numbers are pure compute.
//!
//! Two shapes, because they stress different parts of the stack:
//!
//! - **decode** — batch 1, seq 1. One token at a time; memory bound.
//! - **prefill** — batch 128, seq 128. Tensorization and smem tiling.
//!
//! The first call is measured on its own: it pays compilation and BeamSearch
//! for every shape, which would otherwise swamp the steady-state number.
//!
//! Run with: `cargo run --release --bin llama_bench`

use std::time::{Duration, Instant};

use llama::{Llama, LlamaConfig};
use zyx::{DType, Tensor, ZyxError};

const ITERS: usize = 5;
const DECODE_BATCH: i64 = 1;
const DECODE_SEQ: i64 = 1;
const PREFILL_BATCH: i64 = 128;
const PREFILL_SEQ: i64 = 128;

fn fmt(d: Duration) -> String {
    if d.as_secs_f64() >= 1.0 {
        format!("{:.3} s", d.as_secs_f64())
    } else {
        format!("{:.3} ms", d.as_secs_f64() * 1e3)
    }
}

fn bench(name: &str, cfg: &LlamaConfig, batch: i64, seq: i64) -> Result<(), ZyxError> {
    let mut model = Llama::random(cfg, DType::F16);
    let input = Tensor::from(vec![0u32; (batch * seq) as usize]).reshape([batch, seq])?;

    // Warmup also compiles: the first call pays JIT + BeamSearch for every
    // distinct shape, so it is reported separately rather than averaged in.
    let t = Instant::now();
    let out = model.forward(&input, 0);
    let dev = out.device();
    dev.synchronize()?;
    let compile = t.elapsed();

    let mut times = Vec::with_capacity(ITERS);
    for i in 0..ITERS {
        let t = Instant::now();
        let out = model.forward(&input, i + 1);
        out.device().synchronize()?;
        times.push(t.elapsed());
    }
    times.sort_unstable();
    let median = times[ITERS / 2];

    println!("{name}  batch={batch} seq={seq}  f16  dev={dev:?}");
    println!("  compile+autotune : {}", fmt(compile));
    println!("  steady state     : {}  (median of {ITERS})", fmt(median));
    println!(
        "  iterations       : {}",
        times.iter().map(|d| fmt(*d)).collect::<Vec<_>>().join(", ")
    );
    println!();
    Ok(())
}

fn main() -> Result<(), ZyxError> {
    Tensor::set_implicit_casts(false);
    // Llama-3.2-3B shapes: 28 layers, hidden 3072, 24 heads / 8 kv heads.
    let cfg = LlamaConfig::default();
    println!("llama-3.2-3b shapes, random weights, {ITERS} iterations each\n");
    bench("decode ", &cfg, DECODE_BATCH, DECODE_SEQ)?;
    bench("prefill", &cfg, PREFILL_BATCH, PREFILL_SEQ)?;
    Ok(())
}
