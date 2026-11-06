// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Benchmarks for zyx tensor library.
//!
//! Run with: `cargo run --release`

#![allow(unused)]

use std::io::BufRead;
use std::ops::{Add, Div, Mul, Sub};
use zyx::{DType, Module, Tensor, ZyxError};

fn matmul_bench() -> Result<(), ZyxError> {
    //println!("=== MatMul ===");
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
        a.matmul(b.t())?;
    }
    //println!();
    Ok(())
}

fn reduce_bench() -> Result<(), ZyxError> {
    //println!("=== Reduce ===");
    for dims in [
        &[4096u64][..],
        &[16384],
        &[65536],
        &[4096, 4096],
        &[256, 1024, 512],
        &[64, 64, 64],
    ] {
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        x.sum_all();
        x.mean_all();
        x.max_all();
        x.min_all();
        x.prod_all();
        x.var_all();
        x.std_all();
    }
    //println!();
    Ok(())
}

fn softmax_bench() -> Result<(), ZyxError> {
    //println!("=== Softmax ===");
    for &(dims, axes) in &[
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
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        x.softmax(axes.iter().copied())?;
    }
    //println!();
    Ok(())
}

fn embedding_bench() -> Result<(), ZyxError> {
    //println!("=== Embedding (index_select) ===");
    for &(vocab_size, embed_dim, seq_len) in
        &[(10000, 768, 64), (50000, 1024, 256), (100000, 768, 128)]
    {
        let embedding = Tensor::rand([vocab_size, embed_dim], DType::F32)?;
        let idx: Vec<i32> = (0..seq_len).map(|i| (i % vocab_size)).collect();
        let indices = Tensor::from(idx);
        embedding.index_select(0, indices)?;
    }
    //println!();
    Ok(())
}

fn gelu_bench() -> Result<(), ZyxError> {
    //println!("=== GELU Activation ===");
    for dims in [&[4096u64][..], &[16384], &[1024, 4096], &[256, 1024, 512]] {
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        x.gelu();
    }
    //println!();
    Ok(())
}

fn activation_bench() -> Result<(), ZyxError> {
    //println!("=== Other Activations ===");
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
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        x.relu();
        x.sigmoid();
        x.tanh();
        x.exp();
        x.log(Tensor::from(2.0));
    }
    //println!();
    Ok(())
}

fn ln_softmax_bench() -> Result<(), ZyxError> {
    //println!("=== Ln Softmax ===");
    for &(dims, axes) in &[
        (&[4096u64][..], &[0i32][..]),
        (&[256, 4096], &[0i32][..]),
        (&[512, 1024], &[1i32][..]),
        (&[32, 768, 1024], &[1i32][..]),
        (&[64, 512], &[0i32][..]),
    ] {
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        x.ln_softmax(axes.iter().copied())?;
    }
    //println!();
    Ok(())
}

fn reduce_axis_bench() -> Result<(), ZyxError> {
    //println!("=== Reduce Axis ===");
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
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        let last = dims.len() as i32 - 1;
        x.sum([last])?;
        x.max([last])?;
        x.min([last])?;
        x.mean([last])?;
        x.var([last])?;
        // Reduce along axis 0 as well
        if dims.len() > 1 {
            x.sum([0])?;
            x.max([0])?;
            x.mean([0])?;
        }
    }
    //println!();
    Ok(())
}

fn silu_like_bench() -> Result<(), ZyxError> {
    //println!("=== SiLU-like (x * sigmoid(x)) ===");
    for dims in [
        &[4096u64][..],
        &[16384],
        &[1024, 4096],
        &[512, 1024],
        &[256, 1024, 512],
        &[16, 768, 2048],
    ] {
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        x.sigmoid().mul(&x);
    }
    //println!();
    Ok(())
}

fn layer_norm_like_bench() -> Result<(), ZyxError> {
    //println!("=== Layer Norm-like (mean+var+normalize) ===");
    for dims in [&[128, 768][..], &[64, 1024], &[32, 2048], &[16, 768, 1024]] {
        let last = *dims.last().unwrap();
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        let gamma = Tensor::rand([last], DType::F32)?;
        let beta = Tensor::rand([last], DType::F32)?;
        let mean = x.mean([-1])?.unsqueeze(-1)?;
        let centered = x.sub(&mean);
        let var = centered.var([-1])?.unsqueeze(-1)?;
        let std = var.add(Tensor::from(1e-5)).sqrt();
        let norm = centered.div(&std);
        norm.mul(&gamma).add(&beta);
    }
    //println!();
    Ok(())
}

fn reduce_multi_axis_bench() -> Result<(), ZyxError> {
    //println!("=== Reduce Multi Axis ===");
    for dims in [
        &[64, 64, 64][..],
        &[32, 128, 256],
        &[16, 256, 128],
        &[8, 512, 64],
    ] {
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        x.sum([0, 1])?;
        x.max([0, 1])?;
        x.mean([1, 2])?;
        x.var([0, 2])?;
    }
    //println!();
    Ok(())
}

/// Force `t` to execute — and therefore to compile, autotune and measure its
/// kernel — by reading one element back. Eager ops are lazy until a value is
/// read; dropping a result does not launch anything.
fn force(t: Tensor) -> Result<(), ZyxError> {
    let _ = t.item::<f32>();
    Ok(())
}

fn large_matmul_bench() -> Result<(), ZyxError> {
    //println!("=== Large MatMul ===");
    let dims: &[u64] = &[
        256, 320, 384, 448, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1792, 2048, 2304,
        2560, 3072, 3584, 4096, 4608, 5120, 6144,
    ];
    // Square
    for &n in dims {
        let a = Tensor::rand([n, n], DType::F32)?;
        let b = Tensor::rand([n, n], DType::F32)?;
        force(a.matmul(b.t())?)?;
    }
    // Vary M (tall matrices)
    for &m in dims.iter().filter(|&&m| m != 1024) {
        let a = Tensor::rand([m, 1024], DType::F32)?;
        let b = Tensor::rand([1024, 1024], DType::F32)?;
        force(a.matmul(b.t())?)?;
    }
    // Vary N (wide matrices)
    for &n in dims.iter().filter(|&&n| n != 1024) {
        let a = Tensor::rand([1024, 1024], DType::F32)?;
        let b = Tensor::rand([n, 1024], DType::F32)?;
        force(a.matmul(b.t())?)?;
    }
    // Vary K (inner dimension)
    for &k in dims.iter().filter(|&&k| k != 1024) {
        let a = Tensor::rand([1024, k], DType::F32)?;
        let b = Tensor::rand([1024, k], DType::F32)?;
        force(a.matmul(b.t())?)?;
    }
    //println!();
    Ok(())
}

// ---- measurement database ----

/// Append-only sqlite sink for autotune measurements.
///
/// `zyx` itself never touches a database: the autotuner writes one JSON object
/// per measured candidate to `ZYX_BENCH_OUT` (see
/// `Kernel::record_measurement`). The benchmark collects that stream for the
/// duration of the run and appends every row to `zyx-bench/bench.db`, creating
/// the table on first use. Rows are never updated or deleted — repeating a run
/// appends duplicates on purpose (more samples per config).
const DB_FILE: &str = "bench.db";

/// One line of the `ZYX_BENCH_OUT` stream (`schema: 1`).
#[derive(serde::Deserialize)]
struct Measurement {
    schema: i64,
    tag: String,
    dev: String,
    cc: [i64; 2],
    seed: String,
    kernel: String,
    seq: serde_json::Value,
    nanos: Option<i64>,
    status: String,
    features: serde_json::Value,
}

fn open_db(path: &std::path::Path) -> rusqlite::Result<rusqlite::Connection> {
    let conn = rusqlite::Connection::open(path)?;
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         CREATE TABLE IF NOT EXISTS measurements (
             id       INTEGER PRIMARY KEY AUTOINCREMENT,
             ts       INTEGER NOT NULL,
             schema   INTEGER NOT NULL,
             tag      TEXT    NOT NULL,
             dev      TEXT    NOT NULL,
             cc0      INTEGER NOT NULL,
             cc1      INTEGER NOT NULL,
             seed     TEXT    NOT NULL,
             kernel   TEXT    NOT NULL,
             seq      TEXT    NOT NULL,
             nanos    INTEGER,
             status   TEXT    NOT NULL,
             features TEXT    NOT NULL
         );",
    )?;
    Ok(conn)
}

/// Append every measurement in `stream` to the database at `db`.
fn ingest(db: &std::path::Path, stream: &std::path::Path) -> Result<usize, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(stream)?;
    let mut conn = open_db(db)?;
    let tx = conn.transaction()?;
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs() as i64;
    let mut inserted = 0usize;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO measurements
                 (ts, schema, tag, dev, cc0, cc1, seed, kernel, seq, nanos, status, features)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        )?;
        for line in std::io::BufReader::new(file).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let m: Measurement = serde_json::from_str(&line)?;
            stmt.execute(rusqlite::params![
                ts,
                m.schema,
                m.tag,
                m.dev,
                m.cc[0],
                m.cc[1],
                m.seed,
                m.kernel,
                m.seq.to_string(),
                m.nanos,
                m.status,
                m.features.to_string(),
            ])?;
            inserted += 1;
        }
    }
    tx.commit()?;
    Ok(inserted)
}

fn bench_stream_path() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("zyx-bench-{}.jsonl", std::process::id()))
}

fn main() -> Result<(), ZyxError> {
    let db = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(DB_FILE);
    let stream = bench_stream_path();
    // The autotuner appends one JSON object per measured candidate here; main
    // drains it into sqlite once the benches are done. Nothing is updated.
    std::env::set_var("ZYX_BENCH_OUT", &stream);

    let result = run_benches();

    match ingest(&db, &stream) {
        Ok(n) => println!("appended {n} measurements to {}", db.display()),
        Err(e) => eprintln!("failed to write {}: {e}", db.display()),
    }
    let _ = std::fs::remove_file(&stream);
    result
}

fn run_benches() -> Result<(), ZyxError> {
    //matmul_bench()?;
    //reduce_bench()?;
    //softmax_bench()?;
    //embedding_bench()?;
    //gelu_bench()?;
    //activation_bench()?;
    //ln_softmax_bench()?;
    //reduce_axis_bench()?;
    //silu_like_bench()?;
    //layer_norm_like_bench()?;
    //reduce_multi_axis_bench()?;
    large_matmul_bench()?;
    Ok(())
}
