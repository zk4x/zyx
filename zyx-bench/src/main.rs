// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Benchmarks for zyx tensor library.
//!
//! Every benchmark lives in its own module under `bench/` and is runnable on
//! its own by name:
//!
//! ```text
//! cargo run --release --bin zyx-bench -- --list
//! cargo run --release --bin zyx-bench -- llama-matmul
//! ```
//!
//! No argument keeps the historical default (the large-matmul sweep). Measured
//! autotune candidates are appended to `bench.db`; set `ZYX_BENCH_TAG` to label
//! a run.

#![allow(unused)]

mod bench;

use std::io::BufRead;
use zyx::ZyxError;

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

/// Every benchmark, runnable on its own by name; no argument keeps the
/// historical default (the large-matmul sweep).
const BENCHES: &[&str] = &[
    "large-matmul",
    "matmul",
    "reduce",
    "softmax",
    "embedding",
    "gelu",
    "activation",
    "ln-softmax",
    "reduce-axis",
    "silu-like",
    "layer-norm-like",
    "reduce-multi-axis",
    "llama-matmul",
    "llama-rmsnorm",
    "llama-rope",
    "llama-softmax",
    "llama-swiglu",
    "llama-embedding",
    "llama-kv-assign",
    "llama-attention",
    "mma-nt",
];

fn run_benches() -> Result<(), ZyxError> {
    let arg = std::env::args().nth(1);
    let name = match arg.as_deref() {
        None => return bench::large_matmul::bench(),
        Some("--list") => {
            for b in BENCHES {
                println!("{b}");
            }
            return Ok(());
        }
        Some(name) => name,
    };
    match name {
        "large-matmul" => bench::large_matmul::bench(),
        "matmul" => bench::matmul::bench(),
        "reduce" => bench::reduce::bench(),
        "softmax" => bench::softmax::bench(),
        "embedding" => bench::embedding::bench(),
        "gelu" => bench::gelu::bench(),
        "activation" => bench::activation::bench(),
        "ln-softmax" => bench::ln_softmax::bench(),
        "reduce-axis" => bench::reduce_axis::bench(),
        "silu-like" => bench::silu_like::bench(),
        "layer-norm-like" => bench::layer_norm_like::bench(),
        "reduce-multi-axis" => bench::reduce_multi_axis::bench(),
        "llama-matmul" => bench::llama::matmul::bench(),
        "llama-rmsnorm" => bench::llama::rmsnorm::bench(),
        "llama-rope" => bench::llama::rope::bench(),
        "llama-softmax" => bench::llama::softmax::bench(),
        "llama-swiglu" => bench::llama::swiglu::bench(),
        "llama-embedding" => bench::llama::embedding::bench(),
        "llama-kv-assign" => bench::llama::kv_assign::bench(),
        "llama-attention" => bench::llama::attention::bench(),
        "mma-nt" => bench::mma_nt::bench(),
        _ => panic!("unknown bench `{name}`; `--list` prints every name"),
    }
}
