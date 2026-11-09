// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Performance regression suite: one name, one dtype, one shape, one device,
//! one time.
//!
//! Separate from `bench.db` on purpose. `bench.db` is the cost-model corpus:
//! every *candidate* kernel with its IR, optimization path and feature vector.
//! This suite stores nothing about kernels — a row is
//! `(name, dtype, shape, device, nanos, code_version)` — and the number is the
//! wall-clock time of the whole operation (device-synchronized), not any
//! single launch's timer.
//!
//! ```text
//! zyx-bench perf           # measure, compare with the last saved run, fail on a regression
//! zyx-bench perf --save    # measure and make this run the baseline
//! zyx-bench perf --tol 5   # regression threshold in percent (default 10)
//! ```
//!
//! A regression is an error, so this exits non-zero: a suite that reports a
//! slowdown and still passes is the thing this exists to prevent.

use std::path::PathBuf;
use std::time::Instant;

use rusqlite::Connection;
use zyx::{DType, Dev, Tensor, ZyxError};

use crate::bench::bench_dev;

/// Warmup executions before any timing: the first force of a shape compiles
/// and autotunes it, which is not the operation's cost.
const WARMUP: usize = 3;
/// Timed batches; the best batch wins, so one another-process hiccup does not
/// become the number.
const TRIALS: usize = 5;
/// Executions per batch.
const REPS: usize = 20;
/// Regression threshold in percent, when `--tol` is absent.
pub(crate) const DEFAULT_TOL: f64 = 10.0;

/// One measured operation: inputs are created once, the op is rebuilt per
/// execution (lazy graph construction only; the compiled kernel is cached).
pub(crate) struct Case {
    /// Operation name, e.g. `matmul/prefill/q_proj`.
    pub name: &'static str,
    /// Operand dtype, e.g. `f16`.
    pub dtype: &'static str,
    /// Shape label, e.g. `[128x3072]x[3072x3072]`.
    pub shape: String,
    /// Operands, already on the device.
    pub inputs: Vec<Tensor>,
    /// The operation itself, built from `inputs`.
    pub op: fn(&[Tensor]) -> Result<Tensor, ZyxError>,
}

fn matmul(i: &[Tensor]) -> Result<Tensor, ZyxError> {
    i[0].matmul(i[1].t())
}

fn swiglu(i: &[Tensor]) -> Result<Tensor, ZyxError> {
    use std::ops::Mul;
    let g = i[0].matmul(i[1].t())?;
    let u = i[0].matmul(i[2].t())?;
    g.swish().mul(&u).matmul(i[3].t())
}

fn rmsnorm(i: &[Tensor]) -> Result<Tensor, ZyxError> {
    use std::ops::{Add, Div, Mul};
    let x = &i[0];
    let mean_sq = x.clone().mul(x).mean([-1])?.unsqueeze(-1)?;
    let norm = x.div(&mean_sq.add(Tensor::from(1e-5f32)).rsqrt());
    Ok(norm.mul(&i[1]))
}

fn rope(i: &[Tensor]) -> Result<Tensor, ZyxError> {
    i[0].rope(i[1].clone(), i[2].clone())
}

fn softmax(i: &[Tensor]) -> Result<Tensor, ZyxError> {
    i[0].softmax([-1])
}

fn attention(i: &[Tensor]) -> Result<Tensor, ZyxError> {
    use std::ops::Mul;
    let n_rep = 3;
    let k = crate::bench::llama::shapes::repeat_kv(i[1].clone(), n_rep)?;
    let v = crate::bench::llama::shapes::repeat_kv(i[2].clone(), n_rep)?;
    let scores = i[0].matmul(k.transpose(2, 3)?)?.mul(&i[3]);
    scores.softmax([-1])?.matmul(v)
}

fn embedding(i: &[Tensor]) -> Result<Tensor, ZyxError> {
    i[0].index_select(0, i[1].clone())
}

fn kv_assign(i: &[Tensor]) -> Result<Tensor, ZyxError> {
    let shape = i[1].shape();
    let seq = shape[0].item::<i64>();
    i[0].narrow(0, 0i64, seq)?.assign(&i[1])?;
    i[0].narrow(0, 0i64, seq)
}

/// The manifest: llama-3.2-3B shapes, f16, exactly the operations the
/// `llama/*` benches time, at the sizes `BENCHMARKS.md` reports.
///
/// `llama e2e` is absent: it needs the GGUF checkpoint, which is a manual,
/// on-demand run, not a regression gate.
fn manifest(dev: Dev) -> Result<Vec<Case>, ZyxError> {
    use crate::bench::llama::shapes::{
        CTX, EPS, HEAD_DIM, HEADS, HIDDEN, INTERMEDIATE, KV_HEADS, SCALE, SEQ, VOCAB,
    };

    let mut cases = Vec::new();
    let mut add = |name: &'static str, shape: String, inputs: Vec<Tensor>, op: fn(&[Tensor]) -> Result<Tensor, ZyxError>| {
        cases.push(Case { name, dtype: "f16", shape, inputs, op });
    };

    // --- projections, both phases (q/o share a shape; both are reported) ---
    const PROJS: [(&str, i64, i64); 7] = [
        ("q_proj", HIDDEN, HEADS * HEAD_DIM),
        ("k_proj", HIDDEN, KV_HEADS * HEAD_DIM),
        ("v_proj", HIDDEN, KV_HEADS * HEAD_DIM),
        ("o_proj", HEADS * HEAD_DIM, HIDDEN),
        ("gate_proj", HIDDEN, INTERMEDIATE),
        ("up_proj", HIDDEN, INTERMEDIATE),
        ("down_proj", INTERMEDIATE, HIDDEN),
    ];
    for (phase, seq) in [("prefill", SEQ), ("decode", 1)] {
        for (name, k, n) in PROJS {
            let x = Tensor::rand([seq, k], DType::F16)?.to(dev)?;
            let w = Tensor::rand([n, k], DType::F16)?.to(dev)?;
            add(
                Box::leak(format!("matmul/{phase}/{name}").into_boxed_str()),
                format!("[{seq}x{k}]x[{k}x{n}]"),
                vec![x, w],
                matmul,
            );
        }
    }
    {
        let x = Tensor::rand([1i64, HIDDEN], DType::F16)?.to(dev)?;
        let w = Tensor::rand([VOCAB, HIDDEN], DType::F16)?.to(dev)?;
        add("matmul/decode/lm_head", format!("[1x{HIDDEN}]x[{HIDDEN}x{VOCAB}]"), vec![x, w], matmul);
    }

    // --- swiglu: `down(silu(gate(x)) * up(x))` ---
    for seq in [1i64, SEQ] {
        let x = Tensor::rand([seq, HIDDEN], DType::F16)?.to(dev)?;
        let gate = Tensor::rand([INTERMEDIATE, HIDDEN], DType::F16)?.to(dev)?;
        let up = Tensor::rand([INTERMEDIATE, HIDDEN], DType::F16)?.to(dev)?;
        let down = Tensor::rand([HIDDEN, INTERMEDIATE], DType::F16)?.to(dev)?;
        add(
            Box::leak(format!("swiglu/{}", if seq == 1 { "decode" } else { "prefill" }).into_boxed_str()),
            format!("[1x{seq}x{HIDDEN}]"),
            vec![x, gate, up, down],
            swiglu,
        );
    }

    // --- rmsnorm ---
    for seq in [1i64, SEQ] {
        let x = Tensor::rand([1, seq, HIDDEN], DType::F16)?.to(dev)?;
        let w = Tensor::rand([HIDDEN], DType::F16)?.to(dev)?;
        add(
            Box::leak(format!("rmsnorm/{}", if seq == 1 { "decode" } else { "prefill" }).into_boxed_str()),
            format!("[1x{seq}x{HIDDEN}]"),
            vec![x, w],
            rmsnorm,
        );
    }
    let _ = EPS;

    // --- rope ---
    for seq in [1i64, SEQ] {
        let x = Tensor::rand([1, HEADS, seq, HEAD_DIM], DType::F16)?.to(dev)?;
        let cos = Tensor::rand([CTX, HEAD_DIM / 2], DType::F16)?.to(dev)?;
        let sin = Tensor::rand([CTX, HEAD_DIM / 2], DType::F16)?.to(dev)?;
        let cos = cos.narrow(0, 0i64, seq)?;
        let sin = sin.narrow(0, 0i64, seq)?;
        add(
            Box::leak(format!("rope/{}", if seq == 1 { "decode" } else { "prefill" }).into_boxed_str()),
            format!("[1x{HEADS}x{seq}x{HEAD_DIM}]"),
            vec![x, cos, sin],
            rope,
        );
    }

    // --- softmax over the cache axis ---
    for (seq, cache) in [(1i64, 1i64), (1, SEQ), (SEQ, SEQ)] {
        let x = Tensor::rand([HEADS, seq, cache], DType::F16)?.to(dev)?;
        add(
            Box::leak(format!("softmax/[q{seq}xc{cache}]").into_boxed_str()),
            format!("[{HEADS}x{seq}x{cache}]"),
            vec![x],
            softmax,
        );
    }

    // --- gqa attention ---
    for (seq, cache) in [(1i64, 1i64), (1, SEQ), (SEQ, SEQ)] {
        let q = Tensor::rand([1, HEADS, seq, HEAD_DIM], DType::F16)?.to(dev)?;
        let k = Tensor::rand([1, KV_HEADS, cache, HEAD_DIM], DType::F16)?.to(dev)?;
        let v = Tensor::rand([1, KV_HEADS, cache, HEAD_DIM], DType::F16)?.to(dev)?;
        let scale = Tensor::from(SCALE).cast(DType::F16).to(dev)?;
        add(
            Box::leak(format!("attention/[q{seq}xc{cache}]").into_boxed_str()),
            format!("q[1x{HEADS}x{seq}x{HEAD_DIM}] kv[1x{KV_HEADS}x{cache}x{HEAD_DIM}]"),
            vec![q, k, v, scale],
            attention,
        );
    }

    // --- embedding gather ---
    {
        let w = Tensor::rand([VOCAB, HIDDEN], DType::F16)?.to(dev)?;
        for seq in [1i64, SEQ] {
            // `index_select` takes a 1-D index; the module's `[1, seq]` ids are
            // the same gather with a leading batch axis.
            let idx = Tensor::from(vec![0u32; seq as usize]).to(dev)?;
            add(
                Box::leak(format!("embedding/{}", if seq == 1 { "decode" } else { "prefill" }).into_boxed_str()),
                format!("[{seq}] of [{VOCAB}x{HIDDEN}]"),
                vec![w.clone(), idx],
                embedding,
            );
        }
    }

    // --- kv cache write ---
    for seq in [1i64, SEQ] {
        let cache = Tensor::zeros([CTX, KV_HEADS, HEAD_DIM], DType::F16).to(dev)?;
        let k = Tensor::rand([seq, KV_HEADS, HEAD_DIM], DType::F16)?.to(dev)?;
        add(
            Box::leak(format!("kv_assign/{}", if seq == 1 { "decode" } else { "prefill" }).into_boxed_str()),
            format!("[{seq}x{KV_HEADS}x{HEAD_DIM}] into [{CTX}x..]"),
            vec![cache, k],
            kv_assign,
        );
    }

    Ok(cases)
}

/// `zyx-bench/perf.db`: the baselines. No kernel IR, no optimization path —
/// only what the suite compares.
fn db_path() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("perf.db")
}

fn open_db(path: &PathBuf) -> rusqlite::Result<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         CREATE TABLE IF NOT EXISTS runs (
             id           INTEGER PRIMARY KEY AUTOINCREMENT,
             ts           INTEGER NOT NULL,
             code_version TEXT    NOT NULL,
             dev          TEXT    NOT NULL
         );
         CREATE TABLE IF NOT EXISTS perf (
             run_id INTEGER NOT NULL,
             name   TEXT    NOT NULL,
             dtype  TEXT    NOT NULL,
             shape  TEXT    NOT NULL,
             dev    TEXT    NOT NULL,
             nanos  INTEGER NOT NULL
         );",
    )?;
    Ok(conn)
}

/// Force the graph with a device-synchronized read of the tensor's first
/// element, in the tensor's own dtype: a cast to f32 would add a full-size
/// elementwise kernel to every timed execution.
fn force_element(t: &Tensor) -> Result<(), ZyxError> {
    match t.dtype() {
        DType::F16 => {
            let _ = t.item::<zyx::f16>();
        }
        DType::F32 => {
            let _ = t.item::<f32>();
        }
        other => todo!("perf: forcing a {other} tensor"),
    }
    Ok(())
}

/// Time `case.op` end to end: build the lazy graph, force it with a
/// device-synchronized read of one element, best of [`TRIALS`] batches.
fn measure(case: &Case) -> Result<u64, ZyxError> {
    for _ in 0..WARMUP {
        force_element(&(case.op)(&case.inputs)?)?;
    }
    let mut best = u64::MAX;
    for _ in 0..TRIALS {
        let start = Instant::now();
        for _ in 0..REPS {
            force_element(&(case.op)(&case.inputs)?)?;
        }
        let nanos = u64::try_from(start.elapsed().as_nanos() / REPS as u128).unwrap_or(u64::MAX);
        best = best.min(nanos);
    }
    Ok(best)
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// The code version of the tree that built this binary, exactly as the
/// measurement recorder labels `bench.db` rows.
fn code_version() -> String {
    use std::process::Command;
    let git = |args: &[&str]| -> Option<String> {
        Command::new("git")
            .args(args)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let commit = git(&["rev-parse", "--short=8", "HEAD"]).filter(|commit| !commit.is_empty());
    match commit {
        None => "unknown".to_string(),
        Some(commit) => match git(&["status", "--porcelain"]) {
            Some(status) if !status.is_empty() => format!("{commit}+dirty"),
            _ => commit,
        },
    }
}

fn ms(nanos: u64) -> f64 {
    nanos as f64 / 1e6
}

/// Run the suite. `save` makes this run the new baseline; otherwise the run is
/// compared with the most recent saved baseline on the same device and the
/// function fails when any operation regressed by more than `tol` percent.
pub(crate) fn run(save: bool, tol: f64) -> Result<(), ZyxError> {
    let dev = bench_dev();
    let dev_label = format!("{dev:?}");
    let version = code_version();
    println!("perf suite: {dev_label}, {version}, tolerance {tol:.1}%");

    let cases = manifest(dev)?;
    let mut measured: Vec<(String, String, String, u64)> = Vec::with_capacity(cases.len());
    for case in &cases {
        let nanos = measure(case)?;
        println!(
            "{:38} {:4} {:40} {:10.4} ms",
            case.name,
            case.dtype,
            case.shape,
            ms(nanos)
        );
        measured.push((case.name.to_string(), case.dtype.to_string(), case.shape.clone(), nanos));
    }

    let path = db_path();
    let conn = open_db(&path).map_err(|e| ZyxError::kernel_error(e.to_string().into()))?;

    // Baseline: the newest saved run on this device.
    let baseline: Option<(i64, String)> = {
        let mut stmt = conn
            .prepare("SELECT id, code_version FROM runs WHERE dev = ?1 ORDER BY id DESC LIMIT 1")
            .map_err(|e| ZyxError::kernel_error(e.to_string().into()))?;
        let mut rows = stmt
            .query([&dev_label])
            .map_err(|e| ZyxError::kernel_error(e.to_string().into()))?;
        let row = rows.next().map_err(|e| ZyxError::kernel_error(e.to_string().into()))?;
        match row {
            Some(row) => Some((
                row.get(0).map_err(|e| ZyxError::kernel_error(e.to_string().into()))?,
                row.get(1).map_err(|e| ZyxError::kernel_error(e.to_string().into()))?,
            )),
            None => None,
        }
    };

    let mut regressions: Vec<String> = Vec::new();
    if let Some((run_id, base_version)) = &baseline {
        println!("\nagainst run #{run_id} ({base_version}):");
        let mut stmt = conn
            .prepare("SELECT nanos FROM perf WHERE run_id = ?1 AND name = ?2 AND shape = ?3 AND dtype = ?4")
            .map_err(|e| ZyxError::kernel_error(e.to_string().into()))?;
        for (name, dtype, shape, nanos) in &measured {
            let base: Option<i64> = stmt
                .query_row(rusqlite::params![run_id, name, shape, dtype], |row| row.get(0))
                .ok();
            let Some(base) = base else {
                println!("{:38} {:4} {:40} no baseline", name, dtype, shape);
                continue;
            };
            let base = base.max(1) as f64;
            let change = ((*nanos).max(1) as f64 - base) / base * 100.0;
            let flag = if change > tol { "REGRESSION" } else { "" };
            println!(
                "{:38} {:4} {:40} {:9.4} ms (was {:9.4}, {:+.1}%) {}",
                name,
                dtype,
                shape,
                ms(*nanos),
                ms(base as u64),
                change,
                flag
            );
            if change > tol {
                regressions.push(format!("{name} {} {shape}: {change:+.1}%", dtype));
            }
        }
    } else {
        println!("\nno baseline for {dev_label}: run `zyx-bench perf --save` to record one");
    }

    if save {
        let tx = conn.unchecked_transaction().map_err(|e| ZyxError::kernel_error(e.to_string().into()))?;
        tx.execute(
            "INSERT INTO runs (ts, code_version, dev) VALUES (?1, ?2, ?3)",
            rusqlite::params![now(), version, dev_label],
        )
        .map_err(|e| ZyxError::kernel_error(e.to_string().into()))?;
        let run_id = tx.last_insert_rowid();
        {
            let mut stmt = tx
                .prepare("INSERT INTO perf (run_id, name, dtype, shape, dev, nanos) VALUES (?1, ?2, ?3, ?4, ?5, ?6)")
                .map_err(|e| ZyxError::kernel_error(e.to_string().into()))?;
            for (name, dtype, shape, nanos) in &measured {
                stmt.execute(rusqlite::params![run_id, name, dtype, shape, dev_label, *nanos as i64])
                    .map_err(|e| ZyxError::kernel_error(e.to_string().into()))?;
            }
        }
        tx.commit().map_err(|e| ZyxError::kernel_error(e.to_string().into()))?;
        println!("\nsaved run #{run_id} to {}", path.display());
    }

    if regressions.is_empty() {
        Ok(())
    } else {
        Err(ZyxError::kernel_error(
            format!("{} performance regression(s) over {tol:.1}%:\n  {}", regressions.len(), regressions.join("\n  ")).into(),
        ))
    }
}
