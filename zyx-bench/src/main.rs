// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Benchmarks for zyx tensor library.
//!
//! Each benchmark is a logical `(folder, test)` pair in [`BENCHES`]; the
//! source module is irrelevant to its name. Exactly one runs per invocation:
//!
//! ```text
//! cargo run --release --bin zyx-bench -- --list
//! cargo run --release --bin zyx-bench -- llama prefill_gemm
//! ```
//!
//! Measurements go straight into `zyx-bench/bench.db`, next to this crate:
//! there are no environment variables, no temporary streams and no side
//! directories. Each row carries the candidate's optimization path and the
//! stored IR, so `zyx-bench reindex` can recompute the cost features without
//! running anything again.

#![allow(unused)]

mod bench;

use std::path::{Path, PathBuf};

use rusqlite::Connection;
use zyx::ZyxError;
use zyx::kernel::Features;
use zyx::kernel::autotune::{Measurement, set_measurement_recorder};

// ---- benchmark registry ----

/// One benchmark: a logical `(folder, test)` name and the function that runs
/// it. The file layout is not the API.
struct Bench {
    folder: &'static str,
    name: &'static str,
    run: fn() -> Result<(), ZyxError>,
}

/// Every benchmark, runnable individually as `<folder> <test>`.
const BENCHES: &[Bench] = &[
    Bench { folder: "llama", name: "prefill_gemm", run: bench::llama::matmul::prefill_gemm },
    Bench { folder: "llama", name: "decode_gemm", run: bench::llama::matmul::decode_gemm },
    Bench { folder: "llama", name: "rmsnorm", run: bench::llama::rmsnorm::bench },
    Bench { folder: "llama", name: "rope", run: bench::llama::rope::bench },
    Bench { folder: "llama", name: "softmax", run: bench::llama::softmax::bench },
    Bench { folder: "llama", name: "swiglu", run: bench::llama::swiglu::bench },
    Bench { folder: "llama", name: "embedding", run: bench::llama::embedding::bench },
    Bench { folder: "llama", name: "kv_assign", run: bench::llama::kv_assign::bench },
    Bench { folder: "llama", name: "attention", run: bench::llama::attention::bench },
    Bench { folder: "llama", name: "e2e", run: bench::llama::e2e::e2e },
    Bench { folder: "zyx", name: "large_matmul", run: bench::large_matmul::bench },
    Bench { folder: "zyx", name: "matmul", run: bench::matmul::bench },
    Bench { folder: "zyx", name: "reduce", run: bench::reduce::bench },
    Bench { folder: "zyx", name: "softmax", run: bench::softmax::bench },
    Bench { folder: "zyx", name: "embedding", run: bench::embedding::bench },
    Bench { folder: "zyx", name: "gelu", run: bench::gelu::bench },
    Bench { folder: "zyx", name: "activation", run: bench::activation::bench },
    Bench { folder: "zyx", name: "ln_softmax", run: bench::ln_softmax::bench },
    Bench { folder: "zyx", name: "reduce_axis", run: bench::reduce_axis::bench },
    Bench { folder: "zyx", name: "silu_like", run: bench::silu_like::bench },
    Bench { folder: "zyx", name: "layer_norm_like", run: bench::layer_norm_like::bench },
    Bench { folder: "zyx", name: "reduce_multi_axis", run: bench::reduce_multi_axis::bench },
    Bench { folder: "zyx", name: "mma_nt", run: bench::mma_nt::bench },
];

// ---- measurement database ----

/// The database lives next to this crate, always: `zyx-bench/bench.db`.
fn db_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("bench.db")
}

/// Open (and create on first use) the measurement database.
///
/// Rows are never updated or deleted: repeating a run appends duplicates on
/// purpose (more samples per config). `kernels` is content-addressed by hash
/// and `kernel_features` holds the latest recomputed features.
fn open_db(path: &Path) -> rusqlite::Result<Connection> {
    let conn = Connection::open(path)?;
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
             grid     TEXT,
             n_launches INTEGER,
             code_version TEXT,
             status   TEXT    NOT NULL,
             features TEXT    NOT NULL
         );
         CREATE TABLE IF NOT EXISTS kernels (
             hash     TEXT PRIMARY KEY,
             ir       BLOB NOT NULL
         );
         CREATE TABLE IF NOT EXISTS kernel_features (
             hash     TEXT PRIMARY KEY,
             features TEXT NOT NULL
         );",
    )?;
    // `grid`, `n_launches` and `code_version` were added after the first
    // databases existed, and `CREATE TABLE IF NOT EXISTS` leaves those
    // untouched: add each column once. Old rows keep `NULL` — those values
    // were never recorded, and no value can be invented for them.
    for column in ["grid TEXT", "n_launches INTEGER", "code_version TEXT"] {
        let name = column.split(' ').next().unwrap();
        let present: i64 = conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('measurements') WHERE name = ?1",
            [name],
            |row| row.get(0),
        )?;
        if present == 0 {
            conn.execute(&format!("ALTER TABLE measurements ADD COLUMN {column}"), [])?;
        }
    }
    Ok(conn)
}

fn features_json(features: &Features) -> String {
    let values = features.values();
    let mut json = String::with_capacity(768);
    json.push('{');
    for (i, name) in Features::NAMES.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!("\"{name}\":{}", values[i]));
    }
    json.push('}');
    json
}

fn seq_json(seq: &[(usize, u64)]) -> String {
    let mut json = String::from("[");
    for (i, (optimization, config)) in seq.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!("[{optimization},{config}]"));
    }
    json.push(']');
    json
}

/// Install the recorder that writes every measured candidate straight into
/// `bench.db`: the measurement row, plus the kernel IR (once per hash).
fn install_recorder(conn: Connection, tag: String, code_version: String) {
    let mut seen = std::collections::HashSet::<u64>::new();
    set_measurement_recorder(Some(Box::new(move |m: &Measurement| {
        let hash = format!("{:016x}", m.kernel_hash);
        if seen.insert(m.kernel_hash) {
            if let Err(e) = conn.execute(
                "INSERT OR IGNORE INTO kernels (hash, ir) VALUES (?1, ?2)",
                rusqlite::params![hash, m.ir],
            ) {
                eprintln!("zyx-bench: storing kernel {hash}: {e}");
            }
        }
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        if let Err(e) = conn.execute(
            "INSERT INTO measurements
                 (ts, schema, tag, dev, cc0, cc1, seed, kernel, seq, nanos, grid, n_launches, code_version, status, features)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            rusqlite::params![
                ts,
                2i64,
                tag,
                m.dev,
                m.cc[0],
                m.cc[1],
                format!("{:016x}", m.seed),
                hash,
                seq_json(&m.seq),
                m.nanos,
                m.grid.map(|g| format!("[{},{},{}]", g[0], g[1], g[2])),
                i64::try_from(m.n_launches).unwrap_or(i64::MAX),
                code_version,
                m.status,
                features_json(&m.features),
            ],
        ) {
            eprintln!("zyx-bench: storing measurement: {e}");
        }
    })));
}

/// Recompute `features()` for every stored kernel and refresh
/// `kernel_features`.
///
/// The feature vector is a pure function of the kernel IR and its
/// `DeviceInfo` snapshot, so a feature or cost-model change applies to all
/// historical rows without re-running a single launch. No device is opened.
fn reindex(db: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use nanoserde::DeBin;
    use zyx::kernel::Kernel;

    let mut conn = open_db(db)?;
    let kernels: Vec<(String, Vec<u8>)> = {
        let mut stmt = conn.prepare("SELECT hash, ir FROM kernels")?;
        let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        let out = rows.collect::<Result<_, _>>()?;
        out
    };
    let total = kernels.len();
    let mut failed = 0usize;
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare("INSERT OR REPLACE INTO kernel_features (hash, features) VALUES (?1, ?2)")?;
        for (hash, ir) in kernels {
            let kernel = match Kernel::deserialize_bin(&ir) {
                Ok(kernel) => kernel,
                Err(e) => {
                    failed += 1;
                    eprintln!("reindex: kernel {hash}: {e:?}");
                    continue;
                }
            };
            stmt.execute(rusqlite::params![hash, features_json(&kernel.features())])?;
        }
    }
    tx.commit()?;
    println!("reindexed {} kernels ({failed} failed to decode)", total - failed);
    Ok(())
}

fn main() -> Result<(), ZyxError> {
    let mut args = std::env::args().skip(1);
    let Some(folder) = args.next() else {
        eprintln!("usage: zyx-bench <folder> <test>\n       zyx-bench --list\n       zyx-bench reindex");
        std::process::exit(2);
    };
    if folder == "--list" {
        use std::io::Write;
        let mut out = std::io::stdout();
        for bench in BENCHES {
            // A closed pipe (`--list | head`) is a normal end, not a panic.
            if writeln!(out, "{} {}", bench.folder, bench.name).is_err() {
                return Ok(());
            }
        }
        return Ok(());
    }
    if folder == "reindex" {
        // `zyx-bench reindex <hash>` prints one stored kernel's IR instead of
        // refreshing features: the exact artifact of a recorded measurement,
        // decodable without a device.
        if let Some(hash) = args.next() {
            use nanoserde::DeBin;
            use zyx::kernel::Kernel;
            let conn = open_db(&db_path()).map_err(|e| ZyxError::kernel_error(e.to_string().into()))?;
            let ir: Vec<u8> = conn
                .query_row("SELECT ir FROM kernels WHERE hash = ?1", [&hash], |row| row.get(0))
                .map_err(|e| ZyxError::kernel_error(format!("no kernel {hash}: {e}").into()))?;
            let kernel = Kernel::deserialize_bin(&ir)
                .map_err(|e| ZyxError::kernel_error(format!("kernel {hash}: {e:?}").into()))?;
            println!("kernel {hash}");
            kernel.debug();
            return Ok(());
        }
        return reindex(&db_path()).map_err(|e| ZyxError::kernel_error(e.to_string().into()));
    }
    let Some(name) = args.next() else {
        eprintln!("zyx-bench: missing test name; one benchmark runs per invocation\n       zyx-bench {folder} <test>");
        std::process::exit(2);
    };
    let Some(bench) = BENCHES.iter().find(|b| b.folder == folder && b.name == name) else {
        eprintln!("zyx-bench: unknown benchmark `{folder} {name}`; `--list` prints every name");
        std::process::exit(2);
    };
    let tag = format!("{folder}/{name}");
    // The commit this binary was built from: rows from different trees must
    // stay separable. Resolved at run time so a stale binary is caught by the
    // tree it reports; `unknown` when git or the checkout is absent. A dirty
    // tree is marked, because the commit alone would then name code that was
    // never the code that ran.
    let code_version = {
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
    };
    let conn = open_db(&db_path()).map_err(|e| ZyxError::kernel_error(e.to_string().into()))?;
    install_recorder(conn, tag, code_version);
    let result = (bench.run)();
    set_measurement_recorder(None);
    result
}
