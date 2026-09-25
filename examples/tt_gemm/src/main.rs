// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Multi-core BF16 GEMM for Tenstorrent Blackhole.
//!
//! The output tile grid is sharded over the whole Tensix core grid: core
//! (gx, gy) owns `MT_PER_CORE x NT_PER_CORE` 32x32 output tiles and streams
//! `KT_TILES` K-tiles per output through `matmul_tile` accumulation under
//! the 16-bit DST path (all-BF16, no F32 storage anywhere).
//!
//! Inputs are uniform rand in a small range; correctness is checked against
//! a reference matmul on CUDA (C when CUDA is absent). The best of
//! `TIMED_ITERS` launches sets the reported TFLOPS.
//!
//! Run with `./run.sh`. `ZYX_TT_DUMP_ONLY=1` compiles without launching.

use std::time::Instant;

use zyx::kernel::{Dev, Kernel, MemScope};
use zyx::{DType, Tensor, ZyxError};

/// 32x32 hardware tile elements.
const TILE_ELEMS: i64 = 1024;
/// Output tiles owned by one core along M and N.
const MT_PER_CORE: i64 = 4;
const NT_PER_CORE: i64 = 4;
/// K tiles accumulated per output tile (K = KT_TILES * 32).
const KT_TILES: i64 = 32;
/// Timed launches; the best time sets the reported TFLOPS.
const TIMED_ITERS: usize = 10;
/// Absolute tolerance (BF16 accumulation over KT_TILES * 32 terms).
const TOL: f32 = 1.0;

fn main() -> Result<(), ZyxError> {
    if !Dev::all().iter().any(|d| matches!(d, Dev::TT(0))) {
        println!("tt_gemm: no Tenstorrent device, skipping");
        return Ok(());
    }
    let info = Dev::TT(0).info();
    let rows = info.max_global_work_dims[0];
    let cols = info.max_global_work_dims[1];

    let mt = rows * MT_PER_CORE;
    let nt = cols * NT_PER_CORE;
    let m = mt * 32;
    let n = nt * 32;
    let k = KT_TILES * 32;
    println!("tt_gemm: grid {rows}x{cols}, M={m} N={n} K={k} ({MT_PER_CORE}x{NT_PER_CORE} tiles/core)");

    // Small-magnitude rand inputs: keeps BF16 quantization + accumulation
    // error tight against the F32 reference.
    let a_f32 = Tensor::rand([m, k], DType::F32)? * 0.0625;
    let b_f32 = Tensor::rand([k, n], DType::F32)? * 0.0625;

    let ref_dev = Dev::all().iter().find(|d| matches!(d, Dev::Cuda(_))).copied().unwrap_or(Dev::C);
    println!("tt_gemm: reference device {ref_dev:?}");
    let a_ref = a_f32.clone().to(ref_dev)?;
    let b_ref = b_f32.clone().to(ref_dev)?;
    let c_ref: Vec<f32> = a_ref.matmul(&b_ref)?.to_vec()?;

    let a_t = a_f32.tilize()?.cast(DType::BF16).to(Dev::TT(0))?;
    let b_t = b_f32.tilize()?.cast(DType::BF16).to(Dev::TT(0))?;

    let mut kernel = Kernel::new(Dev::TT(0));
    let a = kernel.param(DType::BF16);
    let b = kernel.param(DType::BF16);
    let out = kernel.param_mut(DType::BF16);

    let ca = kernel.circular_storage(DType::BF16, 1);
    let cb = kernel.circular_storage(DType::BF16, 1);
    let cout = kernel.circular_storage(DType::BF16, 1);

    let gx = kernel.group_range(0, rows);
    let gy = kernel.group_range(1, cols);

    // Reader: A tile (mt,kt) at mt*Kt+kt, B tile (kt,nt) at kt*Nt+nt.
    kernel.loop_over(MT_PER_CORE, |kernel, mti| {
        kernel.loop_over(NT_PER_CORE, |kernel, nti| {
            kernel.loop_over(KT_TILES, |kernel, kti| {
                let mt_idx = kernel.mad(gx, MT_PER_CORE, mti);
                let at = kernel.mad(mt_idx, KT_TILES, kti);
                let abase = kernel.mad(at, TILE_ELEMS, 0);
                let ta = kernel.load_global_tile(a, abase);
                kernel.store_circular(ca, ta, 0);
                let nt_idx = kernel.mad(gy, NT_PER_CORE, nti);
                let bt = kernel.mad(kti, nt, nt_idx);
                let bbase = kernel.mad(bt, TILE_ELEMS, 0);
                let tb = kernel.load_global_tile(b, bbase);
                kernel.store_circular(cb, tb, 0);
            });
        });
    });
    kernel.barrier();
    // Compute: one acc cone per output tile, Kt accumulation steps.
    kernel.loop_over(MT_PER_CORE, |kernel, _mti| {
        kernel.loop_over(NT_PER_CORE, |kernel, _nti| {
            let acc = kernel.storage(DType::BF16, MemScope::Register, TILE_ELEMS);
            kernel.loop_over(KT_TILES, |kernel, _kti| {
                let va = kernel.load_circular(ca, 0);
                let vb = kernel.load_circular(cb, 0);
                let av = kernel.load_register_tile(acc, 0);
                let f = kernel.matmul_tile(va, vb, av);
                kernel.store_register_tile(acc, f, 0);
            });
            let f = kernel.load_register_tile(acc, 0);
            kernel.store_circular(cout, f, 0);
        });
    });
    kernel.barrier();
    // Writer: output tile (mt,nt) at mt*Nt+nt, row-major.
    kernel.loop_over(MT_PER_CORE, |kernel, mti| {
        kernel.loop_over(NT_PER_CORE, |kernel, nti| {
            let mt_idx = kernel.mad(gx, MT_PER_CORE, mti);
            let nt_idx = kernel.mad(gy, NT_PER_CORE, nti);
            let ot = kernel.mad(mt_idx, nt, nt_idx);
            let obase = kernel.mad(ot, TILE_ELEMS, 0);
            let v = kernel.load_circular(cout, 0);
            kernel.store_global_tile(out, v, obase);
        });
    });

    kernel.verify();
    let compiled = kernel.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("tt_gemm: dump only, skipping launch");
        return Ok(());
    }

    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let mut best = f64::INFINITY;
    let mut z = Vec::new();
    for _ in 0..TIMED_ITERS {
        let start = Instant::now();
        let outs = compiled.forward(&[&a_t, &b_t], vec![[m, n]])?;
        z = outs[0].to(Dev::C)?.cast(DType::F32).untilize(m, n)?.to_vec()?;
        let dt = start.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    println!("tt_gemm: best {best:.6}s over {TIMED_ITERS} iters = {:.3} TFLOPS", flops / best / 1e12);

    assert_eq!(z.len(), c_ref.len());
    let mut bad = 0usize;
    let mut max_err = 0f32;
    let mut sum_err = 0f64;
    for (v, e) in z.iter().zip(c_ref.iter()) {
        let err = (v - e).abs();
        if err > max_err {
            max_err = err;
        }
        sum_err += f64::from(err);
        if err >= TOL {
            bad += 1;
        }
    }
    println!("tt_gemm: bad {bad} / {}, max err {max_err:.4}, mean err {:.6}", z.len(), sum_err / z.len() as f64);
    assert!(bad == 0, "tt_gemm: {bad} mismatches vs CUDA reference, max err {max_err}");

    Ok(())
}
