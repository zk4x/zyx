// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Multi-core BF16 GEMM for Tenstorrent Blackhole.
//!
//! The output tile grid is sharded over the whole Tensix core grid: core
//! (gx, gy) owns `MT_PER_CORE x NT_PER_CORE` 32x32 output tiles and streams
//! `KT_TILES` K-tiles per output through `matmul_tile` accumulation under
//! the 16-bit DST path (all-BF16, no F32 storage anywhere).
//!
//! Row-stationary reuse (v2): each A tile is pushed once per (mt, kt)
//! and shared across the whole output row (`NT_PER_CORE` matmuls, one
//! live acc cone each) instead of re-streamed per output tile.
//!
//! Inputs are uniform rand in a small range; correctness is checked against
//! a reference matmul on CUDA (C when CUDA is absent). The best of
//! `TIMED_ITERS` launches sets the reported TFLOPS.
//!
//! Run with `./run.sh`. `ZYX_TT_DUMP_ONLY=1` compiles without launching.

use zyx::kernel::autotune::BeamSearch;
use zyx::kernel::{Dev, Kernel, MemScope};
use zyx::{DType, Tensor, ZyxError};

/// 32x32 hardware tile elements.
const TILE_ELEMS: i64 = 1024;
/// K tiles accumulated per output tile (K = KT_TILES * 32).
const KT_TILES: i64 = 32;
/// Absolute tolerance (BF16 accumulation over KT_TILES * 32 terms).
const TOL: f32 = 1.0;

/// Build one fully-prepped GEMM seed: row-stationary reuse, double
/// buffering, `sb_h x sb_w` compute subblock. Returned linearized +
/// scheduled + folded + DCE'd, ready for `BeamSearch::run` (empty
/// optimizations: each seed launches as-is, timed, winner takes all).
fn build_gemm(
    rows: i64,
    cols: i64,
    mt_per_core: i64,
    nt_per_core: i64,
    sb_h: i64,
    sb_w: i64,
    groups_per_strip: i64,
) -> Result<Kernel, ZyxError> {
    let nt = cols * nt_per_core;
    let mt_groups = mt_per_core / sb_h;
    assert_eq!(
        mt_per_core % sb_h,
        0,
        "tt_gemm: MT must split into row groups"
    );
    assert_eq!(
        nt_per_core % sb_w,
        0,
        "tt_gemm: NT must split into subblock cols"
    );
    assert_eq!(
        mt_groups % groups_per_strip,
        0,
        "tt_gemm: row groups must split into strips"
    );
    // Live DST tiles: strip accs + A group + B row + matmul temp.
    // Budget is 16 in BF16 mode.
    assert!(
        groups_per_strip * sb_h * sb_w + sb_h + sb_w + 1 <= 16,
        "tt_gemm: strip exceeds DST budget"
    );

    let mut kernel = Kernel::new(Dev::TT(0));
    let a = kernel.param(DType::BF16);
    let b = kernel.param(DType::BF16);
    let out = kernel.param_mut(DType::BF16);

    // Double-buffered: depth 2 gives the CB FIFOs slack so the reader
    // runs ahead of compute (and compute ahead of the writer) instead
    // of strict lockstep.
    let ca = kernel.circular_storage(DType::BF16, 2);
    let cb = kernel.circular_storage(DType::BF16, 2);
    let cout = kernel.circular_storage(DType::BF16, 2);

    let gx = kernel.group_range(0, rows);
    let gy = kernel.group_range(1, cols);

    // Reader (B-stationary): per (nhi, strip, kti) push the B row once,
    // then the strip's A groups. A tile (mt,kt) at mt*Kt+kt, B tile
    // (kt,nt) at kt*Nt+nt. Each B row is pushed once per strip instead
    // of once per row group (B traffic divided by groups_per_strip).
    // NOTE: barriers are section markers (reader|compute|writer), not
    // runtime sync — cross-section sync is the CB FIFO itself. Exactly
    // two barriers are required.
    // Global row-base of this core's MT block (loop-invariant).
    let mt_base = kernel.mad(gx, mt_per_core, 0);
    let strips = mt_groups / groups_per_strip;
    kernel.loop_over(nt_per_core / sb_w, |kernel, nhi| {
        kernel.loop_over(strips, |kernel, strip| {
            kernel.loop_over(KT_TILES, |kernel, kti| {
                let nt_off = kernel.mad(nhi, sb_w, 0);
                kernel.loop_over(sb_w, |kernel, nti| {
                    let nt_rel = kernel.add(nt_off, nti);
                    let nt_idx = kernel.mad(gy, nt_per_core, nt_rel);
                    let bt = kernel.mad(kti, nt, nt_idx);
                    let bbase = kernel.mad(bt, TILE_ELEMS, 0);
                    let tb = kernel.load_global_tile(b, bbase);
                    kernel.store_circular(cb, tb, 0);
                });
                for gi in 0..groups_per_strip {
                    let mtp = kernel.mad(strip, groups_per_strip, gi);
                    for r in 0..sb_h {
                        let mt_loc = kernel.mad(mtp, sb_h, r);
                        let mt_idx = kernel.add(mt_base, mt_loc);
                        let at = kernel.mad(mt_idx, KT_TILES, kti);
                        let abase = kernel.mad(at, TILE_ELEMS, 0);
                        let ta = kernel.load_global_tile(a, abase);
                        kernel.store_circular(ca, ta, 0);
                    }
                }
            });
        });
    });
    // Compute: per (nhi, strip) accumulate the strip's outputs; one B row
    // feeds every row group in the strip per K step (pop order matches
    // reader push order exactly: B row, then A groups in strip order).
    kernel.barrier();
    kernel.loop_over(nt_per_core / sb_w, |kernel, _nhi| {
        kernel.loop_over(strips, |kernel, _strip| {
            let accs: Vec<_> = (0..groups_per_strip * sb_h * sb_w)
                .map(|_| kernel.storage(DType::BF16, MemScope::Register, TILE_ELEMS))
                .collect();
            kernel.loop_over(KT_TILES, |kernel, _kti| {
                let vb: Vec<_> = (0..sb_w).map(|_| kernel.load_circular(cb, 0)).collect();
                for gi in 0..groups_per_strip as usize {
                    let va: Vec<_> = (0..sb_h).map(|_| kernel.load_circular(ca, 0)).collect();
                    for n in 0..sb_w as usize {
                        for r in 0..sb_h as usize {
                            let acc = accs[gi * (sb_h as usize) * (sb_w as usize)
                                + r * (sb_w as usize)
                                + n];
                            let av = kernel.load_register_tile(acc, 0);
                            let f = kernel.matmul_tile(va[r], vb[n], av);
                            kernel.store_register_tile(acc, f, 0);
                        }
                    }
                }
            });
            for acc in accs {
                let f = kernel.load_register_tile(acc, 0);
                kernel.store_circular(cout, f, 0);
            }
        });
    });
    kernel.barrier();
    // Writer: pops cout in push order (nhi, strip, g, r, n) and scatters
    // to output tile (mt,nt) at mt*Nt+nt, row-major.
    kernel.loop_over(nt_per_core / sb_w, |kernel, nhi| {
        kernel.loop_over(strips, |kernel, strip| {
            for gi in 0..groups_per_strip {
                let mtp = kernel.mad(strip, groups_per_strip, gi);
                for r in 0..sb_h {
                    for n in 0..sb_w {
                        let mt_loc = kernel.mad(mtp, sb_h, r);
                        let mt_idx = kernel.add(mt_base, mt_loc);
                        let nt_base = kernel.mad(nhi, sb_w, 0);
                        let nt_rel = kernel.add(nt_base, n);
                        let nt_idx = kernel.mad(gy, nt_per_core, nt_rel);
                        let ot = kernel.mad(mt_idx, nt, nt_idx);
                        let obase = kernel.mad(ot, TILE_ELEMS, 0);
                        let v = kernel.load_circular(cout, 0);
                        kernel.store_global_tile(out, v, obase);
                    }
                }
            }
        });
    });

    // Custom-kernel prep (mirrors `Kernel::compile` minus codegen and
    // minus `linearize`, a noop here — no reshape/pad/permute ops):
    // BeamSearch launches seeds as-is with empty optimizations.
    kernel.constant_folding();
    kernel.dead_code_elimination();
    kernel.verify();
    Ok(kernel)
}

fn main() -> Result<(), ZyxError> {
    if !Dev::all().iter().any(|d| matches!(d, Dev::TT(0))) {
        println!("tt_gemm: no Tenstorrent device, skipping");
        return Ok(());
    }
    let info = Dev::TT(0).info()?;
    let rows = info.max_global_work_dims[0];
    let cols = info.max_global_work_dims[1];

    let ref_dev = Dev::all()
        .iter()
        .find(|d| matches!(d, Dev::Cuda(_)))
        .copied()
        .unwrap_or(Dev::C);
    println!("tt_gemm: reference device {ref_dev:?}");

    // Search space: per-core tiles x subblock shapes. (mt, nt) change
    // the problem size, so inputs + reference are rebuilt per pair;
    // subblocks only change the kernel. TFLOPS normalizes across sizes.
    let mut best_tflops = 0f64;
    for (mt_per_core, nt_per_core) in [(8, 8), (8, 16), (16, 8), (16, 16), (16, 32), (32, 16)] {
        let m = rows * mt_per_core * 32;
        let n = cols * nt_per_core * 32;
        let k = KT_TILES * 32;

        let a_f32 = Tensor::rand([m, k], DType::F32)? * 0.0625;
        let b_f32 = Tensor::rand([k, n], DType::F32)? * 0.0625;
        let c_ref: Vec<f32> = a_f32
            .clone()
            .to(ref_dev)?
            .matmul(&b_f32.clone().to(ref_dev)?)?
            .to_vec()?;
        let a_t = a_f32.tilize()?.cast(DType::BF16).to(Dev::TT(0))?;
        let b_t = b_f32.tilize()?.cast(DType::BF16).to(Dev::TT(0))?;
        // Scratch output: launch binding only, overwritten by the kernel.
        let out_t = Tensor::zeros([m, n], DType::BF16).to(Dev::TT(0))?;

        let mut seeds = Vec::new();
        let mut names = Vec::new();
        for (sb_h, sb_w, g) in [(1, 4, 1), (2, 4, 1), (4, 2, 1), (2, 2, 2), (1, 4, 2), (1, 2, 4)] {
            seeds.push(build_gemm(
                rows,
                cols,
                mt_per_core,
                nt_per_core,
                sb_h,
                sb_w,
                g,
            )?);
            names.push(format!("MT{mt_per_core}NT{nt_per_core}SB{sb_h}x{sb_w}G{g}"));
        }
        let (winner, nanos) =
            BeamSearch::new().run(seeds, &[&a_t, &b_t, &out_t], &[], |_| {}, |_| 0)?;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let tflops = flops / nanos as f64 / 1e3;
        println!("tt_gemm: M={m} N={n} K={k}: winner {tflops:.3} TFLOPS ({nanos}ns)");
        for name in &names {
            println!("tt_gemm:   seed {name}");
        }

        // Verify the winner against the reference with a single launch.
        // Perf comes from beam search's device-side nanos, not host timing.
        let compiled = winner.compile()?;
        if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
            println!("tt_gemm: dump only, skipping launch");
            return Ok(());
        }
        let outs = compiled.forward(&[&a_t, &b_t], vec![[m, n]])?;
        let z: Vec<f32> = outs[0]
            .to(Dev::C)?
            .cast(DType::F32)
            .untilize(m, n)?
            .to_vec()?;

        assert_eq!(z.len(), c_ref.len());
        let mut bad = 0usize;
        let mut max_err = 0f32;
        for (v, e) in z.iter().zip(c_ref.iter()) {
            let err = (v - e).abs();
            if err > max_err {
                max_err = err;
            }
            if err >= TOL {
                bad += 1;
            }
        }
        println!("tt_gemm: bad {bad} / {}, max err {max_err:.4}", z.len());
        assert!(
            bad == 0,
            "tt_gemm: {bad} mismatches vs reference, max err {max_err}"
        );
        best_tflops = best_tflops.max(tflops);
    }
    println!("tt_gemm: overall best {best_tflops:.3} TFLOPS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // TEMP: revert. Host-side dump (no device).
    #[test]
    fn dump_all() {
        let kernel = build_gemm(10, 11, 8, 8, 1, 4, 1).unwrap();
        let prog = kernel.generate_tenstorrent().unwrap();
        std::fs::write("/tmp/opencode/r.c", &prog.reader_src).unwrap();
        std::fs::write("/tmp/opencode/c.c", &prog.compute_src).unwrap();
        std::fs::write("/tmp/opencode/w.c", &prog.writer_src).unwrap();
    }
}
