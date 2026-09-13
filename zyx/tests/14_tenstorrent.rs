// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Golden-shape kernel for the Tenstorrent elementwise lowering.
//!
//! Hand-writes the target IR with the kernel API: a fully tiled elementwise
//! kernel `z = x + sin(y)`. DRAM holds tiles in 16x16-face order (TT's own
//! layout model), so reader/writer move whole tiles with single sequential
//! NOC transfers and no swizzle anywhere; compute works on faces natively.

#![cfg(feature = "tenstorrent")]

use zyx::kernel::{Dev, Kernel, MemScope};
use zyx::{DType, Tensor, ZyxError};

#[test]
fn elementwise_golden_kernel() -> Result<(), ZyxError> {
    const TDIM: u16 = 32;
    const TILE_ELEMS: u16 = TDIM * TDIM;

    let mut k = Kernel::new(Dev::TT(0));
    let x = k.param(DType::BF16);
    let y = k.param(DType::BF16);
    let n_tiles = k.variable(DType::I64);
    let z = k.param_mut(DType::BF16);

    // Circular buffers (like shared memory tiling on CUDA)
    let cx = k.storage(DType::BF16, MemScope::Circular, TILE_ELEMS as i64);
    let cy = k.storage(DType::BF16, MemScope::Circular, TILE_ELEMS as i64);
    let cz = k.storage(DType::BF16, MemScope::Circular, TILE_ELEMS as i64);

    // One group range; every group index owns one tile. Length = n_tiles.
    let g = k.group_range(0, n_tiles);

    // Tile element base = g * 1024
    let tile_elems = k.const_idx(TILE_ELEMS);
    let zero = k.const_idx(0);
    let tbase = k.mad(g, tile_elems, zero);

    // ---- Reader part: whole-tile DRAM -> CB transfers ----
    let tx = k.load_tile(x, tbase, TDIM, TDIM, TDIM as u32);
    k.store_tile(cx, tx, zero, TDIM, TDIM, TDIM as u32);
    let ty = k.load_tile(y, tbase, TDIM, TDIM, TDIM as u32);
    k.store_tile(cy, ty, zero, TDIM, TDIM, TDIM as u32);
    k.barrier();

    // ---- Compute part: z = x + sin(y), faces natively ----
    let ta = k.load_tile(cx, zero, TDIM, TDIM, TDIM as u32);
    let tb = k.load_tile(cy, zero, TDIM, TDIM, TDIM as u32);
    let ts = k.sin(tb);
    let tc = k.add(ta, ts);
    k.store_tile(cz, tc, zero, TDIM, TDIM, TDIM as u32);

    // ---- Writer part: whole-tile CB -> DRAM transfer ----
    k.barrier();
    let v = k.load_tile(cz, zero, TDIM, TDIM, TDIM as u32);
    k.store_tile(z, v, tbase, TDIM, TDIM, TDIM as u32);

    k.verify();
    k.debug();

    let compiled = k.compile()?;

    // Face slot -> linear index within a tile.
    let lin = |s: usize| {
        let (face, local) = (s / 256, s % 256);
        let (fr0, fc0) = (face / 2, face % 2);
        fr0 * 16 * 32 + fc0 * 16 + (local / 16) * 32 + local % 16
    };
    // Encode one tile's linear values into face order for DRAM.
    let tile_encode = |lin_vals: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; 1024];
        for p in 0..1024 {
            out[p] = lin_vals[lin(p)];
        }
        out
    };

    // Launch: four tiles -> n_tiles = 4. x = 1.0 (layout-agnostic);
    // y linear values repeat [0, 64) per tile (bf16-exact).
    let x_t = Tensor::from(vec![1.0f32; 4096]).to(Dev::C)?.cast(DType::BF16).to(Dev::TT(0))?;
    let mut y_dram = Vec::with_capacity(4096);
    for t in 0..4 {
        let lin_vals: Vec<f32> = (0..1024).map(|j| ((t * 1024 + j) % 64) as f32).collect();
        y_dram.extend(tile_encode(&lin_vals));
    }
    let y_t = Tensor::from(y_dram).to(Dev::C)?.cast(DType::BF16).to(Dev::TT(0))?;
    let n_tiles_t = Tensor::variable(4i64);
    let out = compiled.forward(&[&x_t, &y_t, &n_tiles_t], vec![[4096i64]])?;

    let z: Vec<f32> = out[0].to(Dev::C)?.cast(DType::F32).to_vec()?;
    assert_eq!(z.len(), 4096);
    // DRAM position p (tile t, slot s) holds linear z[t*1024 + lin(s)].
    let mut bad = 0;
    for (p, &v) in z.iter().enumerate() {
        let j = p / 1024 * 1024 + lin(p % 1024);
        let expected = 1.0 + ((j % 64) as f32).sin();
        if (v - expected).abs() >= 1e-2 {
            if bad < 20 {
                println!("z[{p}] = {v}, expected {expected}, diff {}", v - expected);
            }
            bad += 1;
        }
    }
    println!("bad: {bad} / 4096");
    assert_eq!(bad, 0);

    Ok(())
}

#[test]
fn tenstorrent_nine_page_read() -> Result<(), ZyxError> {
    const TDIM: u16 = 32;
    const TILE_ELEMS: u16 = TDIM * TDIM;
    const N: i64 = 10;

    let mut k = Kernel::new(Dev::TT(0));
    let x = k.param(DType::BF16);
    let y = k.param(DType::BF16);
    let n_tiles = k.variable(DType::I64);
    let z = k.param_mut(DType::BF16);

    let cx = k.storage(DType::BF16, MemScope::Circular, TILE_ELEMS as i64);
    let cy = k.storage(DType::BF16, MemScope::Circular, TILE_ELEMS as i64);
    let cz = k.storage(DType::BF16, MemScope::Circular, TILE_ELEMS as i64);

    let g = k.group_range(0, n_tiles);
    let tile_elems = k.const_idx(TILE_ELEMS);
    let zero = k.const_idx(0);
    let tbase = k.mad(g, tile_elems, zero);

    let tx = k.load_tile(x, tbase, TDIM, TDIM, TDIM as u32);
    k.store_tile(cx, tx, zero, TDIM, TDIM, TDIM as u32);
    let ty = k.load_tile(y, tbase, TDIM, TDIM, TDIM as u32);
    k.store_tile(cy, ty, zero, TDIM, TDIM, TDIM as u32);
    k.barrier();

    let ta = k.load_tile(cx, zero, TDIM, TDIM, TDIM as u32);
    let tb = k.load_tile(cy, zero, TDIM, TDIM, TDIM as u32);
    let ts = k.sin(tb);
    let tc = k.add(ta, ts);
    k.store_tile(cz, tc, zero, TDIM, TDIM, TDIM as u32);

    k.barrier();
    let v = k.load_tile(cz, zero, TDIM, TDIM, TDIM as u32);
    k.store_tile(z, v, tbase, TDIM, TDIM, TDIM as u32);

    k.verify();
    let compiled = k.compile()?;

    let lin = |s: usize| {
        let (face, local) = (s / 256, s % 256);
        let (fr0, fc0) = (face / 2, face % 2);
        fr0 * 16 * 32 + fc0 * 16 + (local / 16) * 32 + local % 16
    };
    let tile_encode = |lin_vals: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; 1024];
        for p in 0..1024 {
            out[p] = lin_vals[lin(p)];
        }
        out
    };

    let x_t = Tensor::from(vec![1.0f32; 10240]).to(Dev::C)?.cast(DType::BF16).to(Dev::TT(0))?;
    let mut y_dram = Vec::with_capacity(10240);
    for t in 0..N {
        let lin_vals: Vec<f32> = (0..1024).map(|j| ((t * 1024 + j) % 64) as f32).collect();
        y_dram.extend(tile_encode(&lin_vals));
    }
    let y_t = Tensor::from(y_dram).to(Dev::C)?.cast(DType::BF16).to(Dev::TT(0))?;
    let n_tiles_t = Tensor::variable(N);
    let out = compiled.forward(&[&x_t, &y_t, &n_tiles_t], vec![[10240i64]])?;

    let z: Vec<f32> = out[0].to(Dev::C)?.cast(DType::F32).to_vec()?;
    assert_eq!(z.len(), 10240);
    let mut bad = 0;
    for (p, &v) in z.iter().enumerate() {
        let j = p / 1024 * 1024 + lin(p % 1024);
        let expected = 1.0 + ((j % 64) as f32).sin();
        if (v - expected).abs() >= 1e-2 {
            if bad < 20 {
                println!("z[{p}] = {v}, expected {expected}, diff {}", v - expected);
            }
            bad += 1;
        }
    }
    println!("bad: {bad} / 10240");
    assert_eq!(bad, 0);

    Ok(())
}

/// TEMP bisection rung 1: pure dataflow (no reduce). Reader streams 4
/// tiles, compute copies cin->cacc, writer drains. Proves CBs/waits/
/// pops/pushes before the reduce op is suspected.
#[test]
fn tenstorrent_copy_4tile_rung() -> Result<(), ZyxError> {
    const TDIM: u16 = 32;
    const TILE_ELEMS: i64 = 1024;
    const WT: i64 = 4;

    let mut k = Kernel::new(Dev::TT(0));
    let x = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let cin = k.storage(DType::F16, MemScope::Circular, TILE_ELEMS);
    let cacc = k.storage(DType::F16, MemScope::Circular, TILE_ELEMS);

    let _g = k.group_range(0, 1);
    let cwt = k.const_idx(WT);
    let c1024 = k.const_idx(TILE_ELEMS);
    let zero = k.const_idx(0);

    k.loop_over(cwt, |k, ki| {
        let tbase = k.mad(ki, c1024, zero);
        let tx = k.load_tile(x, tbase, TDIM, TDIM, TDIM as u32);
        k.store_tile(cin, tx, zero, TDIM, TDIM, TDIM as u32);
    });
    k.barrier();
    k.loop_over(cwt, |k, _ki| {
        let va = k.load_tile(cin, zero, TDIM, TDIM, TDIM as u32);
        k.store_tile(cacc, va, zero, TDIM, TDIM, TDIM as u32);
    });
    k.barrier();
    k.loop_over(cwt, |k, ki| {
        let tbase = k.mad(ki, c1024, zero);
        let v = k.load_tile(cacc, zero, TDIM, TDIM, TDIM as u32);
        k.store_tile(out, v, tbase, TDIM, TDIM, TDIM as u32);
    });

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let data: Vec<f32> = (0..32 * 128).map(|j| j as f32 * 0.015625).collect();
    let x_t = Tensor::from_vec(data.clone(), [32, 128])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&x_t], vec![[32, 128]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 128)?.to_vec()?;
    assert_eq!(z.len(), 4096);
    let mut bad = 0;
    for (p, (&v, &e)) in z.iter().zip(data.iter()).enumerate() {
        if (v - e).abs() >= 3e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {e}");
            }
            bad += 1;
        }
    }
    println!("rung bad: {bad} / 4096");
    assert_eq!(bad, 0);

    Ok(())
}

/// TEMP bisection rung 2: seed + self-loop acc traffic with plain
/// copies (no reduce). Reader seeds acc, compute overwrites acc with
/// input each iter, writer drains. If green, dataflow is fully
/// exonerated and the hang is in the reduce config sequence.
#[test]
fn tenstorrent_acc_copy_rung() -> Result<(), ZyxError> {
    const TDIM: u16 = 32;
    const TILE_ELEMS: i64 = 1024;
    const WT: i64 = 4;

    let mut k = Kernel::new(Dev::TT(0));
    let x = k.param(DType::F16);
    let m = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let cin = k.storage(DType::F16, MemScope::Circular, TILE_ELEMS);
    let cacc = k.storage(DType::F16, MemScope::Circular, 2 * TILE_ELEMS);
    let cout = k.storage(DType::F16, MemScope::Circular, TILE_ELEMS);

    let _g = k.group_range(0, 1);
    let cwt = k.const_idx(WT);
    let c1024 = k.const_idx(TILE_ELEMS);
    let zero = k.const_idx(0);

    let tm = k.load_tile(m, zero, TDIM, TDIM, TDIM as u32);
    k.store_tile(cacc, tm, zero, TDIM, TDIM, TDIM as u32);
    k.loop_over(cwt, |k, ki| {
        let tbase = k.mad(ki, c1024, zero);
        let tx = k.load_tile(x, tbase, TDIM, TDIM, TDIM as u32);
        k.store_tile(cin, tx, zero, TDIM, TDIM, TDIM as u32);
    });
    k.barrier();
    k.loop_over(cwt, |k, _ki| {
        let va = k.load_tile(cin, zero, TDIM, TDIM, TDIM as u32);
        k.store_tile(cout, va, zero, TDIM, TDIM, TDIM as u32);
        let a = k.load_tile(cacc, zero, TDIM, TDIM, TDIM as u32);
        k.store_tile(cacc, a, zero, TDIM, TDIM, TDIM as u32);
    });
    k.barrier();
    k.loop_over(cwt, |k, ki| {
        let tbase = k.mad(ki, c1024, zero);
        let v = k.load_tile(cout, zero, TDIM, TDIM, TDIM as u32);
        k.store_tile(out, v, tbase, TDIM, TDIM, TDIM as u32);
    });
    let v = k.load_tile(cacc, zero, TDIM, TDIM, TDIM as u32);
    let obase = k.mad(cwt, c1024, zero);
    k.store_tile(out, v, obase, TDIM, TDIM, TDIM as u32);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    // Acc round-trips the seed: expected = min tile everywhere.
    let data: Vec<f32> = (0..32 * 128).map(|j| j as f32 * 0.015625).collect();
    let to_tt = |v: Vec<f32>, rows: i64, cols: i64| -> Result<Tensor, ZyxError> {
        Tensor::from_vec(v, [rows, cols])?.tilize()?.cast(DType::F16).to(Dev::TT(0))
    };
    let x_t = to_tt(data.clone(), 32, 128)?;
    let m_t = to_tt(vec![-65504.0f32; 1024], 32, 32)?;
    let out_bufs = compiled.forward(&[&x_t, &m_t], vec![[160, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(160, 32)?.to_vec()?;
    assert_eq!(z.len(), 5120);
    // Output tile t holds input tile t bitwise: untilized (R, C) in tile
    // row tr = R/32, slot l = (R%32)*32+C maps to input (l/32, tr*32+l%32).
    let mut bad = 0;
    for r in 0..128 {
        for c in 0..32 {
            let tr = r / 32;
            let l = (r % 32) * 32 + c;
            let expected = ((l / 32) * 128 + tr * 32 + l % 32) as f32 * 0.015625;
            let v = z[(r * 32 + c) as usize];
            if (v - expected).abs() >= 3e-2 {
                if bad < 10 {
                    println!("z[{}] = {v}, expected {expected}", r * 32 + c);
                }
                bad += 1;
            }
        }
    }
    for l in 4096..5120 {
        if (z[l] + 65504.0).abs() >= 1.0 {
            if bad < 10 {
                println!("z[{l}] = {}, expected -65504", z[l]);
            }
            bad += 1;
        }
    }
    println!("rung2 bad: {bad} / 5120");
    assert_eq!(bad, 0);

    Ok(())
}

#[test]
fn tenstorrent_row_max_reduce() -> Result<(), ZyxError> {
    const TDIM: u16 = 32;
    const TILE_ELEMS: i64 = 1024;
    const WT: i64 = 4;

    let mut k = Kernel::new(Dev::TT(0));
    let x = k.param(DType::F16);
    let s = k.param(DType::F16);
    let m = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let cin = k.storage(DType::F16, MemScope::Circular, TILE_ELEMS);
    let csc = k.storage(DType::F16, MemScope::Circular, TILE_ELEMS);
    let cout = k.storage(DType::F16, MemScope::Circular, TILE_ELEMS);

    let _g = k.group_range(0, 1);
    let cwt = k.const_idx(WT);
    let cone = k.const_idx(1);
    let c1024 = k.const_idx(TILE_ELEMS);
    let zero = k.const_idx(0);

    // Reader: stream WT input tiles + scaler tiles.
    k.loop_over(cwt, |k, ki| {
        let tbase = k.mad(ki, c1024, zero);
        let tx = k.load_tile(x, tbase, TDIM, TDIM, TDIM as u32);
        k.store_tile(cin, tx, zero, TDIM, TDIM, TDIM as u32);
        let ts = k.load_tile(s, zero, TDIM, TDIM, TDIM as u32);
        k.store_tile(csc, ts, zero, TDIM, TDIM, TDIM as u32);
    });
    k.barrier();

    // Compute: per tile reduce rows, fold running max directly into acc
    // (TT reference shape: reduce_tile accumulates into the acc CB, no temp).
    // Register-scoped acc: load/store are SSA threading, no CB traffic.
    let cacc = k.storage(DType::F16, MemScope::Register, TILE_ELEMS);
    let tm = k.load_tile(m, zero, TDIM, TDIM, TDIM as u32);
    k.store_tile(cacc, tm, zero, TDIM, TDIM, TDIM as u32);
    k.loop_over(cwt, |k, _ki| {
        let va = k.load_tile(cin, zero, TDIM, TDIM, TDIM as u32);
        let vs = k.load_tile(csc, zero, TDIM, TDIM, TDIM as u32);
        let a = k.load_tile(cacc, zero, TDIM, TDIM, TDIM as u32);
        let f = k.reduce_tile(va, vs, a, BOp::Max, TileReduceKind::Row);
        k.store_tile(cacc, f, zero, TDIM, TDIM, TDIM as u32);
    });
    // Epilogue: pack final acc to cout (writer never touches cacc, so no
    // race for the seed: cout gets its only tile after all compute).
    k.loop_over(cone, |k, _ki| {
        let f = k.load_tile(cacc, zero, TDIM, TDIM, TDIM as u32);
        k.store_tile(cout, f, zero, TDIM, TDIM, TDIM as u32);
    });
    k.barrier();

    // Writer: drain cout.
    k.loop_over(cone, |k, _ki| {
        let v = k.load_tile(cout, zero, TDIM, TDIM, TDIM as u32);
        k.store_tile(out, v, zero, TDIM, TDIM, TDIM as u32);
    });

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    // val(r, c) = r*0.5 + c*2^-7: exact in BF16, row max at c=127.
    let data: Vec<f32> = (0..32 * 128)
        .map(|j| {
            let (r, c) = (j / 128, j % 128);
            r as f32 * 0.5 + c as f32 * 0.0078125
        })
        .collect();
    let to_tt = |v: Vec<f32>, rows: i64, cols: i64| -> Result<Tensor, ZyxError> {
        Tensor::from_vec(v, [rows, cols])?.tilize()?.cast(DType::F16).to(Dev::TT(0))
    };
    let x_t = to_tt(data, 32, 128)?;
    let s_t = to_tt(vec![1.0f32; 1024], 32, 32)?;
    let m_t = to_tt(vec![-65504.0f32; 1024], 32, 32)?;
    let out_bufs = compiled.forward(&[&x_t, &s_t, &m_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for c in 0..32 {
        let expected = c as f32 * 0.5 + 0.9921875;
        if (z[c] - expected).abs() >= 3e-2 {
            if bad < 20 {
                println!("z[{c}] = {}, expected {expected}, diff {}", z[c], z[c] - expected);
            }
            bad += 1;
        }
    }
    println!("reduce bad: {bad} / 32");
    assert_eq!(bad, 0);

    Ok(())
}
