// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Golden-shape kernel for the Tenstorrent elementwise lowering.
//!
//! Hand-writes the target IR with the kernel API: a fully tiled elementwise
//! kernel `z = x + sin(y)`. DRAM holds tiles in 16x16-face order (TT's own
//! layout model), so reader/writer move whole tiles with single sequential
//! NOC transfers and no swizzle anywhere; compute works on faces natively.

//#![cfg(feature = "tenstorrent")]

use zyx::kernel::{BOp, Dev, Kernel, MemScope, TileReduceKind};
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
    let cx = k.circular_storage(DType::BF16, 1);
    let cy = k.circular_storage(DType::BF16, 1);
    let cz = k.circular_storage(DType::BF16, 1);

    // One group range; every group index owns one tile. Length = n_tiles.
    let g = k.group_range(0, n_tiles);

    // Tile element base = g * 1024
    let tile_elems = k.const_idx(TILE_ELEMS);
    let zero = k.const_idx(0);
    let tbase = k.mad(g, tile_elems, zero);

    // ---- Reader part: whole-tile DRAM -> CB transfers ----
    let tx = k.load_global_tile(x, tbase);
    k.store_circular(cx, tx, zero);
    let ty = k.load_global_tile(y, tbase);
    k.store_circular(cy, ty, zero);
    k.barrier();

    // ---- Compute part: z = x + sin(y), faces natively ----
    let ta = k.load_circular(cx, zero);
    let tb = k.load_circular(cy, zero);
    let ts = k.sin(tb);
    let tc = k.add(ta, ts);
    k.store_circular(cz, tc, zero);

    // ---- Writer part: whole-tile CB -> DRAM transfer ----
    k.barrier();
    let v = k.load_circular(cz, zero);
    k.store_global_tile(z, v, tbase);

    k.verify();
    k.debug();

    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

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

    let cx = k.circular_storage(DType::BF16, 1);
    let cy = k.circular_storage(DType::BF16, 1);
    let cz = k.circular_storage(DType::BF16, 1);

    let g = k.group_range(0, n_tiles);
    let tile_elems = k.const_idx(TILE_ELEMS);
    let zero = k.const_idx(0);
    let tbase = k.mad(g, tile_elems, zero);

    let tx = k.load_global_tile(x, tbase);
    k.store_circular(cx, tx, zero);
    let ty = k.load_global_tile(y, tbase);
    k.store_circular(cy, ty, zero);
    k.barrier();

    let ta = k.load_circular(cx, zero);
    let tb = k.load_circular(cy, zero);
    let ts = k.sin(tb);
    let tc = k.add(ta, ts);
    k.store_circular(cz, tc, zero);

    k.barrier();
    let v = k.load_circular(cz, zero);
    k.store_global_tile(z, v, tbase);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

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

/// Simple copy: reader streams tiles, compute copies cin->cout,
/// writer drains. Covers CB dataflow with honest names.
#[test]
fn tenstorrent_copy() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let x = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let cin = k.circular_storage(DType::F16, 1);
    let cout = k.circular_storage(DType::F16, 1);

    let _g = k.group_range(0, 1);

    k.loop_over(4, |k, ki| {
        let tbase = k.mad(ki, 1024, 0);
        let tx = k.load_global_tile(x, tbase);
        k.store_circular(cin, tx, 0);
    });
    k.barrier();
    k.loop_over(4, |k, _ki| {
        let va = k.load_circular(cin, 0);
        k.store_circular(cout, va, 0);
    });
    k.barrier();
    k.loop_over(4, |k, ki| {
        let tbase = k.mad(ki, 1024, 0);
        let v = k.load_circular(cout, 0);
        k.store_global_tile(out, v, tbase);
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
    println!("copy bad: {bad} / 4096");
    assert_eq!(bad, 0);

    Ok(())
}

/// Pad-style move, mirroring `pad_move_tt`: F32 tiles, empty compute
/// section (two barriers back-to-back). Reader streams WT tiles,
/// writer drains them. Passes iff reader/writer/CB/DRAM paths are
/// correct; also exercises the 8-tile FP32 DST compiler path in-tree.
#[test]
fn tenstorrent_pad_move() -> Result<(), ZyxError> {
    const TILE_ELEMS: i64 = 1024;
    const WT: i64 = 4;

    let mut k = Kernel::new(Dev::TT(0));
    let x = k.param(DType::F32);
    let out = k.param_mut(DType::F32);

    let cdata = k.circular_storage(DType::F32, 1);

    let _g = k.group_range(0, 1);
    let cwt = k.const_idx(WT);
    let c1024 = k.const_idx(TILE_ELEMS);
    let zero = k.const_idx(0);

    k.loop_over(cwt, |k, ki| {
        let tbase = k.mad(ki, c1024, zero);
        let tx = k.load_global_tile(x, tbase);
        k.store_circular(cdata, tx, zero);
    });
    k.barrier();
    k.barrier();
    k.loop_over(cwt, |k, ki| {
        let tbase = k.mad(ki, c1024, zero);
        let v = k.load_circular(cdata, zero);
        k.store_global_tile(out, v, tbase);
    });

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let data: Vec<f32> = (0..32 * 128).map(|j| j as f32 * 0.015625).collect();
    let x_t = Tensor::from_vec(data.clone(), [32, 128])?.tilize()?.to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&x_t], vec![[32, 128]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.untilize(32, 128)?.to_vec()?;
    assert_eq!(z.len(), 4096);
    let mut bad = 0;
    for (p, (&v, &e)) in z.iter().zip(data.iter()).enumerate() {
        if (v - e).abs() > 1e-6 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {e}");
            }
            bad += 1;
        }
    }
    println!("pad move bad: {bad} / 4096");
    assert_eq!(bad, 0);

    Ok(())
}

#[test]
fn tenstorrent_row_max_reduce() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let x = k.param(DType::F16);
    let s = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let cin = k.circular_storage(DType::F16, 1);
    let csc = k.circular_storage(DType::F16, 1);
    let cout = k.circular_storage(DType::F16, 1);

    let _g = k.group_range(0, 1);

    // Reader: stream WT input tiles + scaler tiles (LLK-mandated ones).
    k.loop_over(4, |k, ki| {
        let tbase = k.mad(ki, 1024, 0);
        let tx = k.load_global_tile(x, tbase);
        k.store_circular(cin, tx, 0);
        let ts = k.load_global_tile(s, 0);
        k.store_circular(csc, ts, 0);
    });
    k.barrier();

    // Compute: fold the running row-max into the Register acc over WT
    // tiles, then pack it once. The acc lives in DST the whole time
    // (acquire seeds it, the pack drains it); the load/store pair is
    // pure SSA threading, no traffic.
    let acc = k.storage(DType::F16, MemScope::Register, 1024);
    k.loop_over(4, |k, _ki| {
        let va = k.load_circular(cin, 0);
        let vs = k.load_circular(csc, 0);
        let av = k.load_register_tile(acc, 0);
        let f = k.reduce_tile(va, vs, av, BOp::Max, TileReduceKind::Col);
        k.store_register_tile(acc, f, 0);
    });
    let f = k.load_register_tile(acc, 0);
    k.store_circular(cout, f, 0);
    k.barrier();

    // Writer: drain the single output tile.
    k.loop_over(1, |k, _ki| {
        let v = k.load_circular(cout, 0);
        k.store_global_tile(out, v, 0);
    });

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    // val(r, c) = r*0.5 + c*2^-7: col max at r=31 of WT tile 3
    // (c = 96+j): 31*0.5 + (96+j)*2^-7 = 16.25 + j*2^-7.
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
    let out_bufs = compiled.forward(&[&x_t, &s_t], vec![[32, 32]])?;

    let host = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?;
    println!("{host}");
    let z: Vec<f32> = host.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for c in 0..32 {
        let expected = 16.25 + c as f32 * 0.0078125;
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

/// Single 32x32 transpose: one WT tile through `transpose_wh`.
#[test]
fn tenstorrent_transpose_tile() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let x = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let cin = k.circular_storage(DType::F16, 1);
    let cout = k.circular_storage(DType::F16, 1);

    let _g = k.group_range(0, 1);

    let tx = k.load_global_tile(x, 0);
    k.store_circular(cin, tx, 0);
    k.barrier();
    let va = k.load_circular(cin, 0);
    let t = k.transpose_tile(va);
    k.store_circular(cout, t, 0);
    k.barrier();
    let v = k.load_circular(cout, 0);
    k.store_global_tile(out, v, 0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    // val(r, c) = r*32 + c: exact in F16, transpose swaps to c*32 + r.
    let data: Vec<f32> = (0..32 * 32).map(|j| j as f32).collect();
    let x_t = Tensor::from_vec(data, [32, 32])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&x_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for r in 0..32 {
        for c in 0..32 {
            let expected = (c * 32 + r) as f32;
            if (z[r * 32 + c] - expected).abs() >= 3e-2 {
                if bad < 10 {
                    println!("z[{r}][{c}] = {}, expected {expected}", z[r * 32 + c]);
                }
                bad += 1;
            }
        }
    }
    println!("transpose bad: {bad} / 1024");
    assert_eq!(bad, 0);

    Ok(())
}

/// Official `eltwise_sfpu` shape: compute runs per-tile
/// wait/acquire/copy/exp/commit/wait/reserve/pack/pop/release/push,
/// with startup + copy/op inits hoisted ahead of the loop.
#[test]
fn tenstorrent_eltwise_exp_sfpu() -> Result<(), ZyxError> {
    const WT: i64 = 4;

    let mut k = Kernel::new(Dev::TT(0));
    let x = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let cin = k.circular_storage(DType::F16, 1);
    let cout = k.circular_storage(DType::F16, 1);

    let _g = k.group_range(0, 1);

    k.loop_over(WT, |k, ki| {
        let tbase = k.mad(ki, 1024, 0);
        let tx = k.load_global_tile(x, tbase);
        k.store_circular(cin, tx, 0);
    });
    k.barrier();
    k.loop_over(WT, |k, _ki| {
        let va = k.load_circular(cin, 0);
        let ve = k.exp(va);
        k.store_circular(cout, ve, 0);
    });
    k.barrier();
    k.loop_over(WT, |k, ki| {
        let tbase = k.mad(ki, 1024, 0);
        let v = k.load_circular(cout, 0);
        k.store_global_tile(out, v, tbase);
    });

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    // Exp inputs kept in [0, 2): F16-exact steps, no overflow.
    let data: Vec<f32> = (0..32 * 128).map(|j| (j % 32) as f32 * 0.0625).collect();
    let x_t = Tensor::from_vec(data.clone(), [32, 128])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&x_t], vec![[32, 128]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 128)?.to_vec()?;
    assert_eq!(z.len(), 4096);
    let mut bad = 0;
    for (p, (&v, &e)) in z.iter().zip(data.iter()).enumerate() {
        let expected = e.exp();
        if (v - expected).abs() >= 3e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {expected}");
            }
            bad += 1;
        }
    }
    println!("exp bad: {bad} / 4096");
    assert_eq!(bad, 0);

    Ok(())
}

/// Single-kind binary add: proves `add_binary_tile` alone before
/// debugging the mixed exp→add cone.
#[test]
fn tenstorrent_eltwise_add() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::F16);
    let b = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let ca = k.circular_storage(DType::F16, 1);
    let cb = k.circular_storage(DType::F16, 1);
    let cout = k.circular_storage(DType::F16, 1);

    let _g = k.group_range(0, 1);

    let ta = k.load_global_tile(a, 0);
    k.store_circular(ca, ta, 0);
    let tb = k.load_global_tile(b, 0);
    k.store_circular(cb, tb, 0);
    k.barrier();
    let va = k.load_circular(ca, 0);
    let vb = k.load_circular(cb, 0);
    let s = k.add(va, vb);
    k.store_circular(cout, s, 0);
    k.barrier();
    let v = k.load_circular(cout, 0);
    k.store_global_tile(out, v, 0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let data_a: Vec<f32> = (0..32 * 32).map(|j| (j % 32) as f32 * 0.0625).collect();
    let data_b: Vec<f32> = (0..32 * 32).map(|j| ((j + 7) % 32) as f32 * 0.0625).collect();
    let to_tt = |v: Vec<f32>| -> Result<Tensor, ZyxError> {
        Tensor::from_vec(v, [32, 32])?.tilize()?.cast(DType::F16).to(Dev::TT(0))
    };
    let a_t = to_tt(data_a.clone())?;
    let b_t = to_tt(data_b.clone())?;
    let out_bufs = compiled.forward(&[&a_t, &b_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for (p, ((&x, &y), &v)) in data_a.iter().zip(data_b.iter()).zip(z.iter()).enumerate() {
        let expected = x + y;
        if (v - expected).abs() >= 3e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {expected}");
            }
            bad += 1;
        }
    }
    println!("add bad: {bad} / 1024");
    assert_eq!(bad, 0);

    Ok(())
}

/// Probe: back-to-back in-place exps on the SAME slot, no add. Bisects
/// the mixed exp→add zeros: green means chained exps are fine and the
/// break is add-after-exp (or the second slot); red means chained SFPU
/// on one slot is the break.
#[test]
fn tenstorrent_probe_dual_exp() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let ca = k.circular_storage(DType::F16, 1);
    let cout = k.circular_storage(DType::F16, 1);

    let _g = k.group_range(0, 1);

    let ta = k.load_global_tile(a, 0);
    k.store_circular(ca, ta, 0);
    k.barrier();
    // Probe-only: does the official `init_sfpu` setup fix multi-op episodes?
    let _sfpu = k.asm("init_sfpu({0}, {1})", &[ca, cout]);
    let va = k.load_circular(ca, 0);
    let ea = k.exp(va);
    let eb = k.exp(ea);
    k.store_circular(cout, eb, 0);
    k.barrier();
    let v = k.load_circular(cout, 0);
    k.store_global_tile(out, v, 0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let data: Vec<f32> = (0..32 * 32).map(|j| (j % 32) as f32 * 0.0625).collect();
    let a_t = Tensor::from_vec(data.clone(), [32, 32])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for (p, (&x, &v)) in data.iter().zip(z.iter()).enumerate() {
        let expected = x.exp().exp();
        if (v - expected).abs() >= 5e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {expected}");
            }
            bad += 1;
        }
    }
    println!("dual-exp bad: {bad} / 1024");
    assert_eq!(bad, 0);

    Ok(())
}

/// Mixed-kind cone (exp then add): proves the `Programmed` phase
/// cursor — `exp` hoists as the anchor, `add_binary_tile_init` goes
/// inline once at the unary→binary switch.
#[test]
fn tenstorrent_mixed_exp_add() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::F16);
    let b = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let ca = k.circular_storage(DType::F16, 1);
    let cb = k.circular_storage(DType::F16, 1);
    let cout = k.circular_storage(DType::F16, 1);

    let _g = k.group_range(0, 1);

    let ta = k.load_global_tile(a, 0);
    k.store_circular(ca, ta, 0);
    let tb = k.load_global_tile(b, 0);
    k.store_circular(cb, tb, 0);
    k.barrier();
    let va = k.load_circular(ca, 0);
    let ea = k.exp(va);
    let vb = k.load_circular(cb, 0);
    let eb = k.exp(vb);
    let s = k.add(ea, eb);
    k.store_circular(cout, s, 0);
    k.barrier();
    let v = k.load_circular(cout, 0);
    k.store_global_tile(out, v, 0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    // Inputs in [0, 2): F16-exact steps, no overflow.
    let data_a: Vec<f32> = (0..32 * 32).map(|j| (j % 32) as f32 * 0.0625).collect();
    let data_b: Vec<f32> = (0..32 * 32).map(|j| ((j + 7) % 32) as f32 * 0.0625).collect();
    let to_tt = |v: Vec<f32>| -> Result<Tensor, ZyxError> {
        Tensor::from_vec(v, [32, 32])?.tilize()?.cast(DType::F16).to(Dev::TT(0))
    };
    let a_t = to_tt(data_a.clone())?;
    let b_t = to_tt(data_b.clone())?;
    let out_bufs = compiled.forward(&[&a_t, &b_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for (p, ((&x, &y), &v)) in data_a.iter().zip(data_b.iter()).zip(z.iter()).enumerate() {
        let expected = x.exp() + y.exp();
        if (v - expected).abs() >= 3e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {expected}");
            }
            bad += 1;
        }
    }
    println!("mixed bad: {bad} / 1024");
    assert_eq!(bad, 0);

    Ok(())
}

/// Mixed-kind cone (transpose then exp): Transpose→Unary switch.
#[test]
fn tenstorrent_mixed_transpose_exp() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let x = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let cin = k.circular_storage(DType::F16, 1);
    let cout = k.circular_storage(DType::F16, 1);

    let _g = k.group_range(0, 1);

    let tx = k.load_global_tile(x, 0);
    k.store_circular(cin, tx, 0);
    k.barrier();
    let va = k.load_circular(cin, 0);
    let t = k.transpose_tile(va);
    let e = k.exp(t);
    k.store_circular(cout, e, 0);
    k.barrier();
    let v = k.load_circular(cout, 0);
    k.store_global_tile(out, v, 0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    // Inputs in [0, 2): F16-exact steps, no overflow.
    let data: Vec<f32> = (0..32 * 32).map(|j| (j % 32) as f32 * 0.0625).collect();
    let x_t = Tensor::from_vec(data.clone(), [32, 32])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&x_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for r in 0..32 {
        for c in 0..32 {
            let expected = data[c * 32 + r].exp();
            if (z[r * 32 + c] - expected).abs() >= 3e-2 {
                if bad < 10 {
                    println!("z[{r}][{c}] = {}, expected {expected}", z[r * 32 + c]);
                }
                bad += 1;
            }
        }
    }
    println!("mixed transpose-exp bad: {bad} / 1024");
    assert_eq!(bad, 0);

    Ok(())
}

/// Mixed-kind cone (exp then column-sum): Unary→Reduce switch,
/// the softmax-sum pattern. The exp drains through a CB: reduce
/// takes CB tile loads only.
#[test]
fn tenstorrent_mixed_exp_reduce() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let x = k.param(DType::F16);
    let s = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let cin = k.circular_storage(DType::F16, 1);
    let csc = k.circular_storage(DType::F16, 1);
    let cmid = k.circular_storage(DType::F16, 1);
    let cout = k.circular_storage(DType::F16, 1);

    let _g = k.group_range(0, 1);

    let tx = k.load_global_tile(x, 0);
    k.store_circular(cin, tx, 0);
    let ts = k.load_global_tile(s, 0);
    k.store_circular(csc, ts, 0);
    k.barrier();
    let acc = k.storage(DType::F16, MemScope::Register, 1024);
    let va = k.load_circular(cin, 0);
    let e = k.exp(va);
    k.store_circular(cmid, e, 0);
    let ve = k.load_circular(cmid, 0);
    let vs = k.load_circular(csc, 0);
    let av = k.load_register_tile(acc, 0);
    let f = k.reduce_tile(ve, vs, av, BOp::Add, TileReduceKind::Col);
    k.store_register_tile(acc, f, 0);
    let g = k.load_register_tile(acc, 0);
    k.store_circular(cout, g, 0);
    k.barrier();
    let v = k.load_circular(cout, 0);
    k.store_global_tile(out, v, 0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    // Inputs in [0, 1): column sums stay small in F16.
    let data: Vec<f32> = (0..32 * 32).map(|j| (j % 16) as f32 * 0.0625).collect();
    let to_tt = |v: Vec<f32>, rows: i64, cols: i64| -> Result<Tensor, ZyxError> {
        Tensor::from_vec(v, [rows, cols])?.tilize()?.cast(DType::F16).to(Dev::TT(0))
    };
    let x_t = to_tt(data.clone(), 32, 32)?;
    let s_t = to_tt(vec![1.0f32; 1024], 32, 32)?;
    let out_bufs = compiled.forward(&[&x_t, &s_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for c in 0..32 {
        let expected: f32 = (0..32).map(|r| data[r * 32 + c].exp()).sum();
        if (z[c] - expected).abs() >= 5e-1 {
            if bad < 10 {
                println!("z[{c}] = {}, expected {expected}", z[c]);
            }
            bad += 1;
        }
    }
    println!("mixed exp-reduce bad: {bad} / 32");
    assert_eq!(bad, 0);

    Ok(())
}

/// Official `matmul_single_core` shape: reader streams A(mt,kt) + B(kt,nt)
/// tiles in mt/nt/kt order, compute accumulates Kt tiles per output with
/// the acquire hoisted above the Kt loop, writer drains row-major.
#[test]
fn tenstorrent_matmul_single_core() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::F16);
    let b = k.param(DType::F16);
    let out = k.param_mut(DType::F32);

    let ca = k.circular_storage(DType::F16, 1);
    let cb = k.circular_storage(DType::F16, 1);
    let cout = k.circular_storage(DType::F32, 1);

    let _g = k.group_range(0, 1);

    // Reader: A tile (mt,kt) at mt*Kt+kt, B tile (kt,nt) at kt*Nt+nt.
    k.loop_over(1, |k, mti| {
        k.loop_over(2, |k, nti| {
            k.loop_over(2, |k, kti| {
                let at = k.mad(mti, 2, kti);
                let abase = k.mad(at, 1024, 0);
                let ta = k.load_global_tile(a, abase);
                k.store_circular(ca, ta, 0);
                let bt = k.mad(kti, 2, nti);
                let bbase = k.mad(bt, 1024, 0);
                let tb = k.load_global_tile(b, bbase);
                k.store_circular(cb, tb, 0);
            });
        });
    });
    k.barrier();
    // Compute: one acc cone per output tile, Kt accumulation steps.
    k.loop_over(1, |k, _mti| {
        k.loop_over(2, |k, _nti| {
            let acc = k.storage(DType::F32, MemScope::Register, 1024);
            k.loop_over(2, |k, _kti| {
                let va = k.load_circular(ca, 0);
                let vb = k.load_circular(cb, 0);
                let av = k.load_register_tile(acc, 0);
                let f = k.matmul_tile(va, vb, av);
                k.store_register_tile(acc, f, 0);
            });
            let f = k.load_register_tile(acc, 0);
            k.store_circular(cout, f, 0);
        });
    });
    k.barrier();
    // Writer: output tile (mt,nt) at mt*Nt+nt, row-major.
    k.loop_over(1, |k, mti| {
        k.loop_over(2, |k, nti| {
            let ot = k.mad(mti, 2, nti);
            let obase = k.mad(ot, 1024, 0);
            let v = k.load_circular(cout, 0);
            k.store_global_tile(out, v, obase);
        });
    });

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    // A[32,64] @ B[64,64], values kept small for F16 exactness.
    let a_data: Vec<f32> = (0..32 * 64).map(|j| (j % 4) as f32 * 0.0625).collect();
    let b_data: Vec<f32> = (0..64 * 64).map(|j| ((j / 4) % 4) as f32 * 0.0625).collect();
    let mut expected = vec![0.0f32; 32 * 64];
    for r in 0..32 {
        for c in 0..64 {
            let mut s = 0.0f32;
            for t in 0..64 {
                s += a_data[r * 64 + t] * b_data[t * 64 + c];
            }
            expected[r * 64 + c] = s;
        }
    }
    let a_t = Tensor::from_vec(a_data, [32, 64])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let b_t = Tensor::from_vec(b_data, [64, 64])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a_t, &b_t], vec![[32, 64]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.untilize(32, 64)?.to_vec()?;
    assert_eq!(z.len(), 2048);
    let mut bad = 0;
    for (p, (&v, &e)) in z.iter().zip(expected.iter()).enumerate() {
        if (v - e).abs() >= 5e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {e}");
            }
            bad += 1;
        }
    }
    println!("mm bad: {bad} / 2048");
    assert_eq!(bad, 0);

    Ok(())
}

/// Mixed-kind cone (matmul then bias-add): Matmul→Binary switch.
/// Same geometry as `tenstorrent_matmul_single_core` plus a per-nt
/// bias tile added to each acc cone.
#[test]
fn tenstorrent_mixed_matmul_bias() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::F16);
    let b = k.param(DType::F16);
    let c = k.param(DType::F32);
    let out = k.param_mut(DType::F32);

    let ca = k.circular_storage(DType::F16, 1);
    let cb = k.circular_storage(DType::F16, 1);
    let cc = k.circular_storage(DType::F32, 1);
    let cout = k.circular_storage(DType::F32, 1);

    let _g = k.group_range(0, 1);

    // Reader: A tile (mt,kt) at mt*Kt+kt, B tile (kt,nt) at kt*Nt+nt,
    // C bias tile (mt,nt) at mt*Nt+nt.
    k.loop_over(1, |k, mti| {
        k.loop_over(2, |k, nti| {
            k.loop_over(2, |k, kti| {
                let at = k.mad(mti, 2, kti);
                let abase = k.mad(at, 1024, 0);
                let ta = k.load_global_tile(a, abase);
                k.store_circular(ca, ta, 0);
                let bt = k.mad(kti, 2, nti);
                let bbase = k.mad(bt, 1024, 0);
                let tb = k.load_global_tile(b, bbase);
                k.store_circular(cb, tb, 0);
            });
            let ct = k.mad(mti, 2, nti);
            let cbase = k.mad(ct, 1024, 0);
            let tc = k.load_global_tile(c, cbase);
            k.store_circular(cc, tc, 0);
        });
    });
    k.barrier();
    // Compute: one acc cone per output tile, bias added after the Kt
    // accumulation steps.
    k.loop_over(1, |k, _mti| {
        k.loop_over(2, |k, _nti| {
            let acc = k.storage(DType::F32, MemScope::Register, 1024);
            k.loop_over(2, |k, _kti| {
                let va = k.load_circular(ca, 0);
                let vb = k.load_circular(cb, 0);
                let av = k.load_register_tile(acc, 0);
                let f = k.matmul_tile(va, vb, av);
                k.store_register_tile(acc, f, 0);
            });
            let f = k.load_register_tile(acc, 0);
            let vc = k.load_circular(cc, 0);
            let s = k.add(f, vc);
            k.store_circular(cout, s, 0);
        });
    });
    k.barrier();
    // Writer: output tile (mt,nt) at mt*Nt+nt, row-major.
    k.loop_over(1, |k, mti| {
        k.loop_over(2, |k, nti| {
            let ot = k.mad(mti, 2, nti);
            let obase = k.mad(ot, 1024, 0);
            let v = k.load_circular(cout, 0);
            k.store_global_tile(out, v, obase);
        });
    });

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    // A[32,64] @ B[64,64] + C[32,64], values kept small.
    let a_data: Vec<f32> = (0..32 * 64).map(|j| (j % 4) as f32 * 0.0625).collect();
    let b_data: Vec<f32> = (0..64 * 64).map(|j| ((j / 4) % 4) as f32 * 0.0625).collect();
    let c_data: Vec<f32> = (0..32 * 64).map(|j| (j % 8) as f32 * 0.0625).collect();
    let mut expected = vec![0.0f32; 32 * 64];
    for r in 0..32 {
        for c in 0..64 {
            let mut s = 0.0f32;
            for t in 0..64 {
                s += a_data[r * 64 + t] * b_data[t * 64 + c];
            }
            expected[r * 64 + c] = s + c_data[r * 64 + c];
        }
    }
    let a_t = Tensor::from_vec(a_data, [32, 64])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let b_t = Tensor::from_vec(b_data, [64, 64])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let c_t = Tensor::from_vec(c_data, [32, 64])?.tilize()?.cast(DType::F32).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a_t, &b_t, &c_t], vec![[32, 64]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.untilize(32, 64)?.to_vec()?;
    assert_eq!(z.len(), 2048);
    let mut bad = 0;
    for (p, (&v, &e)) in z.iter().zip(expected.iter()).enumerate() {
        if (v - e).abs() >= 5e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {e}");
            }
            bad += 1;
        }
    }
    println!("mixed mm-bias bad: {bad} / 2048");
    assert_eq!(bad, 0);

    Ok(())
}
