// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! TEMP DIAG size probe: 14_tenstorrent passes with 2-page buffers; lm_head
//! fails reading a 9-page buffer. Same kernel shape, 9 tiles per buffer.

#![cfg(feature = "tenstorrent")]

use zyx::kernel::{BOp, Dev, Kernel, MemScope, TileReduceKind};
use zyx::{DType, Tensor, ZyxError};

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
    // Two pages: the running partial stays valid while the next one reserves.
    let cacc = k.storage(DType::F16, MemScope::Circular, 2 * TILE_ELEMS);

    let _g = k.group_range(0, 1);
    let cwt = k.const_idx(WT);
    let c1024 = k.const_idx(TILE_ELEMS);
    let zero = k.const_idx(0);

    // Reader: min seed to acc, then stream WT input tiles + scaler tiles.
    let tm = k.load_tile(m, zero, TDIM, TDIM, TDIM as u32);
    k.store_tile(cacc, tm, zero, TDIM, TDIM, TDIM as u32);
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
    k.loop_over(cwt, |k, _ki| {
        let va = k.load_tile(cin, zero, TDIM, TDIM, TDIM as u32);
        let vs = k.load_tile(csc, zero, TDIM, TDIM, TDIM as u32);
        let a = k.load_tile(cacc, zero, TDIM, TDIM, TDIM as u32);
        let f = k.reduce_tile(va, vs, a, BOp::Max, TileReduceKind::Row);
        k.store_tile(cacc, f, zero, TDIM, TDIM, TDIM as u32);
    });
    k.barrier();

    // Writer: stream acc tile to DRAM.
    let v = k.load_tile(cacc, zero, TDIM, TDIM, TDIM as u32);
    k.store_tile(out, v, zero, TDIM, TDIM, TDIM as u32);

    k.verify();
    let compiled = k.compile()?;

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
    let out_bufs = compiled.forward(&[&x_t, &s_t, &m_t], vec![[TILE_ELEMS]])?;

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
