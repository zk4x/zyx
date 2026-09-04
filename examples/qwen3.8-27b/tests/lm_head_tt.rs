// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Model-shaped Tenstorrent matmul test (runs on TT).
//!
//! Golden: `examples/data/qwen3_8b_lm_head.safetensors` (same file as the
//! CUDA side: x [2,4,64], w [256,64], y [2,4,256]). The kernel could not
//! care less about tiles: host pads X rows 8->32 and face-encodes tiles,
//! then launches one output tile per invocation (M=1, N=1, K=2, host loops
//! over the 8 N tiles). B tiles are plain (k,n) face tiles, no transpose:
//! B[n,kt][r,c] = W[n*32+c, kt*32+r].
//!
//! 3 CBs (ca deep 2, cb deep 2, cc single), Kt=2 unrolled in IR
//! construction, accumulation via add. Mirrors the official TT
//! matmul_single_core structure (hw_startup + mm_init once, acquire zeroes
//! DST, matmul_tiles accumulates per slot).

use zyx::kernel::{Dev, Kernel, MemScope};
use zyx::{DType, Tensor, ZyxError, bf16};

const TDIM: u16 = 32;
const N_TILES: usize = 8;
const K_TILES: usize = 2;

// Face slot -> linear index within a tile.
fn lin(s: usize) -> usize {
    let (face, local) = (s / 256, s % 256);
    let (fr0, fc0) = (face / 2, face % 2);
    fr0 * 16 * 32 + fc0 * 16 + (local / 16) * 32 + local % 16
}

// Row-major tile -> face order for DRAM.
fn tile_encode(tile: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; 1024];
    for p in 0..1024 {
        out[p] = tile[lin(p)];
    }
    out
}

// Face order -> row-major tile.
fn tile_decode(face: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; 1024];
    for p in 0..1024 {
        out[lin(p)] = face[p];
    }
    out
}

#[test]
fn lm_head_tt() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::BF16);
    let b = k.param(DType::BF16);
    let nvar = k.variable(DType::I64);
    let z = k.param_mut(DType::BF16);

    let ca = k.storage(DType::BF16, MemScope::Circular, 2048i64.into());
    let cb = k.storage(DType::BF16, MemScope::Circular, 2048i64.into());
    let cc = k.storage(DType::BF16, MemScope::Circular, 1024i64.into());

    // Single core.
    let one = k.const_idx(1i64);
    let _g = k.group_range(0, one);
    let zero = k.const_idx(0);
    let two = k.const_idx(2i64);
    let te = k.const_idx(1024i64);

    // ---- Reader: A tiles (const offsets), B tiles for this n ----
    for kt in 0..K_TILES {
        let off_a = k.const_idx((kt * 1024) as i64);
        let t = k.load_tile(a, off_a, TDIM, TDIM, TDIM as u32);
        k.store_tile(ca, t, zero, TDIM, TDIM, TDIM as u32);
    }
    for kt in 0..K_TILES {
        // DRAM tile = n*2+kt, element offset = (n*2+kt)*1024.
        let c_kt = k.const_idx(kt as i64);
        let tile = k.mad(nvar, two, c_kt);
        let off = k.mad(tile, te, zero);
        let t = k.load_tile(b, off, TDIM, TDIM, TDIM as u32);
        k.store_tile(cb, t, zero, TDIM, TDIM, TDIM as u32);
    }
    k.barrier();

    // ---- Compute: C = A[0]@B[0] + A[1]@B[1] ----
    let mut acc = k.const_val(bf16::from_f32(0.0));
    for _ in 0..K_TILES {
        // NOTE: both iterations load tile 0: multi-tile CB indexing
        // (copy_tile with tile index) is not implemented yet. Kt=2
        // accumulation across distinct tiles needs it.
        let la = k.load_tile(ca, zero, TDIM, TDIM, TDIM as u32);
        let lb = k.load_tile(cb, zero, TDIM, TDIM, TDIM as u32);
        let t = k.matmul_tile(la, lb);
        acc = k.add(acc, t);
    }
    k.store_tile(cc, acc, zero, TDIM, TDIM, TDIM as u32);

    // ---- Writer: single C tile at DRAM tile n ----
    k.barrier();
    let v = k.load_tile(cc, zero, TDIM, TDIM, TDIM as u32);
    let zoff = k.mad(nvar, te, zero);
    k.store_tile(z, v, zoff, TDIM, TDIM, TDIM as u32);

    k.verify();
    let compiled = k.compile()?;

    // Host: pad X rows 8->32, face-encode A/B tiles, compare decoded C
    // tile rows 0..8 against the model-shaped golden.
    let goldens = Tensor::load("../data/qwen3_8b_lm_head.safetensors")?;
    let x: Vec<f32> = goldens["input"].to_vec()?;
    let w: Vec<f32> = goldens["weight"].to_vec()?;
    let y: Vec<f32> = goldens["output"].to_vec()?;

    // A[kt] = Xp[0:32, kt*32:(kt+1)*32], Xp = x rows + 24 zero rows.
    let mut ap = Vec::with_capacity(2048);
    for kt in 0..K_TILES {
        let mut tile = vec![0.0f32; 1024];
        for r in 0..8 {
            for c in 0..32 {
                tile[r * 32 + c] = x[r * 64 + kt * 32 + c];
            }
        }
        ap.extend(tile_encode(&tile));
    }
    // B[n,kt] = W[n*32:(n+1)*32, kt*32:(kt+1)*32] as (k,n) face tile:
    // tile[r,c] = W[n*32+c, kt*32+r].
    let mut bp = Vec::with_capacity(16384);
    for n in 0..N_TILES {
        for kt in 0..K_TILES {
            let mut tile = vec![0.0f32; 1024];
            for r in 0..32 {
                for c in 0..32 {
                    tile[r * 32 + c] = w[(n * 32 + c) * 64 + kt * 32 + r];
                }
            }
            bp.extend(tile_encode(&tile));
        }
    }

    let a_t = Tensor::from(ap).to(Dev::C)?.cast(DType::BF16).to(Dev::TT(0))?;
    let b_t = Tensor::from(bp).to(Dev::C)?.cast(DType::BF16).to(Dev::TT(0))?;

    // Each launch returns a fresh z (8192 elems); only tile n is written.
    let mut tiles = Vec::with_capacity(N_TILES);
    for n in 0..N_TILES {
        let n_t = Tensor::variable(n as i64);
        let out = compiled.forward(&[&a_t, &b_t, &n_t], vec![[8192i64]])?;
        let z_face: Vec<f32> = out[0].to(Dev::C)?.cast(DType::F32).to_vec()?;
        tiles.push(tile_decode(&z_face[n * 1024..(n + 1) * 1024]));
    }
    // C[n] tile rows 0..8 hold y[b, s, n*32:(n+1)*32].
    let mut bad = 0;
    for m in 0..N_TILES {
        let got = &tiles[m];
        for b in 0..2 {
            for s in 0..4 {
                for c in 0..32 {
                    let v = got[(b * 4 + s) * 32 + c];
                    let e = y[(b * 4 + s) * 256 + m * 32 + c];
                    if (v - e).abs() >= 0.5 {
                        if bad < 10 {
                            println!("m={m} b={b} s={s} c={c}: {v} vs {e}");
                        }
                        bad += 1;
                    }
                }
            }
        }
    }
    println!("bad: {bad} / 2048");
    assert_eq!(bad, 0);
    Ok(())
}
