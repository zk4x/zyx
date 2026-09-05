// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Golden-shape kernel for the Tenstorrent elementwise lowering.
//!
//! Hand-writes the target IR with the kernel API: a fully tiled elementwise
//! kernel `z = x + sin(y)`. DRAM holds tiles in 16x16-face order (TT's own
//! layout model), so reader/writer move whole tiles with single sequential
//! NOC transfers and no swizzle anywhere; compute works on faces natively.

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
    let cx = k.storage(DType::BF16, MemScope::Circular, (TILE_ELEMS as i64).into());
    let cy = k.storage(DType::BF16, MemScope::Circular, (TILE_ELEMS as i64).into());
    let cz = k.storage(DType::BF16, MemScope::Circular, (TILE_ELEMS as i64).into());

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
