// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Split repack_q4k's scales/mins pipeline into stages and dump each to
//! /tmp/opengen/stage_*.safetensors so python can localize the corrupting
//! op. Python truth: d, dmin (F16 bitcast), sc/mm (6-bit), prod [n,8],
//! grid [rows, c32], tile-major [nt, 32].

use zyx::{DType, Module, Tensor, ZyxError};
use std::collections::HashMap;

#[test]
fn probe_repack_stages() -> Result<(), ZyxError> {
    let (_meta, tensors) = Tensor::load_gguf("/home/x/Dev/rust/zyx/examples/models/Qwen3.8-27B-UD-Q4_K_XL.gguf")?;
    let raw = &tensors["blk.1.attn_qkv.weight"];
    let (rows, cols) = (10240i64, 5120i64);
    let n: i64 = rows * cols / 256;

    let raw3 = raw.split([4i64, 12i64, 128i64], 1)?;
    let (dd, s) = (&raw3[0], &raw3[1]);
    let lanes = dd.split([1i64, 1i64, 1i64, 1i64], 1)?;
    let d_bits = (lanes[0].cast(DType::U16) + lanes[1].cast(DType::U16) * 256u16).reshape([n])?;
    let dmin_bits = (lanes[2].cast(DType::U16) + lanes[3].cast(DType::U16) * 256u16).reshape([n])?;
    let d = d_bits.bitcast(DType::F16)?.cast(DType::BF16);
    let dmin = dmin_bits.bitcast(DType::F16)?.cast(DType::BF16);

    let slanes = s.split([1i64; 12], 1)?;
    let mut sc_parts = Vec::with_capacity(8);
    let mut m_parts = Vec::with_capacity(8);
    for j in 0..8usize {
        if j < 4 {
            sc_parts.push((&slanes[j] & 63u8).reshape([n, 1])?);
            m_parts.push((&slanes[j + 4] & 63u8).reshape([n, 1])?);
        } else {
            let sc = (&slanes[j + 4] & 15u8).reshape([n, 1])?
                + (&slanes[j - 4] >> 6u8).reshape([n, 1])? * 16u8;
            let m = (&slanes[j + 4] >> 4u8).reshape([n, 1])?
                + (&slanes[j] >> 6u8).reshape([n, 1])? * 16u8;
            sc_parts.push(sc);
            m_parts.push(m);
        }
    }
    let sc = Tensor::cat(&sc_parts, 1)?.cast(DType::BF16);
    let mm = Tensor::cat(&m_parts, 1)?.cast(DType::BF16);

    // Stage A: d/dmin/sc/mm as BF16 bit patterns + values
    let mut stages: HashMap<String, Tensor> = HashMap::new();
    stages.insert("d_bits".into(), d_bits.reshape([n, 1])?);
    stages.insert("dmin_bits".into(), dmin_bits.reshape([n, 1])?);
    stages.insert("d".into(), d.reshape([n, 1])?);
    stages.insert("dmin".into(), dmin.reshape([n, 1])?);
    stages.insert("sc".into(), sc.clone());
    stages.insert("mm".into(), mm.clone());

    // Stage B: prod [n, 8]
    let c32 = cols / 32;
    let dcol = d.reshape([n, 1])?;
    let dmincol = dmin.reshape([n, 1])?;
    let prod = &dcol * &sc;
    let prodm = &dmincol * &mm;
    stages.insert("prod".into(), prod.clone());
    stages.insert("prodm".into(), prodm.clone());

    // Stage C: grid [rows, c32]
    let grid = prod.reshape([rows, c32])?;
    let gridm = prodm.reshape([rows, c32])?;
    stages.insert("grid".into(), grid.clone());
    stages.insert("gridm".into(), gridm.clone());

    // Stage D: tile-major [nt, 32]
    let ntiles = rows / 32 * c32;
    let scales = grid.reshape([rows / 32, 32, c32])?.permute([0, 2, 1])?.reshape([ntiles, 32])?;
    let mins = gridm.reshape([rows / 32, 32, c32])?.permute([0, 2, 1])?.reshape([ntiles, 32])?;
    stages.insert("scales".into(), scales);
    stages.insert("mins".into(), mins);

    stages.save("/tmp/opengen/stages.safetensors")?;
    println!("stages saved");
    Ok(())
}
