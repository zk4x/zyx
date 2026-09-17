// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Probe the repack_q4k scale/min op chain on the full tensor: print
//! d, dmin, sc, m for super-block 0 and compare with python truth
//! (d = 9.12e-5, dmin = 0.000651, sc0 = 38, m0 = 33).

use zyx::{DType, Tensor, ZyxError};

#[test]
fn probe_scale_chain() -> Result<(), ZyxError> {
    let (_meta, tensors) = Tensor::load_gguf("/home/x/Dev/rust/zyx/examples/models/Qwen3.8-27B-UD-Q4_K_XL.gguf")?;
    let raw = &tensors["blk.1.attn_qkv.weight"];
    println!("raw shape: {:?}", raw.shape());

    let raw3 = raw.split([4i64, 12i64, 128i64], 1)?;
    let (dd, s, _qs) = (&raw3[0], &raw3[1], &raw3[2]);
    let lanes = dd.split([1i64, 1i64, 1i64, 1i64], 1)?;
    let d_bits = (lanes[0].cast(DType::U16) + lanes[1].cast(DType::U16) * 256u16).reshape([204800])?;
    let dmin_bits = (lanes[2].cast(DType::U16) + lanes[3].cast(DType::U16) * 256u16).reshape([204800])?;
    println!("d_bits[0..4] = {:?}", d_bits.narrow(0, 0i64, 4i64)?.to_vec::<u16>()?);
    println!("dmin_bits[0..4] = {:?}", dmin_bits.narrow(0, 0i64, 4i64)?.to_vec::<u16>()?);
    let d = d_bits.bitcast(DType::F16)?.cast(DType::BF16);
    let dmin = dmin_bits.bitcast(DType::F16)?.cast(DType::BF16);
    println!("d bf16 bits[0..4] = {:?}", d.narrow(0, 0i64, 4i64)?.to_le_bytes()?.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect::<Vec<_>>());
    println!("dmin bf16 bits[0..4] = {:?}", dmin.narrow(0, 0i64, 4i64)?.to_le_bytes()?.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect::<Vec<_>>());

    let slanes = s.split([1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64], 1)?;
    let mut sc_parts = Vec::with_capacity(8);
    let mut m_parts = Vec::with_capacity(8);
    for j in 0..8usize {
        if j < 4 {
            sc_parts.push((&slanes[j] & 63u8).reshape([204800, 1])?);
            m_parts.push((&slanes[j + 4] & 63u8).reshape([204800, 1])?);
        } else {
            let sc = (&slanes[j + 4] & 15u8).reshape([204800, 1])?
                + (&slanes[j - 4] >> 6u8).reshape([204800, 1])? * 16u8;
            let m = (&slanes[j + 4] >> 4u8).reshape([204800, 1])?
                + (&slanes[j] >> 6u8).reshape([204800, 1])? * 16u8;
            sc_parts.push(sc);
            m_parts.push(m);
        }
    }
    let sc = Tensor::cat(&sc_parts, 1)?.cast(DType::BF16);
    let mm = Tensor::cat(&m_parts, 1)?.cast(DType::BF16);
    println!("sc[0] = {:?}", sc.narrow(0, 0i64, 1i64)?.to_vec::<zyx::bf16>()?);
    println!("mm[0] = {:?}", mm.narrow(0, 0i64, 1i64)?.to_vec::<zyx::bf16>()?);
    let prod = (d.reshape([204800, 1])? * sc).narrow(0, 0i64, 1i64)?;
    println!("d*sc[0] = {:?}", prod.to_vec::<zyx::bf16>()?);
    println!("truth: sc0=38 m0=33, d*sc0=0.0034654, dmin*m0=0.0214791");
    Ok(())
}
