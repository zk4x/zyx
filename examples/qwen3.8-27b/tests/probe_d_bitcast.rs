// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Probe: d/dmin F16-bitcast extraction from Q4_K super-block 0 of
//! blk.1.attn_qkv.weight. Truth: d = 9.12e-5, dmin = 0.000651.

use zyx::{DType, Tensor, ZyxError};

#[test]
fn probe_d_bitcast() -> Result<(), ZyxError> {
    let (_meta, tensors) = Tensor::load_gguf("/home/x/Dev/rust/zyx/examples/models/Qwen3.8-27B-UD-Q4_K_XL.gguf")?;
    let raw = &tensors["blk.1.attn_qkv.weight"];
    let first = raw.narrow(0, 0i64, 1i64)?; // [1, 144]
    println!("raw[0,0..8] u8: {:?}", first.narrow(1, 0i64, 8i64)?.to_vec::<u8>()?);

    let raw3 = first.split([4i64, 12i64, 128i64], 1)?;
    let dd = &raw3[0];
    let lanes = dd.split([1i64, 1i64, 1i64, 1i64], 1)?;
    let d_bits = (lanes[0].cast(DType::U16) + lanes[1].cast(DType::U16) * 256u16).reshape([1])?;
    let dmin_bits = (lanes[2].cast(DType::U16) + lanes[3].cast(DType::U16) * 256u16).reshape([1])?;
    println!("d_bits u16: {:?}", d_bits.to_vec::<u16>()?);
    println!("dmin_bits u16: {:?}", dmin_bits.to_vec::<u16>()?);
    let d_f16 = d_bits.bitcast(DType::F16)?;
    let dmin_f16 = dmin_bits.bitcast(DType::F16)?;
    println!("d as F16: {:?}", d_f16.to_vec::<zyx::f16>()?);
    println!("dmin as F16: {:?}", dmin_f16.to_vec::<zyx::f16>()?);
    let d = d_f16.cast(DType::BF16);
    let dmin = dmin_f16.cast(DType::BF16);
    println!("d as BF16: {:?}", d.to_vec::<zyx::bf16>()?);
    println!("dmin as BF16: {:?}", dmin.to_vec::<zyx::bf16>()?);
    println!("expected: d = 9.12e-05 (F16 0x36fa), dmin = 0.000651 (F16 0x36aa)");
    Ok(())
}
