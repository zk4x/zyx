// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Diagnostic: does `dot_dtype(F32)` honor F32 accumulation on magnitudes
//! matching llama layer-1 hidden (row maxabs ~631, 8192-term reductions)?
//! `dot` in F16 is expected to overflow; `dot_dtype(F32)` must stay finite.

use zyx::{DType, Tensor, ZyxError};

#[test]
fn accum_dtype_honored() -> Result<(), ZyxError> {
    Tensor::set_implicit_casts(false);
    // Row 0: large alternating values (maxabs ~600, mean abs ~300).
    // Row 1: small values. W: small values (maxabs ~0.05).
    let mut hvec = Vec::with_capacity(2 * 8192);
    for i in 0..8192 {
        let s = if i % 2 == 0 { 1.0f32 } else { -1.0f32 };
        hvec.push(s * (200.0 + (i % 401) as f32));
    }
    for i in 0..8192 {
        hvec.push(((i % 7) as f32 - 3.0) * 0.1);
    }
    let mut wvec = Vec::with_capacity(3072 * 8192);
    for i in 0..3072 * 8192 {
        wvec.push((((i as u64).wrapping_mul(2654435761) >> 16) as u32 % 1000) as f32 / 1000.0 * 0.1 - 0.05);
    }
    let hid = Tensor::from_vec(hvec, [1i64, 2, 8192])?.cast(DType::F16);
    let w = Tensor::from_vec(wvec, [3072i64, 8192])?.cast(DType::F16);
    let d16 = hid.dot(w.t())?;
    let d32 = hid.dot_dtype(w.t(), DType::F32)?.cast(DType::F16);
    let v16: Vec<f32> = d16.cast(DType::F32).try_into().unwrap();
    let v32: Vec<f32> = d32.cast(DType::F32).try_into().unwrap();
    let fin16 = v16.iter().filter(|x| x.is_finite()).count();
    let fin32 = v32.iter().filter(|x| x.is_finite()).count();
    eprintln!("SYNTH dot_f16 finite={fin16}/{} down0={:?}", v16.len(), &v16[..4]);
    eprintln!("SYNTH dot_f32 finite={fin32}/{} down0={:?}", v32.len(), &v32[..4]);
    assert!(fin32 == v32.len(), "dot_dtype(F32) must be finite");
    Ok(())
}
