// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use zyx::{DType, Scalar, Tape, Tensor, ZyxError};

#[test]
fn eager_f32_to_i32() -> Result<(), ZyxError> {
    let data: [f32; 4] = [1.0, -2.0, 3.5, -0.0];
    let x = Tensor::from(data);
    let z = unsafe { x.bitcast(DType::I32) }?;
    let zdata: Vec<i32> = z.try_into()?;
    for (v, b) in data.iter().zip(zdata.iter()) {
        assert_eq!(*v, f32::from_bits(*b as u32));
    }
    assert_eq!(zdata[0], 0x3F800000 as i32);
    Ok(())
}

#[test]
fn eager_roundtrip() -> Result<(), ZyxError> {
    let data: [f32; 6] = [0.25, -13.75, 1024.5, -0.001, 7.0, -8.125];
    let x = Tensor::from(data);
    let y = unsafe { x.bitcast(DType::I32) }?;
    let z = unsafe { y.bitcast(DType::F32) }?;
    let zdata: Vec<f32> = z.try_into()?;
    for (v, r) in data.iter().zip(zdata) {
        assert_eq!(v, &r);
    }
    Ok(())
}

#[test]
fn eager_width_mismatch_errors() -> Result<(), ZyxError> {
    let x = Tensor::from([1.0f32]);
    let err = unsafe { x.bitcast(DType::F16) }.unwrap_err();
    assert!(matches!(err, ZyxError::DTypeError(_)), "expected DTypeError, got {err:?}");
    Ok(())
}

#[test]
fn tape_f32_to_i32() -> Result<(), ZyxError> {
    let data: [f32; 5] = [2.5, -1.5, 0.125, -32.0, 6.75];
    let x = Tensor::from(data);
    let tape = Tape::new([&x])?;
    let z = unsafe { x.bitcast(DType::I32) }?;
    tape.realize([&z])?;
    let zdata: Vec<i32> = z.try_into()?;
    for (v, b) in data.iter().zip(zdata) {
        assert_eq!(*v, f32::from_bits(b as u32));
    }
    Ok(())
}

#[test]
fn tape_roundtrip() -> Result<(), ZyxError> {
    let data: [i32; 4] = [1065353216, -1073741824, 1080033280, -1077936128];
    let x = Tensor::from(data);
    let tape = Tape::new([&x])?;
    let y = unsafe { x.bitcast(DType::F32) }?;
    let z = unsafe { y.bitcast(DType::I32) }?;
    tape.realize([&z])?;
    let zdata: Vec<i32> = z.try_into()?;
    for (v, r) in data.iter().zip(zdata) {
        assert_eq!(v, &r);
    }
    Ok(())
}

#[test]
fn tape_bitcast_of_computed() -> Result<(), ZyxError> {
    let data: [f32; 4] = [1.5, -2.5, 3.25, -4.75];
    let x = Tensor::from(data);
    let tape = Tape::new([&x])?;
    let y = x.sin();
    let z = unsafe { y.bitcast(DType::I32) }?;
    tape.realize([&z])?;
    let zdata: Vec<i32> = z.try_into()?;
    for (v, b) in data.iter().zip(zdata.iter()) {
        // Autotuned fused sin may differ from Rust's sin by 1 ULP, so compare
        // values (not bits) after bit-reinterpreting back to f32.
        assert!(v.sin().is_equal(f32::from_bits(*b as u32)));
    }
    Ok(())
}
