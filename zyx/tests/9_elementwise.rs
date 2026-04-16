// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, Tensor, ZyxError};

#[test]
fn elementwise_sign() -> Result<(), ZyxError> {
    let t = Tensor::from([-2.0f32, 0.0, 3.0]);
    let result = t.sign();
    assert_eq!(result.slice(0..1).unwrap().cast(DType::F32).item::<f32>(), -1.0);
    assert_eq!(result.slice(1..2).unwrap().cast(DType::F32).item::<f32>(), 0.0);
    assert_eq!(result.slice(2..3).unwrap().cast(DType::F32).item::<f32>(), 1.0);
    Ok(())
}

#[test]
fn elementwise_erf() -> Result<(), ZyxError> {
    let t = Tensor::from([0.0f32, 0.5, 1.0]);
    let result = t.erf();
    let v0 = result.slice(0..1).unwrap().cast(DType::F32).item::<f32>();
    let v1 = result.slice(1..2).unwrap().cast(DType::F32).item::<f32>();
    let v2 = result.slice(2..3).unwrap().cast(DType::F32).item::<f32>();
    assert!((v0 - 0.0).abs() < 0.01);
    assert!((v1 - 0.5).abs() < 0.1);
    assert!((v2 - 0.8).abs() < 0.1);
    Ok(())
}

#[test]
fn elementwise_relu6() -> Result<(), ZyxError> {
    let t = Tensor::from([-2.0f32, 0.5, 5.0, 10.0]);
    let result = t.relu6();
    assert_eq!(result.slice(0..1).unwrap().cast(DType::F32).item::<f32>(), 0.0);
    assert_eq!(result.slice(1..2).unwrap().cast(DType::F32).item::<f32>(), 0.5);
    assert_eq!(result.slice(2..3).unwrap().cast(DType::F32).item::<f32>(), 5.0);
    assert_eq!(result.slice(3..4).unwrap().cast(DType::F32).item::<f32>(), 6.0);
    Ok(())
}

#[test]
fn elementwise_softsign() -> Result<(), ZyxError> {
    let t = Tensor::from([2.0f32]);
    let result = t.softsign();
    let val = result.cast(DType::F32).item::<f32>();
    assert!(val > 0.5 && val < 1.0);
    Ok(())
}

#[test]
fn elementwise_hardtanh() -> Result<(), ZyxError> {
    let t = Tensor::from([-5.0f32, 0.0, 5.0]);
    let result = t.hardtanh();
    assert_eq!(result.slice(0..1).unwrap().cast(DType::F32).item::<f32>(), -1.0);
    assert_eq!(result.slice(1..2).unwrap().cast(DType::F32).item::<f32>(), 0.0);
    assert_eq!(result.slice(2..3).unwrap().cast(DType::F32).item::<f32>(), 1.0);
    Ok(())
}

#[test]
fn elementwise_lerp() -> Result<(), ZyxError> {
    let t = Tensor::from([1.0f32]);
    let end = Tensor::from([10.0f32]);
    let result = t.lerp(&end, 0.5);
    let val = result.cast(DType::F32).item::<f32>();
    assert!((val - 5.5).abs() < 0.1);
    Ok(())
}

#[test]
fn elementwise_isfinite() -> Result<(), ZyxError> {
    let t = Tensor::from([1.0f32, 2.0, 3.0]);
    let result = t.isfinite();
    assert_eq!(result.slice(0..1).unwrap().cast(DType::F32).item::<f32>(), 1.0);
    assert_eq!(result.slice(1..2).unwrap().cast(DType::F32).item::<f32>(), 1.0);
    assert_eq!(result.slice(2..3).unwrap().cast(DType::F32).item::<f32>(), 1.0);
    Ok(())
}

#[test]
fn elementwise_prelu() -> Result<(), ZyxError> {
    let t = Tensor::from([-2.0f32, 1.0]);
    let result = t.prelu(0.1);
    let v0 = result.slice(0..1).unwrap().cast(DType::F32).item::<f32>();
    let v1 = result.slice(1..2).unwrap().cast(DType::F32).item::<f32>();
    assert!((v0 - (-0.2)).abs() < 0.01);
    assert_eq!(v1, 1.0);
    Ok(())
}

#[test]
fn elementwise_expm1() -> Result<(), ZyxError> {
    let t = Tensor::from([0.0f32]);
    let result = t.expm1();
    let val = result.cast(DType::F32).item::<f32>();
    assert!((val - 0.0).abs() < 0.01);
    Ok(())
}

#[test]
fn elementwise_log1p() -> Result<(), ZyxError> {
    let t = Tensor::from([0.0f32]);
    let result = t.log1p();
    let val = result.cast(DType::F32).item::<f32>();
    assert!((val - 0.0).abs() < 0.01);
    Ok(())
}
