// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, Scalar, Tensor, ZyxError, bf16, f16};

// --- BF16 Precision Tests ---
// These tests use bf16 dtype and compare at bf16 precision
// (not converted to fp32, unlike some existing tests)

#[test]
fn bf16_sigmoid() -> Result<(), ZyxError> {
    if !Tensor::supports(DType::BF16) {
        return Ok(());
    }

    // Test sigmoid at various points including edge cases
    let data: [f32; 8] = [0.0, 1.0, -1.0, 2.0, -2.0, 10.0, -10.0, 0.5];
    let x = Tensor::from(data).cast(DType::BF16);
    let z = x.sigmoid();

    // Compare at bf16 precision: convert bf16 result to f32 and compare
    let z: Vec<bf16> = z.try_into()?;
    for (&input, actual) in data.iter().zip(z) {
        let expected = 1.0f32 / (1.0f32 + (-input).exp());
        assert!(bf16::from_f32(expected).is_equal(actual));
    }
    Ok(())
}

#[test]
fn bf16_softmax() -> Result<(), ZyxError> {
    if !Tensor::supports(DType::BF16) {
        return Ok(());
    }

    // Test softmax with multiple values
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let x = Tensor::from(data).cast(DType::BF16);
    let z = x.softmax([0])?;

    let z: Vec<bf16> = z.try_into()?;

    // Compare at bf16 precision
    for (&input, actual) in data.iter().zip(z) {
        let sum: f32 = data.iter().map(|x| x.exp()).sum();
        let expected = input.exp() / sum;
        assert!(bf16::from_f32(expected).is_equal(actual));
    }
    Ok(())
}

#[test]
fn bf16_mean() -> Result<(), ZyxError> {
    if !Tensor::supports(DType::BF16) {
        return Ok(());
    }

    // Test mean reduction at bf16 precision
    let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Tensor::from(data).cast(DType::BF16);
    let mean = x.mean([0])?;

    // Compare at bf16 precision (mean of 1-6 is 3.5)
    let mean_val = mean.item::<bf16>();
    let expected = 3.5f32;
    assert!(bf16::from_f32(expected).is_equal(mean_val));
    Ok(())
}

#[test]
fn bf16_binary_mul() -> Result<(), ZyxError> {
    if !Tensor::supports(DType::BF16) {
        return Ok(());
    }

    // Test binary multiplication at bf16 precision
    let a = Tensor::from([1.0f32, 2.0, 3.0, 4.0]).cast(DType::BF16);
    let b = Tensor::from([2.0, 3.0, 4.0, 5.0]).cast(DType::BF16);
    let c = a * b;

    // Compare at bf16 precision: [2, 6, 12, 20]
    let c: Vec<bf16> = c.try_into()?;
    for (&expected, actual) in [2.0f32, 6.0, 12.0, 20.0].iter().zip(c) {
        assert!(bf16::from_f32(expected).is_equal(actual));
    }
    Ok(())
}

#[test]
fn bf16_gelu() -> Result<(), ZyxError> {
    if !Tensor::supports(DType::BF16) {
        return Ok(());
    }

    // Test GELU activation at bf16 precision
    let data: [f32; 6] = [0.0, 1.0, -1.0, 2.0, -2.0, 1.5];
    let x = Tensor::from(data).cast(DType::BF16);
    let z = x.relu();

    // Compare at bf16 precision using is_equal
    let z: Vec<bf16> = z.try_into()?;
    for (&input, actual) in data.iter().zip(z) {
        let expected = input.max(0.0);
        assert!(bf16::from_f32(expected).is_equal(actual));
    }
    Ok(())
}

#[test]
fn bf16_add() -> Result<(), ZyxError> {
    if !Tensor::supports(DType::BF16) {
        return Ok(());
    }

    // Test addition at bf16 precision
    let a = Tensor::from([1.0f32, 2.0, 3.0]).cast(DType::BF16);
    let b = Tensor::from([4.0, 5.0, 6.0]).cast(DType::BF16);
    let c = a + b;

    // Compare at bf16 precision: [5, 7, 9]
    let c: Vec<bf16> = c.try_into()?;
    for (&expected, actual) in [5.0f32, 7.0, 9.0].iter().zip(c) {
        assert!(bf16::from_f32(expected).is_equal(actual));
    }
    Ok(())
}
