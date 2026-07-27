// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, Scalar, Tensor, ZyxError, bf16};

#[test]
fn bf16_sigmoid() -> Result<(), ZyxError> {
    if !Tensor::dtype_capability(DType::BF16).any() {
        return Ok(());
    }

    let data: [f32; 8] = [0.0, 1.0, -1.0, 2.0, -2.0, 10.0, -10.0, 0.5];
    let x = Tensor::from(data).cast(DType::BF16);
    let z = x.sigmoid();

    let z: Vec<bf16> = z.try_into()?;
    for (&input, actual) in data.iter().zip(z) {
        let expected = 1.0f32 / (1.0f32 + (-input).exp());
        assert!(bf16::from_f32(expected).is_equal(actual));
    }
    Ok(())
}

#[test]
fn bf16_mean() -> Result<(), ZyxError> {
    if !Tensor::dtype_capability(DType::BF16).any() {
        return Ok(());
    }

    let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Tensor::from(data).cast(DType::BF16);
    let mean = x.mean([0])?;

    let mean_val = mean.item::<bf16>();
    let expected = 3.5f32;
    assert!(bf16::from_f32(expected).is_equal(mean_val));
    Ok(())
}

#[test]
fn bf16_binary_mul() -> Result<(), ZyxError> {
    if !Tensor::dtype_capability(DType::BF16).any() {
        return Ok(());
    }

    let a = Tensor::from([1.0f32, 2.0, 3.0, 4.0]).cast(DType::BF16);
    let b = Tensor::from([2.0, 3.0, 4.0, 5.0]).cast(DType::BF16);
    let c = a * b;

    let c: Vec<bf16> = c.try_into()?;
    for (&expected, actual) in [2.0f32, 6.0, 12.0, 20.0].iter().zip(c) {
        assert!(bf16::from_f32(expected).is_equal(actual));
    }
    Ok(())
}

#[test]
fn bf16_add1() -> Result<(), ZyxError> {
    if !Tensor::dtype_capability(DType::BF16).any() {
        return Ok(());
    }

    let c = {
        let a = Tensor::from([bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0)]);
        let b = Tensor::from([bf16::from_f32(4.0), bf16::from_f32(5.0), bf16::from_f32(6.0)]);
        a + b.sin()
    };

    let expected = [1.0f32 + 4.0f32.sin(), 2.0 + 5.0f32.sin(), 3.0 + 6.0f32.sin()];
    let c: Vec<bf16> = c.try_into()?;
    for (i, (&exp, &actual)) in expected.iter().zip(c.iter()).enumerate() {
        assert!(bf16::from_f32(exp).is_equal(actual), "bf16_add1[{i}]: expected={}, actual={}", exp, actual.to_f32());
    }
    Ok(())
}

#[test]
fn bf16_add2() -> Result<(), ZyxError> {
    if !Tensor::dtype_capability(DType::BF16).any() {
        return Ok(());
    }

    let c = {
        let a = Tensor::from([1.0f32, 2.0, 3.0]).cast(DType::BF16);
        let b = Tensor::from([4.0f32, 5.0, 6.0]).cast(DType::BF16);
        a + b.sin()
    };

    let expected = [1.0f32 + 4.0f32.sin(), 2.0 + 5.0f32.sin(), 3.0 + 6.0f32.sin()];
    let c: Vec<bf16> = c.try_into()?;
    for (i, (&exp, &actual)) in expected.iter().zip(c.iter()).enumerate() {
        assert!(bf16::from_f32(exp).is_equal(actual), "bf16_add2[{i}]: expected={}, actual={}", exp, actual.to_f32());
    }
    Ok(())
}

#[test]
fn bf16_add3() -> Result<(), ZyxError> {
    if !Tensor::dtype_capability(DType::BF16).any() {
        return Ok(());
    }

    let a = Tensor::from([bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0)]);
    let b = Tensor::from([bf16::from_f32(4.0), bf16::from_f32(5.0), bf16::from_f32(6.0)]);
    let c = a.cast(DType::F32) + b.sin().cast(DType::F32);
    let c = c.cast(DType::BF16);

    let expected = [1.0f32 + 4.0f32.sin(), 2.0 + 5.0f32.sin(), 3.0 + 6.0f32.sin()];
    let c: Vec<bf16> = c.try_into()?;
    for (i, (&exp, &actual)) in expected.iter().zip(c.iter()).enumerate() {
        assert!(bf16::from_f32(exp).is_equal(actual), "bf16_add3[{i}]: expected={}, actual={}", exp, actual.to_f32());
    }
    Ok(())
}

#[test]
fn bf16_add4() -> Result<(), ZyxError> {
    if !Tensor::dtype_capability(DType::BF16).any() {
        return Ok(());
    }

    let n = 54u32;
    let a_data: Vec<bf16> = (0..n).map(|i| bf16::from_f32(i as f32)).collect();
    let b_data: Vec<bf16> = (0..n).map(|i| bf16::from_f32((n - 1 - i) as f32)).collect();
    let a = Tensor::from(a_data);
    let b = Tensor::from(b_data);
    let c = a + b.sin();

    let c: Vec<bf16> = c.try_into()?;
    for (i, actual) in c.iter().enumerate() {
        let exp = i as f32 + (n as f32 - 1.0 - i as f32).sin();
        assert!(bf16::from_f32(exp).is_equal(*actual), "bf16_add4[{i}]: expected={}, actual={}", exp, actual.to_f32());
    }
    Ok(())
}

#[test]
fn bf16_add5() -> Result<(), ZyxError> {
    if !Tensor::dtype_capability(DType::BF16).any() {
        return Ok(());
    }

    let n = 2048;
    let a_data: Vec<bf16> = (0..n).map(|i| bf16::from_f32(i as f32 / 256.)).collect();
    let b_data: Vec<bf16> = (0..n).map(|i| bf16::from_f32(i as f32 / 256.)).collect();
    let a = Tensor::from(a_data);
    let b = Tensor::from(b_data);
    let c = a + b;

    let c: Vec<bf16> = c.try_into()?;
    for (i, actual) in c.iter().enumerate() {
        let exp = 2.0 * i as f32 / 256.0;
        eprintln!("[{i}] exp={exp}, actual={}", actual.to_f32());
    }
    Ok(())
}

#[test]
fn bf16_matmul_1() -> Result<(), ZyxError> {
    if !Tensor::dtype_capability(DType::BF16).any() {
        return Ok(());
    }

    let m = 17;
    let k = 16;
    let n = 19;

    let x_data: Vec<Vec<bf16>> = (0..m).map(|i| (0..k).map(|j| bf16::from_f32((i as f32 + j as f32) % 10.0)).collect()).collect();
    let y_data: Vec<Vec<bf16>> = (0..k).map(|i| (0..n).map(|j| bf16::from_f32((i as f32 - j as f32) % 10.0)).collect()).collect();

    let x = Tensor::from(x_data.clone());
    let y = Tensor::from(y_data.clone());

    let z = x.dot(y)?;

    let mut expected = vec![vec![bf16::from_f32(0.0); n]; m];
    for i in 0..m {
        for kk in 0..k {
            for j in 0..n {
                let val = x_data[i][kk].to_f32() * y_data[kk][j].to_f32();
                expected[i][j] = bf16::from_f32(expected[i][j].to_f32() + val);
            }
        }
    }
    let z: Vec<bf16> = z.try_into()?;
    let expected: Vec<bf16> = expected.into_iter().flatten().collect();

    for (actual, exp) in z.iter().zip(expected.iter()) {
        assert!(
            bf16::from_f32(exp.to_f32()).is_equal(*actual),
            "bf16_matmul_1: expected={}, actual={}",
            exp.to_f32(),
            actual.to_f32()
        );
    }
    Ok(())
}
