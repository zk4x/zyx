// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, Scalar, Tensor, ZyxError};

#[test]
fn add_1() -> Result<(), ZyxError> {
    let datax: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let datay: [f32; 10] = [2.772, -8.327, 1.945, 9.286, 3.989, 8.105, -5.307, 2.865, 3.106, 3.111];
    let x = Tensor::from(datax);
    let y = Tensor::from(datay);
    let z = x + y;
    let dataz: Vec<f32> = z.try_into()?;
    for ((x, y), z) in datax.iter().zip(datay).zip(dataz) {
        assert_eq!(x + y, z);
    }
    Ok(())
}

#[test]
fn add_2() -> Result<(), ZyxError> {
    let x = Tensor::from([2i32, 3, 5, 1, 6]);
    let y = Tensor::from([7i32, 2, 5, 1, 2]);
    let z = &x + y + &x + &x;
    assert_eq!(z, [13i32, 11, 20, 4, 20]);
    Ok(())
}

#[test]
fn add_3() -> Result<(), ZyxError> {
    let datax: [f32; 32] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
        2.301, -1.456, 0.987, 3.421, -6.789, 1.234, -0.567, 4.890, -2.345, 7.654,
        0.111, -9.876, 5.432, -1.234, 8.765, -3.210, 6.543, -4.321, 2.109, -5.678,
        0.999, -0.001,
    ];
    let datay: [f32; 32] = [
        2.772, -8.327, 1.945, 9.286, 3.989, 8.105, -5.307, 2.865, 3.106, 3.111,
        -4.567, 2.345, 6.789, -1.234, 5.678, -3.456, 7.890, -2.109, 4.321, -6.543,
        1.234, 3.456, -7.890, 2.109, -5.678, 4.321, -6.543, 1.098, -3.210, 5.678,
        -0.999, 0.001,
    ];
    let x = Tensor::from(datax);
    let y = Tensor::from(datay);
    let z = x + y;
    let dataz: Vec<f32> = z.try_into()?;
    for ((x, y), z) in datax.iter().zip(datay).zip(dataz) {
        assert_eq!(x + y, z);
    }
    Ok(())
}

#[test]
fn sub() -> Result<(), ZyxError> {
    let datax: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let datay: [f32; 10] = [2.772, -8.327, 1.945, 9.286, 3.989, 8.105, -5.307, 2.865, 3.106, 3.111];
    let x = Tensor::from(datax);
    let y = Tensor::from(datay);
    let z = x - y;
    let dataz: Vec<f32> = z.try_into()?;
    for ((x, y), z) in datax.iter().zip(datay).zip(dataz) {
        assert_eq!(x - y, z);
    }
    Ok(())
}

#[test]
fn mul() -> Result<(), ZyxError> {
    let datax: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let datay: [f32; 10] = [2.772, -8.327, 1.945, 9.286, 3.989, 8.105, -5.307, 2.865, 3.106, 3.111];
    let x = Tensor::from(datax);
    let y = Tensor::from(datay);
    let z = x * y;
    let dataz: Vec<f32> = z.try_into()?;
    for ((x, y), z) in datax.iter().zip(datay).zip(dataz) {
        assert_eq!(x * y, z);
    }
    Ok(())
}

#[test]
fn div() -> Result<(), ZyxError> {
    let datax: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let datay: [f32; 10] = [2.772, -8.327, 1.945, 9.286, 3.989, 8.105, -5.307, 2.865, 3.106, 3.111];
    let x = Tensor::from(datax);
    let y = Tensor::from(datay);
    let z = x / y;
    let dataz: Vec<f32> = z.try_into()?;
    for ((x, y), z) in datax.iter().zip(datay).zip(dataz) {
        assert!((x / y - z).abs() < 0.00001);
    }
    Ok(())
}

#[test]
fn pow1() -> Result<(), ZyxError> {
    let datax: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let datay: [f32; 10] = [2.772, -8.327, 1.945, 9.286, 3.989, 8.105, -5.307, 2.865, 3.106, 3.111];
    let x = Tensor::from(datax);
    let y = Tensor::from(datay);
    let z = x.pow(y)?;
    let dataz: Vec<f32> = z.try_into()?;
    for ((x, y), z) in datax.iter().zip(datay).zip(dataz) {
        let x = x.pow(y);
        assert!(x.is_equal(z));
    }
    Ok(())
}

#[test]
fn maximum() -> Result<(), ZyxError> {
    let datax: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let datay: [f32; 10] = [2.772, -8.327, 1.945, 9.286, 3.989, 8.105, -5.307, 2.865, 3.106, 3.111];
    let x = Tensor::from(datax);
    let y = Tensor::from(datay);
    let z = x.maximum(y)?;
    let dataz: Vec<f32> = z.try_into()?;
    for ((x, y), z) in datax.iter().zip(datay).zip(dataz) {
        assert_eq!(x.max(y), z);
    }
    Ok(())
}

#[test]
fn cmplt() -> Result<(), ZyxError> {
    let datax: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let datay: [f32; 10] = [2.772, -8.327, 1.945, 9.286, 3.989, 8.105, -5.307, 2.865, 3.106, 3.111];
    let x = Tensor::from(datax);
    let y = Tensor::from(datay);
    let z = x.cmplt(y)?.cast(zyx::DType::U32);
    let dataz: Vec<u32> = z.try_into()?;
    for ((x, y), z) in datax.iter().zip(datay).zip(dataz) {
        assert_eq!(x.cmplt(y) as u32, z);
    }
    Ok(())
}

#[test]
fn pow_neg_f64() -> Result<(), ZyxError> {
    if !Tensor::supports(DType::F64).pow() {
        return Ok(());
    }
    let x = Tensor::from([-1.5f64, 2.0, 0.5]);
    let y = Tensor::from([2.0f64, 2.0, 0.5]);
    let z = x.pow(y)?;
    println!("{z}");
    Ok(())
}
