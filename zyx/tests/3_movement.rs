// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{Tensor, ZyxError};

#[test]
fn reshape_1() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[4, 5, 2, 1], [3, 4, 1, 4]]);
    x = x.reshape([8, 1])?;
    x = x.reshape([1, 2, 1, 4])?;
    x = x.reshape([4, 2])?;
    assert_eq!(x, [[4, 5], [2, 1], [3, 4], [1, 4]]);
    Ok(())
}

#[test]
fn reshape_permute_1() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[4, 5, 2, 1], [3, 4, 1, 4]]);
    x = x.reshape([8, 1])?;
    x = x.reshape([1, 2, 1, 4])?.permute([2, 3, 1, 0])?;
    x = x.reshape([4, 2])?.cast(zyx::DType::F32).exp2().cast(zyx::DType::I32);
    assert_eq!(x, [[16, 8], [32, 16], [4, 2], [2, 16]]);
    Ok(())
}

#[test]
fn expand_1() -> Result<(), ZyxError> {
    let a = Tensor::from([[1, 2], [3, 4]]).reshape([1, 1, 1, 4])?;
    let b = Tensor::from([[5, 6], [7, 8]]).reshape([1, 1, 4, 1])?;
    let c = a + b;
    assert_eq!(c, [[[[6, 7, 8, 9], [7, 8, 9, 10], [8, 9, 10, 11], [9, 10, 11, 12]]]]);
    Ok(())
}

#[test]
fn permute_2() -> Result<(), ZyxError> {
    let x = Tensor::from([[4, 5, 2, 1], [3, 4, 1, 4]]);
    let y = x.permute([1, 0])?;
    assert_eq!(y, [[4, 3], [5, 4], [2, 1], [1, 4]]);
    Ok(())
}

#[test]
fn pad_1() -> Result<(), ZyxError> {
    let a = Tensor::from([[1, 2], [3, 4]]);
    let c = a.pad_zeros([(0, 2), (0, 0)])?;
    assert_eq!(c, [[1, 2], [3, 4], [0, 0], [0, 0]]);
    Ok(())
}

#[test]
fn pad_2() -> Result<(), ZyxError> {
    let a = Tensor::from([[1i32, 2], [3, 4]]).reshape([1, 1, 2, 2])?;
    let b = Tensor::from([[5, 6], [7, 8]]).reshape([1, 1, 1, 4])?;
    let c = a.pad_zeros([(0, 0), (0, 0), (0, 2), (0, 2)])? + b;
    assert_eq!(c, [[[[6i32, 8, 7, 8], [8, 10, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]]]]);
    Ok(())
}
