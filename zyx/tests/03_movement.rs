// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

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

#[test]
fn flip_1() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4]);
    let y = x.flip([0])?;
    assert_eq!(y, [4, 3, 2, 1]);
    Ok(())
}

#[test]
fn flip_2() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let y = x.flip([0, 1])?;
    assert_eq!(y, [[6, 5, 4], [3, 2, 1]]);
    Ok(())
}

#[test]
fn flip_3() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let y = x.flip([-1])?;
    assert_eq!(y, [[3, 2, 1], [6, 5, 4]]);
    Ok(())
}

#[test]
fn flip_errors() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3]);
    assert!(x.flip(std::iter::empty::<i32>()).is_err());
    assert!(x.flip([1]).is_err());
    Ok(())
}

#[test]
fn contiguous_1() -> Result<(), ZyxError> {
    // Eager chain: contiguous breaks fusion but preserves the value.
    let x = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let y = (x + 1).contiguous()?;
    assert_eq!(y, [[2, 3, 4], [5, 6, 7]]);
    Ok(())
}

#[test]
fn contiguous_2() -> Result<(), ZyxError> {
    // After a permute, contiguous materializes the transposed value.
    let x = Tensor::from([[1, 2], [3, 4]]);
    let t = x.permute([1, 0])?;
    let c = t.contiguous()?;
    assert_eq!(c, [[1, 3], [2, 4]]);
    Ok(())
}

#[test]
fn contiguous_3() -> Result<(), ZyxError> {
    // Already realized tensor: contiguous is a no-op returning the same value.
    let x = Tensor::from([[1, 2], [3, 4]]);
    let c = x.contiguous()?;
    assert_eq!(c, [[1, 2], [3, 4]]);
    Ok(())
}

#[test]
fn contiguous_4() -> Result<(), ZyxError> {
    // Computed chain followed by contiguous then more computation.
    let x = Tensor::from([1f32, 2., 3.]);
    let y = x.exp2();
    let c = y.contiguous()?;
    let z = c * 2;
    assert_eq!(z, [4.0f32, 8.0, 16.0]);
    Ok(())
}

#[test]
fn narrow_1() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
    let y = x.narrow(0, 1i64, 2i64)?;
    assert_eq!(y, [[5, 6, 7, 8], [9, 10, 11, 12]]);
    // Narrow a single axis of the result with a negative axis.
    let z = y.narrow(-1, 1i64, 2i64)?;
    assert_eq!(z, [[6, 7], [10, 11]]);
    Ok(())
}

#[test]
fn stack_1() -> Result<(), ZyxError> {
    let t0 = Tensor::from([1, 2]);
    let t1 = Tensor::from([3, 4]);
    let s = Tensor::stack_axis([&t0, &t1], 0)?;
    assert_eq!(s, [[1, 2], [3, 4]]);
    let s1 = Tensor::stack_axis([&t0, &t1], 1)?;
    assert_eq!(s1, [[1, 3], [2, 4]]);
    Ok(())
}

#[test]
fn assign_narrow_same_root() -> Result<(), ZyxError> {
    let base = Tensor::from([0f32, 0f32, 0f32, 0f32, 0f32, 0f32]);
    let src1 = Tensor::from([1f32, 2f32]);
    let src2 = Tensor::from([3f32, 4f32]);
    base.narrow(0, 0i64, 2i64)?.assign(&src1)?;
    base.narrow(0, 2i64, 2i64)?.assign(&src2)?;
    let out: Vec<f32> = base.try_into()?;
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0]);
    Ok(())
}
