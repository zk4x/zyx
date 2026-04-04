// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use zyx::{Tensor, ZyxError};

#[test]
fn slice_single_index() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5]);
    let y = x.slice(0)?;
    assert_eq!(y, [1]);
    Ok(())
}

#[test]
fn slice_negative_index() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5]);
    let y = x.slice(-1)?;
    assert_eq!(y, [5]);
    Ok(())
}

#[test]
fn slice_range() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5]);
    let y = x.slice(1..4)?;
    assert_eq!(y, [2, 3, 4]);
    Ok(())
}

#[test]
fn slice_range_from() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5]);
    let y = x.slice(2..)?;
    assert_eq!(y, [3, 4, 5]);
    Ok(())
}

#[test]
fn slice_range_to() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5]);
    let y = x.slice(..3)?;
    assert_eq!(y, [1, 2, 3]);
    Ok(())
}

#[test]
fn slice_range_full() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5]);
    let y = x.slice(..)?;
    assert_eq!(y, [1, 2, 3, 4, 5]);
    Ok(())
}

#[test]
fn slice_2d_single_index() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let y = x.slice(0)?;
    assert_eq!(y, [1, 2, 3]);
    Ok(())
}

#[test]
fn slice_2d_tuple_index() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let y = x.slice((0, 1))?;
    assert_eq!(y, [2]);
    Ok(())
}

#[test]
fn slice_2d_range() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let y = x.slice((.., 1..3))?;
    assert_eq!(y, [[2, 3], [5, 6]]);
    Ok(())
}

#[test]
fn slice_2d_partial() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let y = x.slice((0, ..))?;
    assert_eq!(y, [1, 2, 3]);
    Ok(())
}

#[test]
fn rslice_basic() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let y = x.rslice(-1..)?;
    assert_eq!(y, [[3], [6]]);
    Ok(())
}

#[test]
fn rslice_2d() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let y = x.rslice(..)?;
    assert_eq!(y, [[1, 2, 3], [4, 5, 6]]);
    Ok(())
}

#[test]
fn diagonal_2d() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape([3, 3])?;
    let d = x.diagonal();
    assert_eq!(d, [1, 5, 9]);
    Ok(())
}

#[test]
fn slice_3d() -> Result<(), ZyxError> {
    let data: [i32; 24] = (1..=24).collect::<Vec<_>>().try_into().unwrap();
    let x = Tensor::from(data).reshape([2, 3, 4])?;
    let y = x.slice((1, .., ..))?;
    assert_eq!(y, [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]);
    Ok(())
}

#[test]
fn slice_chain() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let y = x.slice((.., 1..3))?.slice(0)?;
    assert_eq!(y, [2, 3]);
    Ok(())
}

#[test]
fn slice_error_rank_mismatch() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3]);
    let result = x.slice((0, 0));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn slice_error_out_of_bounds() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3]);
    let result = x.slice(10);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn slice_last_element() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5]);
    let y = x.slice(-1)?;
    assert_eq!(y, [5]);
    Ok(())
}

#[test]
fn slice_negative_range() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5]);
    let y = x.slice(-3..)?;
    assert_eq!(y, [3, 4, 5]);
    Ok(())
}

#[test]
fn slice_mixed_tuple() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let y = x.slice((1, 0..2))?;
    assert_eq!(y, [4, 5]);
    Ok(())
}

#[test]
fn slice_all_dims_explicit() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4]);
    let y = x.slice(..)?;
    assert_eq!(y, [1, 2, 3, 4]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_basic() -> Result<(), ZyxError> {
    let x = Tensor::from([[10u16, 20, 30, 40, 50], [11, 21, 31, 41, 51], [12, 22, 32, 42, 52]]);
    let indices = Tensor::from([[0u16, 2, 4], [1, 3, 0], [4, 1, 2]]);
    let gathered = x.gather(1, &indices)?;
    assert_eq!(gathered, [[10u16, 30, 50], [21, 41, 11], [52, 22, 32]]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_axis0() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let indices = Tensor::from([[0, 1], [1, 2], [0, 2]]);
    let gathered = x.gather(0, &indices)?;
    assert_eq!(gathered, [[1, 5], [4, 8], [1, 8]]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_1d() -> Result<(), ZyxError> {
    let x = Tensor::from([10, 20, 30, 40, 50]);
    let indices = Tensor::from([0u16, 2, 4, 1]);
    let gathered = x.gather(0, &indices)?;
    assert_eq!(gathered, [10, 30, 50, 20]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_error_wrong_axis() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2], [3, 4]]);
    let indices = Tensor::from([[0, 1], [1, 0]]);
    let result = x.gather(2, &indices);
    assert!(result.is_err());
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_negative_indices() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let indices = Tensor::from([[-1, 0], [1, -2]]);
    let gathered = x.gather(1, &indices)?;
    assert_eq!(gathered, [[0, 1], [5, 0]]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_duplicate_indices() -> Result<(), ZyxError> {
    let x = Tensor::from([10, 20, 30]);
    let indices = Tensor::from([0u16, 0, 1, 1, 2, 0]);
    let gathered = x.gather(0, &indices)?;
    assert_eq!(gathered, [10, 10, 20, 20, 30, 10]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_3d_tensor() -> Result<(), ZyxError> {
    let x = Tensor::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    let indices = Tensor::from([[[0], [1]], [[1], [0]]]);
    let gathered = x.gather(2, &indices)?;
    assert_eq!(gathered, [[[1], [4]], [[6], [7]]]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_axis_minus1() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let indices = Tensor::from([[0, 2], [1, 0]]);
    let gathered = x.gather(-1, &indices)?;
    assert_eq!(gathered, [[1, 3], [5, 4]]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_axis_minus2() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2], [3, 4], [5, 6]]);
    let indices = Tensor::from([[0], [0], [2]]);
    let gathered = x.gather(-2, &indices)?;
    assert_eq!(gathered.shape(), [3, 1]);
    let result: Vec<i32> = gathered.flatten(..)?.try_into()?;
    assert_eq!(result, [1, 1, 5]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_f32_dtype() -> Result<(), ZyxError> {
    let x = Tensor::from([1.0f32, 2.0, 3.0, 4.0]);
    let indices = Tensor::from([0u16, 2, 3, 1]);
    let gathered = x.gather(0, &indices)?;
    let result: Vec<f32> = gathered.try_into()?;
    assert_eq!(result, [1.0, 3.0, 4.0, 2.0]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_single_element() -> Result<(), ZyxError> {
    let x = Tensor::from([42]);
    let indices = Tensor::from([0u16]);
    let gathered = x.gather(0, &indices)?;
    assert_eq!(gathered, [42]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_all_indices() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5]);
    let indices = Tensor::from([0u16, 1, 2, 3, 4]);
    let gathered = x.gather(0, &indices)?;
    assert_eq!(gathered, [1, 2, 3, 4, 5]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_indices_larger_than_axis() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2], [3, 4]]);
    let indices = Tensor::from([[0, 0, 1], [1, 1, 0]]);
    let gathered = x.gather(1, &indices)?;
    assert_eq!(gathered, [[1, 1, 2], [4, 4, 3]]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_4d_tensor() -> Result<(), ZyxError> {
    let x = Tensor::from([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]);
    let indices = Tensor::from([[[[0], [1]]]]);
    let gathered = x.gather(3, &indices)?;
    assert_eq!(gathered, [[[[1], [4]]]]);
    Ok(())
}
