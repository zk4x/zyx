// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use zyx::{DebugMask, Module, Tensor, ZyxError};

#[test]
fn b_arange_asm() -> Result<(), ZyxError> {
    let _guard = Tensor::with_debug(DebugMask::new(16));
    let x = Tensor::arange(0, 100, 1)?;
    Tensor::realize([&x])?;
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
    assert_eq!(gathered, [[3, 1], [5, 5]]);
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

/*#[cfg(not(feature = "wgpu"))]
#[test]
fn scatter_basic() -> Result<(), ZyxError> {
    let x = Tensor::zeros([3, 3], zyx::DType::I32);
    let src = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let indices = Tensor::from([[0, 1, 2], [0, 1, 2]]);
    let result = x.scatter(0, &indices, &src)?;
    assert_eq!(result.shape(), [3, 3]);
    Ok(())
}*/

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_4d_tensor() -> Result<(), ZyxError> {
    let x = Tensor::from([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]);
    let indices = Tensor::from([[[[0], [1]]]]);
    let gathered = x.gather(3, &indices)?;
    assert_eq!(gathered, [[[[1], [4]]]]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_large_tensor_axis0() -> Result<(), ZyxError> {
    let x = Tensor::arange(0, 10000, 1)?;
    let indices = Tensor::from([0u16, 5000, 9999, 0, 9999]);
    let gathered = x.gather(0, &indices)?;
    let result: Vec<i32> = gathered.try_into()?;
    assert_eq!(result, [0, 5000, 9999, 0, 9999]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_large_2d_tensor() -> Result<(), ZyxError> {
    let data: Vec<i32> = (0..10000).collect();
    let x = Tensor::from(data).reshape([100, 100])?;
    let indices = Tensor::from([[0u16, 50, 99], [99, 0, 50]]);
    let gathered = x.gather(1, &indices)?;
    let result: Vec<i32> = gathered.try_into()?;
    assert_eq!(result, [0, 50, 99, 199, 100, 150]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_mnist_like_sampling() -> Result<(), ZyxError> {
    let n = 1000;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let x = Tensor::from(data).reshape([n, 1])?;
    let indices = Tensor::from([0u16, 500, 999, 0, 999, 100, 200]);
    let gathered = x.gather(0, &indices.reshape([7, 1])?)?;
    let result: Vec<f32> = gathered.try_into()?;
    let expected: Vec<f32> = vec![0.0, 500.0, 999.0, 0.0, 999.0, 100.0, 200.0];
    for (a, b) in result.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-5, "a={a}, b={b}");
    }
    Ok(())
}

#[test]
fn slice_range_clamped_to_dim_size() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5]);
    let y = x.slice(0..100)?;
    assert_eq!(y, [1, 2, 3, 4, 5]);
    Ok(())
}

#[test]
fn slice_negative_range_clamped() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5]);
    let y = x.slice(-10..3)?;
    assert_eq!(y, [1, 2, 3]);
    Ok(())
}

#[test]
fn slice_range_to_clamped() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5]);
    let y = x.slice(..100)?;
    assert_eq!(y, [1, 2, 3, 4, 5]);
    Ok(())
}

#[test]
fn slice_invalid_range_end_less_than_start() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5]);
    let result = x.slice(3..2);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn slice_index_out_of_bounds_positive() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3]);
    let result = x.slice(10);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn slice_index_out_of_bounds_negative() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3]);
    let result = x.slice(-10);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn slice_range_start_out_of_bounds() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3]);
    let result = x.slice(10..20);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn rslice_range_clamped_to_dim_size() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let y = x.rslice(-10..)?;
    assert_eq!(y, [[1, 2, 3], [4, 5, 6]]);
    Ok(())
}

#[test]
fn cumsum_large_tensor() -> Result<(), ZyxError> {
    let n = 1000;
    let data: Vec<i32> = std::iter::repeat_n(1, n).collect();
    let x = Tensor::from(data);
    let result = x.cumsum(0)?;
    let result_vec: Vec<i32> = result.try_into()?;
    for (i, &val) in result_vec.iter().enumerate() {
        assert_eq!(val, (i + 1) as i32, "Mismatch at index {i}");
    }
    Ok(())
}

#[test]
fn arange_large_range() -> Result<(), ZyxError> {
    let x = Tensor::arange(0i64, 1000, 1i64)?;
    assert_eq!(x.shape(), [1000]);
    let result: Vec<i64> = x.try_into()?;
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(val, i as i64, "Mismatch at index {i}");
    }
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn one_hot_large_num_classes() -> Result<(), ZyxError> {
    let indices = Tensor::from([0u16, 1, 2]);
    let one_hot = indices.one_hot(100);
    assert_eq!(one_hot.shape(), [3, 100]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_with_one_hot_large_dim() -> Result<(), ZyxError> {
    let n = 500;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let x = Tensor::from(data);
    let indices = Tensor::from([0u16, n as u16 - 1, 0, n as u16 / 2]);
    let gathered = x.gather(0, &indices)?;
    let result: Vec<f32> = gathered.try_into()?;
    let expected = vec![0.0, (n - 1) as f32, 0.0, (n / 2) as f32];
    for (a, b) in result.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-5, "a={a}, b={b}");
    }
    Ok(())
}

#[test]
fn index_select() -> Result<(), ZyxError> {
    // 2D tensor
    let x = Tensor::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);

    // --- Row selection ---
    let rows = Tensor::from([2, 0]); // pick 3rd row, then 1st row
    let y = x.index_select(0, &rows)?;
    assert_eq!(y, [[7, 8, 9], [1, 2, 3]]);

    // --- Column selection ---
    let cols = Tensor::from([2, 0]); // pick 3rd col, then 1st col
    let y = x.index_select(1, &cols)?;
    assert_eq!(y, [[3, 1], [6, 4], [9, 7]]);

    // --- Negative indices ---
    let rows = Tensor::from([-1, -3]); // last row, first row
    let y = x.index_select(0, &rows)?;
    assert_eq!(y, [[7, 8, 9], [1, 2, 3]]);

    let cols = Tensor::from([-2, -1]); // second last col, last col
    let y = x.index_select(1, &cols)?;
    assert_eq!(y, [[2, 3], [5, 6], [8, 9]]);

    // --- Single element selection ---
    let row = Tensor::from([1]);
    let col = Tensor::from([2]);
    let y = x.index_select(0, &row)?.index_select(1, &col)?;
    assert_eq!(y, [[6]]);

    Ok(())
}

#[test]
fn argmax_comprehensive() -> Result<(), ZyxError> {
    // --- 1D tensor ---
    let x1 = Tensor::from([1, 3, 2, 5]);
    assert_eq!(x1.argmax(), 3); // max is 5 at index 3
    let y = x1.argmax_axis(0)?; // Axis=0
    assert_eq!(y, [3]); // same for axis=0

    // --- 2D tensor ---
    let x2 = Tensor::from([[1, 3, 2], [4, 6, 5], [7, 9, 8]]);

    // Flattened argmax
    assert_eq!(x2.argmax(), 7); // max 9 at index 7

    // Axis=0 (column-wise)
    let y = x2.argmax_axis(0)?;
    assert_eq!(y, [2, 2, 2]); // max in each column

    // Axis=1 (row-wise)
    let y = x2.argmax_axis(1)?;
    assert_eq!(y, [1, 1, 1]); // max in each row

    // Negative values
    let x2_neg = Tensor::from([[-1, -3, -2], [-4, -6, -5]]);
    assert_eq!(x2_neg.argmax(), 0); // max -1 at index 0
    let y = x2_neg.argmax_axis(0)?;
    assert_eq!(y, [0, 0, 0]);
    let y = x2_neg.argmax_axis(1)?;
    assert_eq!(y, [0, 0]);

    // Single-element tensor
    let x_single = Tensor::from([[42]]);
    assert_eq!(x_single.argmax(), 0);
    let y = x_single.argmax_axis(0)?;
    assert_eq!(y, [0]);
    let y = x_single.argmax_axis(1)?;
    assert_eq!(y, [0]);

    // --- 3D tensor ---
    let x3 = Tensor::from([[[1, 5, 2], [4, 0, 3]], [[7, 2, 6], [1, 8, 0]]]);

    // Axis=0
    let y = x3.argmax_axis(0)?;
    assert_eq!(y, [[1, 0, 1], [0, 1, 0]]);

    // Axis=1
    let y = x3.argmax_axis(1)?;
    assert_eq!(y, [[1, 0, 1], [0, 1, 0]]);

    // Axis=2
    let y = x3.argmax_axis(2)?;
    assert_eq!(y, [[1, 0], [0, 1]]);

    Ok(())
}
