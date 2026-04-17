// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{Tensor, ZyxError, tensor::Axis};
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo};

/// Panics on indexing, with a helpful message directing to `.slice(...)`.
impl<I> std::ops::Index<I> for Tensor {
    type Output = Tensor;

    fn index(&self, _index: I) -> &Self::Output {
        panic!(
            "Tensor does not support indexing with `[]` because rust only allows indexing on referece types. \
             Use `.slice(...)` instead, which supports ranges, integers, and tuples. \
             Example: tensor.slice((0..3, -1))"
        );
    }
}

impl Tensor {
    /// Indexes into a tensor using flexible and expressive range-based access.
    ///
    /// This function provides ergonomic slicing capabilities for tensors,
    /// supporting various index types such as integers, ranges, tuples, vectors,
    /// and slices. It returns a new tensor that is a view of the original tensor
    /// based on the provided indices.
    ///
    /// # Arguments
    ///
    /// * `index` - such as:
    ///   - A single integer (e.g., `0`, `-1`)
    ///   - A `Range` (e.g., `0..3`, `2..`, `..5`, `..`)
    ///   - A tuple of integers/ranges for multi-dimensional access (e.g., `(0, 1..3, -1, ..)`), up to 8D
    ///   - A `Vec<Range<i32>>` for dynamic indexing
    ///   - A slice `&[impl IntoRange]` for general-purpose indexing
    ///
    /// # Returns
    ///
    /// * `Result<Tensor, ZyxError>` — The sliced tensor view if indexing is valid,
    ///   or a [`ZyxError::ShapeError`] if any index is out of bounds or mismatched.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zyx::{Tensor, DType};
    /// let x = Tensor::randn([3, 4, 5], DType::F32)?;
    ///
    /// // Select first item from first dimension
    /// let a = x.slice(0)?;
    ///
    /// // Select last element along the last axis
    /// let b = x.slice((.., .., -1))?;
    ///
    /// // Slice second dimension between index 1 and 3
    /// let c = x.slice((0, 1..3, ..))?;
    ///
    /// // Chain indexing calls
    /// let d = x.slice((0, .., -1))?.slice(0)?;
    ///
    /// // Use a slice of ranges
    /// let slice = [0..2, 1..4];
    /// let e = x.slice(slice)?;
    ///
    /// // Use a vector of ranges dynamically
    /// let ranges = vec![0..2, 0..4, 1..5];
    /// let f = x.slice(ranges)?;
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Notes
    ///
    /// - Negative indexing is supported (e.g., `-1` is the last element).
    /// - Omitted dimensions are preserved in the output.
    /// - Useful for flexible slicing, batching, and masking operations.
    ///
    /// # Errors
    ///
    /// Returns a [`ZyxError::ShapeError`] if the indices are invalid, out of bounds,
    /// or don't match the tensor's dimensionality.
    pub fn slice(&self, index: impl IntoIndex) -> Result<Tensor, ZyxError> {
        let shape = self.shape();
        let rank = shape.len();

        let mut squeeze_axes: Vec<Axis> = Vec::new();
        let index = index.into_index();
        let padding_len = index.len();

        if rank < padding_len {
            return Err(ZyxError::shape_error(
                format!("Slice with {padding_len} indices, but tensor has rank {rank}").into(),
            ));
        }

        //let padding = std::iter::repeat_n((0, 0), rank - padding_len);
        //print!("shape={shape:?}");

        let padding = index
            .zip(shape.into_iter())
            .enumerate()
            .map(|(axis, (dim_index, dim_size))| {
                let dim_size = dim_size as i64;
                match dim_index {
                    DimIndex::Range { start, end } => {
                        let s = if start < 0 { (start + dim_size).max(0) } else { start };
                        let s = s.min(dim_size);
                        let e = if end > dim_size {
                            dim_size
                        } else if end < 0 {
                            (end + dim_size).max(0)
                        } else {
                            end
                        };
                        let e = e.min(dim_size).max(0);
                        if e < s {
                            return Err(ZyxError::shape_error(
                                format!("Slice range end {e} is less than start {s} for dimension {axis}").into(),
                            ));
                        }
                        Ok((-(s as i64), -((dim_size as i64) - e as i64)))
                    }
                    DimIndex::Index(i) => {
                        squeeze_axes.push(axis as i32);
                        let i = if i < 0 { i + dim_size } else { i };
                        if i < 0 || i >= dim_size {
                            return Err(ZyxError::shape_error(
                                format!("Index {i} out of bounds for dimension {axis} of size {dim_size}").into(),
                            ));
                        }
                        Ok((-(i as i64), -((dim_size as i64) - i as i64 - 1)))
                    }
                    DimIndex::RangeFull => Ok((0i64, 0i64)),
                    DimIndex::RangeFrom { start } => {
                        let s = if start < 0 { (start + dim_size).max(0) } else { start };
                        let s = s.min(dim_size);
                        Ok((-(s as i64), 0i64))
                    }
                    DimIndex::RangeTo { end } => {
                        let e = if end > dim_size {
                            dim_size
                        } else if end < 0 {
                            (end + dim_size).max(0)
                        } else {
                            end
                        };
                        let e = e.min(dim_size).max(0);
                        Ok((0i64, -((dim_size as i64) - e as i64)))
                    }
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        //let padding_vec: Vec<(i32, i32)> = padding.into_iter().collect();
        //println!("padding={padding_vec:?}");

        let mut result = self.pad_zeros(padding)?;
        result = result.squeeze(squeeze_axes);

        Ok(result)
    }

    /// Same as [Tensor::slice], but instead of indexing from first dimensions, it indexes from last dimensions.
    ///
    /// # Errors
    ///
    /// Returns error if the index is invalid for the tensor shape.
    #[allow(clippy::missing_panics_doc)]
    pub fn rslice(&self, index: impl IntoIndex) -> Result<Tensor, ZyxError> {
        let shape = self.shape();
        let rank = shape.len();
        //print!("shape={shape:?}");

        let mut squeeze_axes: Vec<Axis> = Vec::new();
        let index = index.into_index();
        let padding_len = index.len();

        if padding_len > rank {
            return Err(ZyxError::shape_error(
                format!("Index length {padding_len} > rank {rank}").into(),
            ));
        }

        let padding = index
            .zip(shape.into_iter().rev())
            .enumerate()
            .map(|(axis, (dim_index, dim_size))| {
                let dim_size = dim_size as i64;
                match dim_index {
                    DimIndex::Range { start, end } => {
                        let s = if start < 0 { (start + dim_size).max(0) } else { start };
                        let s = s.min(dim_size);
                        let e = if end > dim_size {
                            dim_size
                        } else if end < 0 {
                            (end + dim_size).max(0)
                        } else {
                            end
                        };
                        let e = e.min(dim_size).max(0);
                        if e < s {
                            return Err(ZyxError::shape_error(
                                format!("Slice range end {e} is less than start {s} for dimension {axis}").into(),
                            ));
                        }
                        Ok((-(s as i64), -((dim_size as i64) - e as i64)))
                    }
                    DimIndex::Index(i) => {
                        squeeze_axes.push(axis as i32);
                        let i = if i < 0 { i + dim_size } else { i };
                        if i < 0 || i >= dim_size {
                            return Err(ZyxError::shape_error(
                                format!("Index {i} out of bounds for dimension {axis} of size {dim_size}").into(),
                            ));
                        }
                        Ok((-(i as i64), -((dim_size as i64) - i as i64 - 1)))
                    }
                    DimIndex::RangeFull => Ok((0i64, 0i64)),
                    DimIndex::RangeFrom { start } => {
                        let s = if start < 0 { (start + dim_size).max(0) } else { start };
                        let s = s.min(dim_size);
                        Ok((-(s as i64), 0i64))
                    }
                    DimIndex::RangeTo { end } => {
                        let e = if end > dim_size {
                            dim_size
                        } else if end < 0 {
                            (end + dim_size).max(0)
                        } else {
                            end
                        };
                        let e = e.min(dim_size).max(0);
                        Ok((0i64, -((dim_size as i64) - e as i64)))
                    }
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let padding = padding
            .into_iter()
            .chain(std::iter::repeat_n((0i64, 0i64), rank - padding_len));

        let mut padding_vec: Vec<(i64, i64)> = padding.into_iter().collect();
        padding_vec.reverse();
        //println!("padding={padding_vec:?}");

        let mut result = self.pad_zeros(padding_vec)?;
        result = result.squeeze(squeeze_axes);

        Ok(result)
    }

    /// Returns a tensor containing only the diagonal elements of this tensor.
    ///
    /// The diagonal is obtained by flattening the input tensor, padding it with zeros to make its last dimension size equal
    /// to the number of rows or columns in the original tensor, reshaping it into a 2D matrix, and then extracting the diagonal.
    ///
    /// # Returns
    ///
    /// * A new tensor containing only the diagonal elements of this tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let arr = Tensor::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape([3, 3])?;
    /// assert_eq!(arr.diagonal(), [1, 5, 9]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn diagonal(&self) -> Tensor {
        let n = *self.shape().last().expect("Shape in invalid state. Internal bug.");
        self.flatten(..)
            .unwrap()
            .rpad_zeros([(0i64, i64::try_from(n).unwrap())])
            .unwrap()
            .reshape([n, n + 1])
            .unwrap()
            .slice((.., 0))
            .unwrap()
            .flatten(..)
            .unwrap()
    }
}

/// Dim index
#[derive(Clone, Debug)]
pub enum DimIndex {
    /// Single index
    Index(i64),
    /// Range
    Range { start: i64, end: i64 },
    /// Range from
    RangeFrom { start: i64 },
    /// Range to
    RangeTo { end: i64 },
    /// Range full
    RangeFull,
}

/// Into index
pub trait IntoIndex {
    fn into_index(self) -> impl Iterator<Item = DimIndex> + ExactSizeIterator + DoubleEndedIterator;
}

impl From<isize> for DimIndex {
    fn from(val: isize) -> DimIndex {
        DimIndex::Index(val as i64)
    }
}

impl From<i32> for DimIndex {
    fn from(val: i32) -> DimIndex {
        DimIndex::Index(val as i64)
    }
}

impl From<i64> for DimIndex {
    fn from(val: i64) -> DimIndex {
        DimIndex::Index(val)
    }
}

impl From<usize> for DimIndex {
    fn from(val: usize) -> DimIndex {
        DimIndex::Index(val as i64)
    }
}

impl From<u64> for DimIndex {
    fn from(val: u64) -> DimIndex {
        DimIndex::Index(val as i64)
    }
}

impl From<Range<isize>> for DimIndex {
    fn from(val: Range<isize>) -> DimIndex {
        DimIndex::Range { start: val.start as i64, end: val.end as i64 }
    }
}

impl From<Range<i32>> for DimIndex {
    fn from(val: Range<i32>) -> DimIndex {
        DimIndex::Range { start: val.start as i64, end: val.end as i64 }
    }
}

impl From<Range<i64>> for DimIndex {
    fn from(val: Range<i64>) -> DimIndex {
        DimIndex::Range { start: val.start, end: val.end }
    }
}

impl From<Range<usize>> for DimIndex {
    fn from(val: Range<usize>) -> DimIndex {
        DimIndex::Range { start: val.start as i64, end: val.end as i64 }
    }
}

impl From<Range<u64>> for DimIndex {
    fn from(val: Range<u64>) -> DimIndex {
        DimIndex::Range { start: val.start as i64, end: val.end as i64 }
    }
}

impl From<RangeInclusive<isize>> for DimIndex {
    fn from(val: RangeInclusive<isize>) -> DimIndex {
        DimIndex::Range { start: *val.start() as i64, end: (*val.end() as i64) + 1 }
    }
}

impl From<RangeInclusive<i32>> for DimIndex {
    fn from(val: RangeInclusive<i32>) -> DimIndex {
        DimIndex::Range { start: *val.start() as i64, end: (*val.end() as i64) + 1 }
    }
}

impl From<RangeInclusive<i64>> for DimIndex {
    fn from(val: RangeInclusive<i64>) -> DimIndex {
        DimIndex::Range { start: *val.start(), end: val.end() + 1 }
    }
}

impl From<RangeInclusive<usize>> for DimIndex {
    fn from(val: RangeInclusive<usize>) -> DimIndex {
        DimIndex::Range { start: *val.start() as i64, end: (*val.end() as i64) + 1 }
    }
}

impl From<RangeInclusive<u64>> for DimIndex {
    fn from(val: RangeInclusive<u64>) -> DimIndex {
        DimIndex::Range { start: *val.start() as i64, end: (*val.end() as i64) + 1 }
    }
}

impl From<RangeFrom<isize>> for DimIndex {
    fn from(val: RangeFrom<isize>) -> DimIndex {
        DimIndex::RangeFrom { start: val.start as i64 }
    }
}

impl From<RangeFrom<i32>> for DimIndex {
    fn from(val: RangeFrom<i32>) -> DimIndex {
        DimIndex::RangeFrom { start: val.start as i64 }
    }
}

impl From<RangeFrom<i64>> for DimIndex {
    fn from(val: RangeFrom<i64>) -> DimIndex {
        DimIndex::RangeFrom { start: val.start }
    }
}

impl From<RangeFrom<usize>> for DimIndex {
    fn from(val: RangeFrom<usize>) -> DimIndex {
        DimIndex::RangeFrom { start: val.start as i64 }
    }
}

impl From<RangeFrom<u64>> for DimIndex {
    fn from(val: RangeFrom<u64>) -> DimIndex {
        DimIndex::RangeFrom { start: val.start as i64 }
    }
}

impl From<RangeTo<isize>> for DimIndex {
    fn from(val: RangeTo<isize>) -> DimIndex {
        DimIndex::RangeTo { end: val.end as i64 }
    }
}

impl From<RangeTo<i32>> for DimIndex {
    fn from(val: RangeTo<i32>) -> DimIndex {
        DimIndex::RangeTo { end: val.end as i64 }
    }
}

impl From<RangeTo<i64>> for DimIndex {
    fn from(val: RangeTo<i64>) -> DimIndex {
        DimIndex::RangeTo { end: val.end }
    }
}

impl From<RangeTo<usize>> for DimIndex {
    fn from(val: RangeTo<usize>) -> DimIndex {
        DimIndex::RangeTo { end: val.end as i64 }
    }
}

impl From<RangeTo<u64>> for DimIndex {
    fn from(val: RangeTo<u64>) -> DimIndex {
        DimIndex::RangeTo { end: val.end as i64 }
    }
}

impl From<RangeFull> for DimIndex {
    fn from(_val: RangeFull) -> DimIndex {
        DimIndex::RangeFull
    }
}

impl<I: Into<DimIndex>> IntoIndex for I {
    fn into_index(self) -> impl Iterator<Item = DimIndex> + ExactSizeIterator + DoubleEndedIterator {
        std::iter::once(self.into())
    }
}

impl<I: Into<DimIndex>, const N: usize> IntoIndex for [I; N] {
    fn into_index(self) -> impl Iterator<Item = DimIndex> + ExactSizeIterator + DoubleEndedIterator {
        self.into_iter().map(|e| e.into())
    }
}

impl<I: Into<DimIndex> + Clone> IntoIndex for &[I] {
    fn into_index(self) -> impl Iterator<Item = DimIndex> + ExactSizeIterator + DoubleEndedIterator {
        self.into_iter().map(|e| e.clone().into())
    }
}

impl<I: Into<DimIndex>> IntoIndex for Vec<I> {
    fn into_index(self) -> impl Iterator<Item = DimIndex> + ExactSizeIterator + DoubleEndedIterator {
        self.into_iter().map(|e| e.into())
    }
}

impl<I0: Into<DimIndex>, I1: Into<DimIndex>> IntoIndex for (I0, I1) {
    fn into_index(self) -> impl Iterator<Item = DimIndex> + ExactSizeIterator + DoubleEndedIterator {
        [self.0.into(), self.1.into()].into_iter()
    }
}

impl<I0: Into<DimIndex>, I1: Into<DimIndex>, I2: Into<DimIndex>> IntoIndex for (I0, I1, I2) {
    fn into_index(self) -> impl Iterator<Item = DimIndex> + ExactSizeIterator + DoubleEndedIterator {
        [self.0.into(), self.1.into(), self.2.into()].into_iter()
    }
}

impl<I0: Into<DimIndex>, I1: Into<DimIndex>, I2: Into<DimIndex>, I3: Into<DimIndex>> IntoIndex for (I0, I1, I2, I3) {
    fn into_index(self) -> impl Iterator<Item = DimIndex> + ExactSizeIterator + DoubleEndedIterator {
        [self.0.into(), self.1.into(), self.2.into(), self.3.into()].into_iter()
    }
}

impl<I0: Into<DimIndex>, I1: Into<DimIndex>, I2: Into<DimIndex>, I3: Into<DimIndex>, I4: Into<DimIndex>> IntoIndex
    for (I0, I1, I2, I3, I4)
{
    fn into_index(self) -> impl Iterator<Item = DimIndex> + ExactSizeIterator + DoubleEndedIterator {
        [self.0.into(), self.1.into(), self.2.into(), self.3.into(), self.4.into()].into_iter()
    }
}

impl<I0: Into<DimIndex>, I1: Into<DimIndex>, I2: Into<DimIndex>, I3: Into<DimIndex>, I4: Into<DimIndex>, I5: Into<DimIndex>>
    IntoIndex for (I0, I1, I2, I3, I4, I5)
{
    fn into_index(self) -> impl Iterator<Item = DimIndex> + ExactSizeIterator + DoubleEndedIterator {
        [
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
        ]
        .into_iter()
    }
}

impl<
    I0: Into<DimIndex>,
    I1: Into<DimIndex>,
    I2: Into<DimIndex>,
    I3: Into<DimIndex>,
    I4: Into<DimIndex>,
    I5: Into<DimIndex>,
    I6: Into<DimIndex>,
> IntoIndex for (I0, I1, I2, I3, I4, I5, I6)
{
    fn into_index(self) -> impl Iterator<Item = DimIndex> + ExactSizeIterator + DoubleEndedIterator {
        [
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
            self.6.into(),
        ]
        .into_iter()
    }
}

impl<
    I0: Into<DimIndex>,
    I1: Into<DimIndex>,
    I2: Into<DimIndex>,
    I3: Into<DimIndex>,
    I4: Into<DimIndex>,
    I5: Into<DimIndex>,
    I6: Into<DimIndex>,
    I7: Into<DimIndex>,
> IntoIndex for (I0, I1, I2, I3, I4, I5, I6, I7)
{
    fn into_index(self) -> impl Iterator<Item = DimIndex> + ExactSizeIterator + DoubleEndedIterator {
        [
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
            self.6.into(),
            self.7.into(),
        ]
        .into_iter()
    }
}
