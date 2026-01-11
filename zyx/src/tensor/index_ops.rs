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
    ///   - A `Vec<Range<isize>>` for dynamic indexing
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
    pub fn slice<I>(&self, index: I) -> Result<Tensor, ZyxError>
    where
        I: IntoIndex,
    {
        let shape = self.shape(); // original tensor shape
        let rank = shape.len();

        let mut squeeze_axes: Vec<Axis> = Vec::new();
        let index = index.into_index();
        let padding_len = index.len();

        if rank < padding_len {
            return Err(ZyxError::shape_error(
                format!("Slice with {padding_len} indices, but tensor has rank {rank}").into(),
            ));
        }

        let padding = std::iter::repeat_n((0, 0), rank - padding_len);

        let padding = padding.chain(
            index
                .zip(shape.into_iter())
                .enumerate()
                .map(|(axis, (dim_index, dim_size))| match dim_index {
                    DimIndex::Range { start, end } => {
                        let s = if start < 0 { start + dim_size as isize } else { start };
                        let e = if end > dim_size as isize {
                            dim_size as isize
                        } else if end < 0 {
                            end + dim_size as isize
                        } else {
                            end
                        };
                        (-s, -(dim_size as isize - e))
                    }
                    DimIndex::Index(i) => {
                        squeeze_axes.push(axis as i32);
                        let i = if i < 0 { i + dim_size as isize } else { i };
                        (-i, -(dim_size as isize - i - 1))
                    }
                    DimIndex::RangeFull => (0, 0),
                    DimIndex::RangeFrom { start } => {
                        let s = if start < 0 { start + dim_size as isize } else { start };
                        (-s, 0)
                    }
                    DimIndex::RangeTo { end } => {
                        let e = if end > dim_size as isize {
                            dim_size as isize
                        } else if end < 0 {
                            end + dim_size as isize
                        } else {
                            end
                        };
                        (0, -(dim_size as isize - e))
                    }
                })
                .rev(),
        );

        let mut result = self.pad_zeros(padding)?;
        result = result.squeeze(squeeze_axes);

        Ok(result)
    }

    /// Same as [Tensor::slice], but instead of indexing from first dimensions, it indexes from last dimensions.
    #[allow(clippy::missing_panics_doc)]
    pub fn rslice(&self, index: impl IntoIndex) -> Result<Tensor, ZyxError> {
        let shape = self.shape(); // original tensor shape
        let rank = shape.len();

        let mut squeeze_axes: Vec<Axis> = Vec::new();
        let index = index.into_index();
        let padding_len = index.len();

        let padding =
            index.zip(shape.into_iter().rev()).enumerate().map(|(axis, (dim_index, dim_size))| match dim_index {
                DimIndex::Range { start, end } => {
                    let s = if start < 0 { start + dim_size as isize } else { start };
                    let e = if end > dim_size as isize {
                        dim_size as isize
                    } else if end < 0 {
                        end + dim_size as isize
                    } else {
                        end
                    };
                    (-s, -(dim_size as isize - e))
                }
                DimIndex::Index(i) => {
                    squeeze_axes.push(axis as i32);
                    let i = if i < 0 { i + dim_size as isize } else { i };
                    (-i, -(dim_size as isize - i - 1))
                }
                DimIndex::RangeFull => (0, 0),
                DimIndex::RangeFrom { start } => {
                    let s = if start < 0 { start + dim_size as isize } else { start };
                    (-s, 0)
                }
                DimIndex::RangeTo { end } => {
                    let e = if end > dim_size as isize {
                        dim_size as isize
                    } else if end < 0 {
                        end + dim_size as isize
                    } else {
                        end
                    };
                    (0, -(dim_size as isize - e))
                }
            });

        let padding = padding.chain(std::iter::repeat_n((0, 0), rank - padding_len));

        let mut result = self.pad_zeros(padding)?;
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
            .pad_zeros([(0, isize::try_from(n).unwrap())])
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
    Index(isize),
    /// Range
    Range { start: isize, end: isize },
    /// Range from
    RangeFrom { start: isize },
    /// Range to
    RangeTo { end: isize },
    /// Range full
    RangeFull,
}

/// Into index
pub trait IntoIndex {
    fn into_index(self) -> impl Iterator<Item = DimIndex> + ExactSizeIterator + DoubleEndedIterator;
}

impl Into<DimIndex> for i32 {
    fn into(self) -> DimIndex {
        DimIndex::Index(self as isize)
    }
}

impl Into<DimIndex> for isize {
    fn into(self) -> DimIndex {
        DimIndex::Index(self)
    }
}

impl Into<DimIndex> for usize {
    fn into(self) -> DimIndex {
        DimIndex::Index(self as isize)
    }
}

impl Into<DimIndex> for Range<i32> {
    fn into(self) -> DimIndex {
        DimIndex::Range { start: self.start as isize, end: self.end as isize }
    }
}

impl Into<DimIndex> for Range<isize> {
    fn into(self) -> DimIndex {
        DimIndex::Range { start: self.start, end: self.end }
    }
}

impl Into<DimIndex> for Range<usize> {
    fn into(self) -> DimIndex {
        DimIndex::Range { start: self.start as isize, end: self.end as isize }
    }
}

impl Into<DimIndex> for RangeInclusive<i32> {
    fn into(self) -> DimIndex {
        DimIndex::Range { start: *self.start() as isize, end: *self.end() as isize + 1 }
    }
}

impl Into<DimIndex> for RangeInclusive<isize> {
    fn into(self) -> DimIndex {
        DimIndex::Range { start: *self.start(), end: self.end() + 1 }
    }
}

impl Into<DimIndex> for RangeInclusive<usize> {
    fn into(self) -> DimIndex {
        DimIndex::Range { start: *self.start() as isize, end: *self.end() as isize + 1 }
    }
}

impl Into<DimIndex> for RangeFrom<i32> {
    fn into(self) -> DimIndex {
        DimIndex::RangeFrom { start: self.start as isize }
    }
}

impl Into<DimIndex> for RangeFrom<isize> {
    fn into(self) -> DimIndex {
        DimIndex::RangeFrom { start: self.start }
    }
}

impl Into<DimIndex> for RangeFrom<usize> {
    fn into(self) -> DimIndex {
        DimIndex::RangeFrom { start: self.start as isize }
    }
}

impl Into<DimIndex> for RangeTo<i32> {
    fn into(self) -> DimIndex {
        DimIndex::RangeTo { end: self.end as isize }
    }
}

impl Into<DimIndex> for RangeTo<isize> {
    fn into(self) -> DimIndex {
        DimIndex::RangeTo { end: self.end }
    }
}

impl Into<DimIndex> for RangeTo<usize> {
    fn into(self) -> DimIndex {
        DimIndex::RangeTo { end: self.end as isize }
    }
}

impl Into<DimIndex> for RangeFull {
    fn into(self) -> DimIndex {
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
        [
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
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
> IntoIndex for (I0, I1, I2, I3, I4, I5)
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
