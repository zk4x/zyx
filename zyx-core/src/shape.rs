extern crate alloc;
use crate::axes::Axes;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ops::Range;

fn to_usize_idx(index: i64, rank: usize) -> usize {
    (index + rank as i64) as usize % rank
}

/// Shape of tensor
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Shape(Box<[usize]>);

impl Shape {
    /// Get shape's rank
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.0.len()
    }

    /// Get number of elements in tensor with this shape
    /// (a product of it's dimensions).
    #[must_use]
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Iter
    #[must_use]
    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.into_iter()
    }

    /// Get shape's strides
    #[must_use]
    pub fn strides(&self) -> Shape {
        let mut a = 1;
        Shape(
            self.0
                .iter()
                .rev()
                .map(|d| {
                    let t = a;
                    a *= d;
                    t
                })
                .collect::<Box<[usize]>>()
                .iter()
                .copied()
                .rev()
                .collect(),
        )
    }

    /// Permute shape's dimensions with axes
    #[must_use]
    pub fn permute(&self, axes: &Axes) -> Self {
        Self(axes.into_iter().map(|axis| self.0[*axis]).collect())
    }

    /// Get axes along which self was expanded to shape
    #[must_use]
    pub fn expand_axes(&self, shape: &Shape) -> Axes {
        let mut vec = self.0.to_vec();
        while vec.len() < shape.rank() {
            vec.insert(0, 1);
        }
        Axes(
            vec.into_iter()
                .zip(shape)
                .enumerate()
                .filter_map(|(a, (d, e))| if d == *e { None } else { Some(a) })
                .collect(),
        )
    }

    pub(crate) fn expand_strides(&self, shape: &Shape, mut old_strides: Shape) -> Shape {
        let mut vec = self.0.to_vec();
        while vec.len() < shape.rank() {
            vec.insert(0, 1);
            old_strides.0 = [0]
                .into_iter()
                .chain(old_strides.0.iter().copied())
                .collect();
        }
        let old_shape: Shape = vec.into();
        Shape(
            old_shape
                .into_iter()
                .zip(shape)
                .zip(&old_strides)
                .map(|((od, nd), st)| if od == nd { *st } else { 0 })
                .collect(),
        )
    }

    /// Reduce self along axes
    #[must_use]
    pub fn reduce(self, axes: &Axes) -> Shape {
        let mut shape = self;
        for a in axes.iter() {
            shape.0[*a] = 1;
        }
        shape
    }
}

impl core::ops::Index<i32> for Shape {
    type Output = usize;
    fn index(&self, index: i32) -> &Self::Output {
        self.0
            .get(to_usize_idx(index as i64, self.rank()))
            .unwrap()
    }
}

impl core::ops::Index<i64> for Shape {
    type Output = usize;
    fn index(&self, index: i64) -> &Self::Output {
        self.0
            .get(to_usize_idx(index, self.rank()))
            .unwrap()
    }
}

impl core::ops::Index<usize> for Shape {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        self.0
            .get(index)
            .unwrap()
    }
}

impl core::ops::Index<Range<i64>> for Shape {
    type Output = [usize];
    fn index(&self, index: Range<i64>) -> &Self::Output {
        let rank = self.rank();
        self.0.get(to_usize_idx(index.start, rank)..to_usize_idx(index.end, rank)).unwrap()
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        Shape(value.iter().copied().collect())
    }
}

impl From<&[usize]> for Shape {
    fn from(value: &[usize]) -> Self {
        Shape(value.iter().copied().collect())
    }
}

impl From<usize> for Shape {
    fn from(value: usize) -> Self {
        Shape(Box::new([value]))
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(value: [usize; N]) -> Self {
        Shape(value.into_iter().collect())
    }
}

impl<'a> IntoIterator for &'a Shape {
    type IntoIter = <&'a [usize] as IntoIterator>::IntoIter;
    type Item = &'a usize;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
