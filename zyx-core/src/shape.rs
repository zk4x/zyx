extern crate alloc;
use crate::axes::Axes;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ops::Range;

fn to_usize_idx(index: i64, rank: usize) -> usize {
    if index >= 0 && index <= rank as i64 {
        index as usize
    } else {
        (index + rank as i64) as usize % rank
    }
}

/// Shape of tensor
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Shape(Box<[usize]>);

impl core::fmt::Display for Shape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{:?}", self.0))
    }
}

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
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &usize> + ExactSizeIterator {
        self.into_iter()
    }

    /// Iter mut
    #[must_use]
    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut usize> + ExactSizeIterator {
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
    /// # Panics
    /// Panics if axes is incorrect.
    #[must_use]
    pub fn permute(&self, axes: &Axes) -> Self {
        //std::println!("self: {self}, axes: {axes:?}");
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

    #[cfg(feature = "std")]
    pub(crate) fn safetensors(&self) -> alloc::string::String {
        let mut res = alloc::format!("{:?}", self.0);
        res.retain(|c| !c.is_whitespace());
        res
    }

    #[cfg(feature = "std")]
    pub(crate) fn from_safetensors(shape: &str) -> Result<Shape, crate::error::ZyxError> {
        Ok(Shape(
            shape
                .split(',')
                .map(|d| {
                    d.parse::<usize>().map_err(|err| {
                        crate::error::ZyxError::ParseError(alloc::format!(
                            "Cannot parse safetensors shape: {err}"
                        ))
                    })
                })
                .collect::<Result<Box<[usize]>, crate::error::ZyxError>>()?,
        ))
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

    /// Pad self with padding
    #[must_use]
    pub fn pad(mut self, padding: &[(i64, i64)]) -> Shape {
        for (i, d) in self.iter_mut().rev().enumerate() {
            if let Some((left, right)) = padding.get(i) {
                *d = (*d as i64 + left + right) as usize;
            } else {
                break;
            }
        }
        self
    }

    /// Get self as vector i64
    #[must_use]
    pub fn vi64(&self) -> Vec<i64> {
        self.0.iter().map(|x| *x as i64).collect()
    }
}

impl core::ops::Index<i32> for Shape {
    type Output = usize;
    fn index(&self, index: i32) -> &Self::Output {
        self.0.get(to_usize_idx(index as i64, self.rank())).unwrap()
    }
}

impl core::ops::Index<i64> for Shape {
    type Output = usize;
    fn index(&self, index: i64) -> &Self::Output {
        self.0.get(to_usize_idx(index, self.rank())).unwrap()
    }
}

impl core::ops::Index<usize> for Shape {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        self.0.get(index).unwrap()
    }
}

impl core::ops::Index<Range<i64>> for Shape {
    type Output = [usize];
    fn index(&self, index: Range<i64>) -> &Self::Output {
        let rank = self.rank();
        self.0
            .get(to_usize_idx(index.start, rank)..to_usize_idx(index.end, rank))
            .unwrap()
    }
}

impl From<Shape> for Vec<usize> {
    fn from(val: Shape) -> Self {
        val.0.into()
    }
}

impl From<&Shape> for Shape {
    fn from(sh: &Shape) -> Self {
        sh.clone()
    }
}

impl From<Box<[usize]>> for Shape {
    fn from(value: Box<[usize]>) -> Self {
        Shape(value)
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
    type Item = &'a usize;
    type IntoIter = <&'a [usize] as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a mut Shape {
    type Item = &'a mut usize;
    type IntoIter = <&'a mut [usize] as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl PartialEq<[usize]> for Shape {
    fn eq(&self, other: &[usize]) -> bool {
        self.rank() == other.len() && self.iter().zip(other).all(|(x, y)| x == y)
    }
}

impl<const RANK: usize> PartialEq<[usize; RANK]> for Shape {
    fn eq(&self, other: &[usize; RANK]) -> bool {
        self.rank() == RANK && self.iter().zip(other).all(|(x, y)| x == y)
    }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        &self.0
    }
}
