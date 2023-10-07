//! Shape
//!
//! See [Shape].

extern crate alloc;
use crate::axes::Axes;
use alloc::{boxed::Box, format, vec::Vec};

/// # Shape
///
/// Shape represents dimensions/axes of a tensor.
/// ```
/// # use zyx::context::Context;
/// # let mut ctx = Context::new();
/// let x = ctx.tensor([[2], [3]]);
/// assert_eq!(x.shape(), [2, 1]);
/// ```
/// Shapes can be created from tuples, arrays, slices or vectors of usize.
/// ```
/// # use zyx::shape::Shape;
/// let s0: Shape = (4, 1, 5, 6).into();
/// assert_eq!(s0, [4, 1, 5, 6]);
/// assert_eq!(s0, (4, 1, 5, 6));
/// ```
/// Shapes can be unpacked into arrays. It sometimes requires type hint.
/// ```
/// # use zyx::shape::Shape;
/// # let s0: Shape = (4, 1, 5, 6).into();
/// let [a, b, c, d]: [usize; 4] = s0.try_into().unwrap();
/// assert_eq!(c, 5);
/// ```
// TODO this should be no more than 8 bytes
#[derive(Debug, Default, Clone, Eq, PartialOrd, Ord, Hash)]
pub struct Shape(Box<[usize]>);

/// # Strides
///
/// Strides represent number of elements to get to the next axis.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Strides(pub(crate) Box<[usize]>);

impl Shape {
    /// Get shape's rank
    #[must_use]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Get number of elements stored in tensor with this shape.
    /// This is this shape's product.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Get shape's strides
    #[must_use]
    pub fn strides(&self) -> Strides {
        let mut a = 1;
        Strides(
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

    #[must_use]
    pub(crate) fn transpose_axes(&self) -> Axes {
        let n = self.rank();
        let mut axes: Box<[usize]> = (0..n).collect();
        axes.swap(n - 1, n - 2);
        Axes(axes)
    }

    /// Transpose shape
    #[must_use]
    pub fn transpose(&self) -> Shape {
        let mut x = self.clone();
        let n = x.0.len();
        x.0.swap(n - 1, n - 2);
        x
    }

    /// Reduce shape, setting dimensions in axes to 1
    #[must_use]
    pub fn reduce(mut self, axes: &Axes) -> Shape {
        for a in axes {
            self.0[*a] = 1;
        }
        self
    }

    // Get axes along which self was expanded to shape
    #[must_use]
    pub(crate) fn expand_axes(&self, shape: &Shape) -> Axes {
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

    #[cfg(feature = "opencl")]
    pub(crate) fn opencl_expand_strides(&self, shape: &Shape) -> Strides {
        let mut vec = self.0.to_vec();
        while vec.len() < shape.rank() {
            vec.insert(0, 1);
        }
        let old_shape: Shape = vec.into();
        Strides(
            old_shape
                .into_iter()
                .zip(shape)
                .zip(&shape.strides())
                .map(|((od, nd), st)| if od == nd { *st } else { 0 })
                .collect(),
        )
    }

    pub(crate) fn expand_strides(&self, shape: &Shape, mut old_strides: Strides) -> Strides {
        let mut vec = self.0.to_vec();
        while vec.len() < shape.rank() {
            vec.insert(0, 1);
            old_strides.0 = [0]
                .into_iter()
                .chain(old_strides.0.iter().copied())
                .collect();
        }
        let old_shape: Shape = vec.into();
        Strides(
            old_shape
                .into_iter()
                .zip(shape)
                .zip(&old_strides)
                .map(|((od, nd), st)| if od == nd { *st } else { 0 })
                .collect(),
        )
    }

    #[cfg(feature = "io")]
    pub(crate) fn safetensors(&self) -> alloc::string::String {
        let mut res = format!("{:?}", self.0);
        res.retain(|c| !c.is_whitespace());
        res
    }

    pub(crate) fn dot(&self, ysh: &Shape) -> Shape {
        let xsh = self;
        assert_eq!(
            xsh[-1],
            if ysh.rank() > 1 { ysh[-2] } else { 1usize },
            "Incorrect dot shapes {xsh} @ {ysh}",
        );
        for (i, (xa, ya)) in xsh.into_iter().zip(ysh).rev().enumerate() {
            if i > 1 {
                assert_eq!(xa, ya, "Incorrect dot shapes {xsh} @ {ysh}");
            }
        }
        let mut shape = alloc::vec![if xsh.rank() > 1 { xsh[-2] } else { 1 }, ysh[-1],];
        if xsh.rank() > 2 || ysh.rank() > 2 {
            if xsh.rank() > ysh.rank() {
                let mut i = xsh.rank() - 3;
                while xsh.rank() > shape.len() {
                    shape.insert(0, xsh[i]);
                    i -= 1;
                }
            } else {
                let mut i = ysh.rank() - 3;
                while ysh.rank() > shape.len() {
                    shape.insert(0, ysh[i]);
                    i -= 1;
                }
            }
        }
        shape.into()
    }
}

impl core::fmt::Display for Shape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let res = format!("{:?}", self.0).replace('[', "(").replace(']', ")");
        f.write_str(&res)
    }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        &self.0
    }
}

impl<const N: usize> TryInto<[usize; N]> for Shape {
    type Error = core::array::TryFromSliceError;
    fn try_into(self) -> Result<[usize; N], Self::Error> {
        self.0.as_ref().try_into()
    }
}

impl Strides {
    /// Permute strides
    #[must_use]
    pub fn permute(&self, axes: &Axes) -> Self {
        Strides(axes.into_iter().map(|axis| self.0[*axis]).collect())
    }
}

impl<'a> IntoIterator for &'a Strides {
    type IntoIter = <&'a [usize] as IntoIterator>::IntoIter;
    type Item = &'a usize;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a Shape {
    type IntoIter = <&'a [usize] as IntoIterator>::IntoIter;
    type Item = &'a usize;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl core::ops::Index<usize> for Shape {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        self.0.get(index).unwrap()
    }
}

impl core::ops::Index<i32> for Shape {
    type Output = usize;
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_possible_wrap)]
    #[allow(clippy::cast_sign_loss)]
    fn index(&self, index: i32) -> &Self::Output {
        self.0
            .get((index + self.rank() as i32) as usize % self.rank())
            .unwrap()
    }
}

impl core::ops::Index<usize> for Strides {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        self.0.get(index).unwrap()
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(value: [usize; N]) -> Self {
        Shape(value.into_iter().collect())
    }
}

impl From<&[usize]> for Shape {
    fn from(value: &[usize]) -> Self {
        Shape(value.iter().copied().collect())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        Shape(value.iter().copied().collect())
    }
}

impl From<Box<[usize]>> for Shape {
    fn from(value: Box<[usize]>) -> Self {
        Shape(value.iter().copied().collect())
    }
}

impl From<usize> for Shape {
    fn from(value: usize) -> Self {
        Shape(Box::new([value]))
    }
}

impl From<(usize, usize)> for Shape {
    fn from(value: (usize, usize)) -> Self {
        Shape(Box::new([value.0, value.1]))
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from(value: (usize, usize, usize)) -> Self {
        Shape(Box::new([value.0, value.1, value.2]))
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from(value: (usize, usize, usize, usize)) -> Self {
        Shape(Box::new([value.0, value.1, value.2, value.3]))
    }
}

impl From<(usize, usize, usize, usize, usize)> for Shape {
    fn from(value: (usize, usize, usize, usize, usize)) -> Self {
        Shape(Box::new([value.0, value.1, value.2, value.3, value.4]))
    }
}

impl From<(usize, usize, usize, usize, usize, usize)> for Shape {
    fn from(value: (usize, usize, usize, usize, usize, usize)) -> Self {
        Shape(Box::new([
            value.0, value.1, value.2, value.3, value.4, value.5,
        ]))
    }
}

impl From<(usize, usize, usize, usize, usize, usize, usize)> for Shape {
    fn from(value: (usize, usize, usize, usize, usize, usize, usize)) -> Self {
        Shape(Box::new([
            value.0, value.1, value.2, value.3, value.4, value.5, value.6,
        ]))
    }
}

impl From<(usize, usize, usize, usize, usize, usize, usize, usize)> for Shape {
    fn from(value: (usize, usize, usize, usize, usize, usize, usize, usize)) -> Self {
        Shape(Box::new([
            value.0, value.1, value.2, value.3, value.4, value.5, value.6, value.7,
        ]))
    }
}

impl<IS: Clone + Into<Shape>> PartialEq<IS> for Shape {
    fn eq(&self, other: &IS) -> bool {
        let other: Shape = other.clone().into();
        for (x, y) in self.0.iter().zip(other.0.iter()) {
            if *x != *y {
                return false;
            }
        }
        true
    }
}
