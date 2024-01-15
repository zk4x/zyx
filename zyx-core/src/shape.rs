extern crate alloc;
use crate::axes::Axes;
use alloc::boxed::Box;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Shape(Box<[usize]>);

impl Shape {
    /// Get shape's rank
    #[must_use]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

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

    // Get axes along which self was expanded to shape
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
            .get((index + self.rank() as i32) as usize % self.rank())
            .unwrap()
    }
}

impl core::ops::Index<i64> for Shape {
    type Output = usize;
    fn index(&self, index: i64) -> &Self::Output {
        self.0
            .get((index + self.rank() as i64) as usize % self.rank())
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
