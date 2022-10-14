//! Shape for multidimensional data structures
//! 
//! Stores dimension sizes for multidimensional data structures.
//! Shape is a trait that is implemented for some basic data types,
//! such as tuples and arrays.
//! 

use crate::ops::{Permute, IntoVec};
use std::ops::Range;

pub trait IntoShape {
    fn shape(self) -> Shape;
}

pub trait IntoDims {
    fn dims(self) -> Dims;
}

#[derive(Debug, Default, Clone, Eq, PartialOrd, Ord)]
pub struct Shape(pub(crate) Vec<usize>);

impl Shape {
    pub fn strides(&self) -> Shape {
        let mut product = 1;
        let mut res = vec![0; self.ndim()];
        for (i, dim) in self.clone().into_iter().enumerate().rev() {
            res[i] = product;
            product *= dim;
        }
        Shape(res)
    }

    pub fn argsort(&self) -> Dims {
        let mut indices: Vec<i32> = (0..self.ndim() as i32).collect();
        indices.sort_by_key(|&i| &self[i as usize]);
        Dims(indices)
    }

    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }
}

impl Permute for Shape {
    type Output = Shape;
    fn permute(self, dims: impl IntoDims) -> Self::Output {
        let mut res = self.0.clone();
        for (i, dim) in dims.dims().into_iter().enumerate() {
            res[i] = self[dim];
        }
        Shape(res)
    }
}

impl IntoIterator for Shape {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl std::ops::IndexMut<i32> for Shape {
    fn index_mut(&mut self, idx: i32) -> &mut Self::Output {
        let n = self.ndim();
        &mut self.0[(n as i32 + idx) as usize % n]
    }
}

impl std::ops::Index<i32> for Shape {
    type Output = usize;
    fn index(&self, idx: i32) -> &Self::Output {
        let n = self.ndim();
        &self[(n as i32 + idx) as usize % n]
    }
}

impl std::ops::IndexMut<usize> for Shape {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        let n = self.ndim();
        &mut self.0[(n + idx) % n]
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = usize;
    fn index(&self, idx: usize) -> &Self::Output {
        let n = self.ndim();
        &self.0[(n + idx) % n]
    }
}

impl std::ops::Index<Range<usize>> for Shape {
    type Output = [usize];
    fn index(&self, idx: Range<usize>) -> &Self::Output {
        let n = self.ndim();
        &self.0[(n + idx.start) % n..(n + idx.end) % n]
    }
}

impl<Other> std::cmp::PartialEq<Other> for Shape
where
    Other: IntoShape + Clone,
{
    fn eq(&self, other: &Other) -> bool {
        self.0 == other.clone().shape().0
    }
}

impl IntoVec<usize> for Shape {
    fn to_vec(&self) -> Vec<usize> {
        self.0.clone()
    }
}

/*#[cfg(feature = "ndarray")]
impl ndarray::IntoDimension for Shape {
    type Dim: Dimension;
}*/

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Dims(pub(crate) Vec<i32>);

impl Dims {
    pub fn strides(&self) -> Dims {
        let mut product = 1;
        let mut res = vec![0; self.ndim()];
        for (i, dim) in self.clone().into_iter().enumerate().rev() {
            res[i] = product;
            product *= dim;
        }
        Dims(res)
    }

    pub fn argsort(&self) -> Dims {
        let mut indices: Vec<i32> = (0..self.ndim() as i32).collect();
        indices.sort_by_key(|&i| &self[i]);
        Dims(indices)
    }

    pub fn numel(&self) -> usize {
        self.0.iter().product::<i32>() as usize
    }

    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }
}

impl IntoIterator for Dims {
    type Item = i32;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl std::ops::Index<i32> for Dims {
    type Output = i32;
    fn index(&self, idx: i32) -> &Self::Output {
        let n = self.ndim();
        &self.0[(n as i32 + idx) as usize % n]
    }
}

use duplicate::duplicate_item;

#[duplicate_item(
    dtype;
    [f32];
    [f64];
    [i8];
    [i16];
    [i32];
    [i64];
    [i128];
    [u8];
    [u16];
    [u32];
    [u64];
    [u128];
    [isize];
    [usize];
)]

impl IntoShape for dtype {
    fn shape(self) -> Shape {
        Shape(vec![1])
    }
}

impl IntoShape for () {
    fn shape(self) -> Shape {
        Shape(Vec::new())
    }
}

impl IntoShape for (usize, usize) {
    fn shape(self) -> Shape {
        Shape(vec![self.0, self.1])
    }
}

impl IntoShape for (usize, usize, usize) {
    fn shape(self) -> Shape {
        Shape(vec![self.0, self.1, self.2])
    }
}

impl IntoShape for (usize, usize, usize, usize) {
    fn shape(self) -> Shape {
        Shape(vec![self.0, self.1, self.2, self.3])
    }
}

impl IntoShape for (usize, usize, usize, usize, usize) {
    fn shape(self) -> Shape {
        Shape(vec![self.0, self.1, self.2, self.3, self.4])
    }
}

impl IntoShape for (usize, usize, usize, usize, usize, usize) {
    fn shape(self) -> Shape {
        Shape(vec![self.0, self.1, self.2, self.3, self.4, self.5])
    }
}

impl IntoShape for (usize, usize, usize, usize, usize, usize, usize) {
    fn shape(self) -> Shape {
        Shape(vec![self.0, self.1, self.2, self.3, self.4, self.5, self.6])
    }
}

impl IntoShape for (usize, usize, usize, usize, usize, usize, usize, usize) {
    fn shape(self) -> Shape {
        Shape(vec![self.0, self.1, self.2, self.3, self.4, self.5, self.6, self.7])
    }
}

impl IntoShape for (usize, usize, usize, usize, usize, usize, usize, usize, usize) {
    fn shape(self) -> Shape {
        Shape(vec![self.0, self.1, self.2, self.3, self.4, self.5, self.6, self.7, self.8])
    }
}

impl IntoShape for (usize, usize, usize, usize, usize, usize, usize, usize, usize, usize) {
    fn shape(self) -> Shape {
        Shape(vec![self.0, self.1, self.2, self.3, self.4, self.5, self.6, self.7, self.8, self.9])
    }
}

impl IntoShape for &[usize] {
    fn shape(self) -> Shape {
        Shape(self.into())
    }
}

impl IntoShape for Vec<usize> {
    fn shape(self) -> Shape {
        Shape(self)
    }
}

impl IntoShape for Shape {
    fn shape(self) -> Shape {
        self
    }
}

impl<const N: usize> IntoShape for [usize; N] {
    fn shape(self) -> Shape {
        Shape(self.into())
    }
}

#[duplicate_item(
    dtype;
    [f32];
    [f64];
    [i8];
    [i16];
    [i32];
    [i64];
    [i128];
    [u8];
    [u16];
    [u32];
    [u64];
    [u128];
    [isize];
    [usize];
)]

impl IntoDims for dtype {
    fn dims(self) -> Dims {
        Dims(vec![1])
    }
}

impl IntoDims for () {
    fn dims(self) -> Dims {
        Dims(Vec::new())
    }
}

impl IntoDims for (i32, i32) {
    fn dims(self) -> Dims {
        Dims(vec![self.0, self.1])
    }
}

impl IntoDims for &[i32] {
    fn dims(self) -> Dims {
        Dims(self.into())
    }
}

impl IntoDims for Vec<i32> {
    fn dims(self) -> Dims {
        Dims(self)
    }
}

impl IntoDims for Dims {
    fn dims(self) -> Dims {
        self
    }
}

impl<const N: usize> IntoDims for [i32; N] {
    fn dims(self) -> Dims {
        Dims(self.into())
    }
}
