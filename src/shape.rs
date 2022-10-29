//! [Shape] and [Dims] for multidimensional data structures
//! 
//! This module defines [Shape] and [Dims] structs. These store `Vec<usize>` and `Vec<i32>` respectively.
//! [Shape] stores the size of tensor's dimensions while [Dims] stores dimension's order,
//! which can also be negative (-1 is last dimension). [Dims] is used as input into functions as [Permute](crate::ops::Permute) or [Sum](crate::ops::Sum),
//! when we need to define along which dimensions we want to perform these operations.
//!
//! > This API is subject to change as we would like to move towards API based on const shape.
//! > That however depends on rust's support for user defined const generics.
//! > This would be particularly nice, becuase it would get rid of all runtime errors, since everything else is already enforced in type system of [buffer](crate::accel), [Variable](crate::tensor::Variable), [Tensor](crate::tensor::Tensor).
//!

use crate::ops::{Permute, IntoVec};
use std::ops::Range;

/// # IntoShape trait
/// 
/// Turn input into [Shape].
/// This trait is implemented for tuples, vec and arrays of usize.
/// 
/// # Example
/// 
/// ```
/// # use zyx::prelude::*;
/// let shape = (2, 4, 1).shape();
/// ```
/// It is used in tensor initialization>.
/// For example create buffer with random values with shape 2 by 3 by 4:
/// ```
/// let x = cpu::Buffer<f32>::randn((2, 3, 4));
/// ```
pub trait IntoShape {
    /// Turn input into Shape
    fn shape(self) -> Shape;
}

/// Turn datatype into [Dims]
/// 
/// This trait is implemented for tuples, vec and arrays of i32.
pub trait IntoDims {
    /// Turn input into Dims
    fn dims(self) -> Dims;
}

/// # Shape
/// 
/// Stores size of dimensions of multidimensional data structures.
#[derive(Debug, Default, Clone, Eq, PartialOrd, Ord)]
pub struct Shape(pub(crate) Vec<usize>);

impl Shape {
    /// Get [Shape] strides
    pub fn strides(&self) -> Shape {
        let mut product = 1;
        let mut res = vec![0; self.ndim()];
        self.clone().into_iter().enumerate().rev().for_each(
            |(i, dim)| {
                res[i] = product;
                product *= dim;
            });
        Shape(res)
    }

    /// Returns the indices that sort [Shape] in ascending order
    pub fn argsort(&self) -> Dims {
        let mut indices: Vec<i32> = (0..self.ndim() as i32).collect();
        indices.sort_by_key(|&i| &self[i as usize]);
        Dims(indices)
    }

    /// Get [Shape's](Shape) number of elements.
    /// This is just a product of all [Shape's](Shape) dimensions.
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Get number of [Shape's](Shape) dimensions.
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Check if the [Shape] is empty
    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }
}

impl Permute for Shape {
    type Output = Shape;
    fn permute(self, dims: impl IntoDims) -> Self::Output {
        Shape(dims.dims().into_iter().map(|dim| self[dim]).collect())
    }
}

/*impl<'a> IntoIterator for &'a Shape {
    type Item = usize;
    type IntoIter = std::slice::Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}*/

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

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut data = self.0.iter().map(|x| format!("{}, ", x)).collect::<String>();
        data.replace_range(data.len()-2..data.len(), "");
        f.write_fmt(format_args!("Shape({})", data))
    }
}

/*#[cfg(feature = "ndarray")]
impl ndarray::IntoDimension for Shape {
    type Dim: Dimension;
}*/

/// # Dims
/// 
/// Stores indices of dimensions of [Shape].
/// We pass this struct to operations like [sum](crate::ops::Sum), [max](crate::ops::Max) or [permute](crate::ops::Permute).
/// So we can also say this struct holds axes of [shapes](Shape) or tensors.
#[derive(Debug, Default, Clone, Eq, PartialOrd, Ord)]
pub struct Dims(pub(crate) Vec<i32>);

impl Dims {
    /// Get strides of [Dims].
    pub fn strides(&self) -> Dims {
        let mut product = 1;
        let mut res = vec![0; self.ndim()];
        self.clone().into_iter().enumerate().rev().for_each(|(i, dim)| {
            res[i] = product;
            product *= dim;
        });
        Dims(res)
    }

    /// Returns the indices that sort [Dims] in ascending order.
    pub fn argsort(&self) -> Dims {
        let mut indices: Vec<i32> = (0..self.ndim() as i32).collect();
        indices.sort_by_key(|&i| &self[i]);
        Dims(indices)
    }

    /// Get number of dimensions/axes stored in [Dims].
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Check if [Dims] are empty.
    /// Unlike Shape, this function only returns true if there are no values stored in [Dims].
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
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
)]

impl IntoShape for dtype {
    fn shape(self) -> Shape {
        Shape(vec![self as usize])
    }
}

impl IntoShape for () {
    fn shape(self) -> Shape {
        Shape(Vec::new())
    }
}

impl IntoShape for usize {
    fn shape(self) -> Shape {
        Shape(vec![self])
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
    [i8];
    [i16];
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
        Dims(vec![self as i32])
    }
}

impl IntoDims for i32 {
    fn dims(self) -> Dims {
        Dims(vec![self])
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

impl IntoDims for (i32, i32, i32) {
    fn dims(self) -> Dims {
        Dims(vec![self.0, self.1, self.2])
    }
}

impl IntoDims for (i32, i32, i32, i32) {
    fn dims(self) -> Dims {
        Dims(vec![self.0, self.1, self.2, self.3])
    }
}

impl IntoDims for (i32, i32, i32, i32, i32) {
    fn dims(self) -> Dims {
        Dims(vec![self.0, self.1, self.2, self.3, self.4])
    }
}

impl IntoDims for (i32, i32, i32, i32, i32, i32) {
    fn dims(self) -> Dims {
        Dims(vec![self.0, self.1, self.2, self.3, self.4, self.5])
    }
}

impl IntoDims for (i32, i32, i32, i32, i32, i32, i32) {
    fn dims(self) -> Dims {
        Dims(vec![self.0, self.1, self.2, self.3, self.4, self.5, self.6])
    }
}

impl IntoDims for (i32, i32, i32, i32, i32, i32, i32, i32) {
    fn dims(self) -> Dims {
        Dims(vec![self.0, self.1, self.2, self.3, self.4, self.5, self.6, self.7])
    }
}

impl IntoDims for (i32, i32, i32, i32, i32, i32, i32, i32, i32) {
    fn dims(self) -> Dims {
        Dims(vec![self.0, self.1, self.2, self.3, self.4, self.5, self.6, self.7, self.8])
    }
}
impl IntoDims for (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
    fn dims(self) -> Dims {
        Dims(vec![self.0, self.1, self.2, self.3, self.4, self.5, self.6, self.7, self.8, self.9])
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

impl<Other> std::cmp::PartialEq<Other> for Dims
where
    Other: IntoDims + Clone,
{
    fn eq(&self, other: &Other) -> bool {
        self.0 == other.clone().dims().0
    }
}

impl std::fmt::Display for Dims {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut data = self.0.iter().map(|x| format!("{}, ", x)).collect::<String>();
        data.replace_range(data.len()-2..data.len(), "");
        f.write_fmt(format_args!("Dims({})", data))
    }
}
