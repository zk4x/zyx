extern crate alloc;
use crate::error::ZyxError;
use crate::tensor::{tensor, IntoTensor, Tensor};
use crate::{dtype::DType, node::Node, scalar::Scalar, shape::Shape, tensor::Id};
use alloc::{
    string::String,
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use core::ops::Range;
use crate::utils::SizedIterator;

/// Backend for [tensors](Tensor).
/// Tensor requires that all backends implement this trait and only this trait.
pub trait Backend: Copy {
    /// Create graph of operations between tensors in dot format for visualization
    #[must_use]
    fn plot_graph<'a, B: Backend + 'a>(self, tensors: impl IntoIterator<Item = &'a Tensor<B>>) -> String;

    /// Create new tensor
    #[must_use]
    fn tensor(self, data: impl IntoTensor<Self>) -> Result<Tensor<Self>, ZyxError> {
        Ok(data.into_tensor(self))
    }

    /// Create new tensor using values from standard normal distribution
    #[must_use]
    fn randn(self, shape: impl Into<Shape>, dtype: DType) -> Result<Tensor<Self>, ZyxError>;

    /// Create new tensor using values from uniform distribution
    #[must_use]
    fn uniform(self, shape: impl Into<Shape>, range: Range<impl Scalar>) -> Result<Tensor<Self>, ZyxError>;

    /// Create new tensor by repeating single value
    #[must_use]
    fn full(self, shape: impl Into<Shape>, value: impl Scalar) -> Result<Tensor<Self>, ZyxError> {
        Ok(tensor(self.store([value])?, self).expand(shape))
    }

    /// Create new tensor by repeating zeroes
    #[must_use]
    fn zeros(self, shape: impl Into<Shape>, dtype: DType) -> Result<Tensor<Self>, ZyxError> {
        match dtype {
            DType::F32 => self.full(shape, 0f32),
            DType::F64 => self.full(shape, 0f64),
            DType::I32 => self.full(shape, 0),
        }
    }

    /// Create new tensor by repeating ones
    #[must_use]
    fn ones(self, shape: impl Into<Shape>, dtype: DType) -> Result<Tensor<Self>, ZyxError> {
        match dtype {
            DType::F32 => self.full(shape, 1f32),
            DType::F64 => self.full(shape, 1f64),
            DType::I32 => self.full(shape, 1),
        }
    }

    /// Create eye tensor
    #[must_use]
    fn eye(self, n: usize, dtype: DType) -> Result<Tensor<Self>, ZyxError> {
        Ok(tensor(
            match dtype {
                DType::F32 => self.store((0..n).flat_map(move | i | (0..n).map(move | j | if j == i { 1f32 } else { 0. })).make_sized(n*n))?,
                DType::F64 => self.store((0..n).flat_map(move | i | (0..n).map(move | j | if j == i { 1f64 } else { 0. })).make_sized(n*n))?,
                DType::I32 => self.store((0..n).flat_map(move | i | (0..n).map(move | j | if j == i { 1i32 } else { 0 })).make_sized(n*n))?,
            },
            self,
        ).reshape([n, n]))
    }

    /// Get shape if tensor x
    #[must_use]
    fn shape(self, x: Id) -> Shape;
    /// Get dtype of tensor x
    #[must_use]
    fn dtype(self, x: Id) -> DType;
    /// Calculate derivatives of x w.r.t. sources.
    /// Returns map source id -> gradient id
    fn backward(self, x: Id, sources: &BTreeSet<Id>) -> Result<BTreeMap<Id, Id>, ZyxError>;
    /// Returns iterator over data stored in backend
    fn load<T: Scalar>(self, id: Id) -> Result<Vec<T>, ZyxError>;
    /// Store iterator into backend as tensor
    fn store<T: Scalar, IT>(self, iter: IT) -> Result<Id, ZyxError>
    where
        IT: IntoIterator<Item=T>,
        IT::IntoIter: ExactSizeIterator;
    /// Create new tensor from given operation
    fn push(self, node: Node) -> Result<Id, ZyxError>;
    /// Decrease reference count of tensor
    fn release(self, x: Id) -> Result<(), ZyxError>;
    /// Increase reference count of tensor
    fn retain(self, x: Id);
}
