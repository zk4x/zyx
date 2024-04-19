extern crate alloc;
use crate::error::ZyxError;
use crate::tensor::{IntoTensor, Tensor};
use crate::utils::SizedIterator;
use crate::{dtype::DType, node::Node, scalar::Scalar, shape::Shape, tensor::Id};
use alloc::{
    collections::{BTreeMap, BTreeSet},
    string::String,
    vec::Vec,
};
use core::ops::Range;
use half::f16;

/// Backend for [tensors](Tensor).
/// Tensor requires that all backends implement this trait and only this trait.
pub trait Backend: Copy {
    /// Create graph of operations between tensors in dot format for visualization
    #[must_use]
    fn plot_graph<'a, B: Backend + 'a>(
        self,
        tensors: impl IntoIterator<Item = &'a Tensor<B>>,
    ) -> String;

    /// Create new tensor
    #[must_use]
    fn tensor(self, data: impl IntoTensor<Self>) -> Result<Tensor<Self>, ZyxError> {
        Ok(data.into_tensor(self))
    }

    /// Create new tensor using values from uniform distribution
    #[must_use]
    fn uniform<T: Scalar>(
        self,
        shape: impl Into<Shape>,
        range: Range<T>,
    ) -> Result<Tensor<Self>, ZyxError>;

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
    fn store<T: Scalar, IT>(self, iter: IT) -> Result<Tensor<Self>, ZyxError>
    where
        IT: IntoIterator<Item = T>,
        IT::IntoIter: ExactSizeIterator;
    /// Create new tensor from given operation
    fn push(self, node: Node) -> Result<Tensor<Self>, ZyxError>;
    /// Decrease reference count of tensor
    fn release(self, x: Id) -> Result<(), ZyxError>;
    /// Increase reference count of tensor
    fn retain(self, x: Id);
    /// Realize tensors
    fn realize(self, tensors: BTreeSet<Id>) -> Result<(), ZyxError>;

    /// Create new tensor using values from standard normal distribution
    fn randn(self, shape: impl Into<Shape>, dtype: DType) -> Result<Tensor<Self>, ZyxError> {
        // Box Muller transform
        let src = self.uniform(2, 0f32..1f32)?;
        Ok(((src.get(0) * (2f32*core::f32::consts::PI)).cos() * (self.ones(1, DType::F32)? - src.get(1)).ln() * -2f32).sqrt().cast(dtype).reshape(shape))
    }

    /// Create new tensor by repeating single value
    fn full(self, shape: impl Into<Shape>, value: impl Scalar) -> Result<Tensor<Self>, ZyxError> {
        Ok(self.store([value])?.expand(shape))
    }

    /// Create new tensor by repeating zeroes
    fn zeros(self, shape: impl Into<Shape>, dtype: DType) -> Result<Tensor<Self>, ZyxError> {
        match dtype {
            DType::F16 => self.full(shape, f16::ZERO),
            DType::F32 => self.full(shape, 0f32),
            DType::F64 => self.full(shape, 0f64),
            DType::I32 => self.full(shape, 0),
        }
    }

    /// Create new tensor by repeating ones
    fn ones(self, shape: impl Into<Shape>, dtype: DType) -> Result<Tensor<Self>, ZyxError> {
        match dtype {
            DType::F16 => self.full(shape, f16::ONE),
            DType::F32 => self.full(shape, 1f32),
            DType::F64 => self.full(shape, 1f64),
            DType::I32 => self.full(shape, 1),
        }
    }

    /// Create eye tensor
    fn eye(self, n: usize, dtype: DType) -> Result<Tensor<Self>, ZyxError> {
        Ok(
            match dtype {
                DType::F16 => self.store(
                    (0..n)
                        .flat_map(move |i| (0..n).map(move |j| if j == i { f16::ONE } else { f16::ZERO }))
                        .make_sized(n * n),
                )?,
                DType::F32 => self.store(
                    (0..n)
                        .flat_map(move |i| (0..n).map(move |j| if j == i { 1f32 } else { 0. }))
                        .make_sized(n * n),
                )?,
                DType::F64 => self.store(
                    (0..n)
                        .flat_map(move |i| (0..n).map(move |j| if j == i { 1f64 } else { 0. }))
                        .make_sized(n * n),
                )?,
                DType::I32 => self.store(
                    (0..n)
                        .flat_map(move |i| (0..n).map(move |j| if j == i { 1i32 } else { 0 }))
                        .make_sized(n * n),
                )?,
            }.reshape([n, n]))
    }
}
