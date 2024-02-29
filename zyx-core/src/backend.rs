extern crate alloc;
use crate::error::ZyxError;
use crate::tensor::{tensor, IntoTensor, Tensor};
use crate::{dtype::DType, node::Node, scalar::Scalar, shape::Shape, tensor::Id};
use alloc::boxed::Box;
use alloc::{
    string::String,
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use core::ops::Range;

/// Backend for [tensors](Tensor).
/// Tensor requires that all backends implement this trait and only this trait.
pub trait Backend: Copy {
    /// Create graph of operations between tensors in dot format for visualization
    #[must_use]
    fn plot_graph<'a, B: Backend + 'a>(self, tensors: impl IntoIterator<Item = &'a Tensor<B>>) -> String;

    /// Create new tensor
    #[must_use]
    fn tensor(self, data: impl IntoTensor<Self>) -> Tensor<Self> {
        data.into_tensor(self)
    }

    /// Create new tensor using values from standard normal distribution
    #[must_use]
    fn randn(self, shape: impl Into<Shape>, dtype: DType) -> Tensor<Self>;

    /// Create new tensor using values from uniform distribution
    #[must_use]
    fn uniform(self, shape: impl Into<Shape>, range: Range<impl Scalar>) -> Tensor<Self>;

    /// Create new tensor by repeating single value
    #[must_use]
    fn full(self, shape: impl Into<Shape>, value: impl Scalar) -> Tensor<Self> {
        fn get_dtype<T: Scalar>(_: T) -> DType {
            T::dtype()
        }
        tensor(
            match get_dtype(value.clone()) {
                DType::F32 => self.push(Node::IterF32(
                    Box::new([value.into_f32()].into_iter()),
                    1.into(),
                )),
                DType::F64 => self.push(Node::IterF64(
                    Box::new([value.into_f64()].into_iter()),
                    1.into(),
                )),
                DType::I32 => self.push(Node::IterI32(
                    Box::new([value.into_i32()].into_iter()),
                    1.into(),
                )),
            }
            .unwrap(), // Can't fail, as this does not call backend
            self,
        ).expand(shape)
    }

    /// Create new tensor by repeating zeroes
    #[must_use]
    fn zeros(self, shape: impl Into<Shape>, dtype: DType) -> Tensor<Self> {
        match dtype {
            DType::F32 => self.full(shape, 0f32),
            DType::F64 => self.full(shape, 0f64),
            DType::I32 => self.full(shape, 0),
        }
    }

    /// Create new tensor by repeating ones
    #[must_use]
    fn ones(self, shape: impl Into<Shape>, dtype: DType) -> Tensor<Self> {
        match dtype {
            DType::F32 => self.full(shape, 1f32),
            DType::F64 => self.full(shape, 1f64),
            DType::I32 => self.full(shape, 1),
        }
    }

    /// Create eye tensor
    #[must_use]
    fn eye(self, n: usize, dtype: DType) -> Tensor<Self> {
        tensor(
            match dtype {
                DType::F32 => self.push(Node::IterF32(
                    Box::new(
                        (0..n).flat_map(move |i| (0..n).map(move |j| if j == i { 1. } else { 0. })),
                    ),
                    [n, n].into(),
                )),
                DType::F64 => self.push(Node::IterF64(
                    Box::new(
                        (0..n).flat_map(move |i| (0..n).map(move |j| if j == i { 1. } else { 0. })),
                    ),
                    [n, n].into(),
                )),
                DType::I32 => self.push(Node::IterI32(
                    Box::new(
                        (0..n).flat_map(move |i| (0..n).map(move |j| if j == i { 1 } else { 0 })),
                    ),
                    [n, n].into(),
                )),
            }
            .unwrap(), // Can't fail, as this does not call backend
            self,
        )
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
    /// Create new tensor from given operation
    fn push(self, node: Node) -> Result<Id, ZyxError>;
    /// Decrease reference count of tensor
    fn release(self, x: Id) -> Result<(), ZyxError>;
    /// Increase reference count of tensor
    fn retain(self, x: Id);
    // Returns tensor detached from gradient tape
    //fn detach(self, x: Id) -> Tensor<Self>;
}
