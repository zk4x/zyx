extern crate alloc;
use crate::{dtype::DType, node::Node, shape::Shape, tensor::Id, scalar::Scalar};
use alloc::{collections::{BTreeMap, BTreeSet}, vec::Vec};
use crate::error::ZyxError;

/// Backend for [tensors](Tensor).
/// Tensor requires that all backends implement this trait and only this trait.
pub trait Backend: Copy {
    /// Create uniform tensor, 0..1 if real, MIN..MAX if integer
    fn _uniform(self, shape: Shape, dtype: DType) -> Id;
    /// Get shape if tensor x
    fn shape(self, x: Id) -> Shape;
    /// Get dtype of tensor x
    fn dtype(self, x: Id) -> DType;
    /// Calculate derivatives of x w.r.t. sources.
    /// Returns map source id -> gradient id
    fn backward(self, x: Id, sources: &BTreeSet<Id>) -> Result<BTreeMap<Id, Id>, ZyxError>;
    /// Returns iterator over data stored in backend
    fn load<T: Scalar>(self, id: Id) -> Result<Vec<T>, ZyxError>;
    /// Create new tensor from given operation
    fn push(self, node: Node) -> Result<Id, ZyxError>;
    /// Set some tensor as leaf, i. e. it no longer "requires grad"
    fn set_leaf(self, x: Id);
    /// Decrease reference count of tensor
    fn release(self, x: Id) -> Result<(), ZyxError>;
    /// Increase reference count of tensor
    fn retain(self, x: Id);
}
