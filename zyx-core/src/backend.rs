extern crate alloc;
use crate::{dtype::DType, node::Node, shape::Shape, tensor::Id, scalar::Scalar};
use alloc::{collections::{BTreeMap, BTreeSet}, vec::Vec};

pub trait Backend: Copy {
    /// Get shape if tensor x
    fn shape(self, x: Id) -> Shape;
    /// Get dtype of tensor x
    fn dtype(self, x: Id) -> DType;
    /// Calculate derivatives of x w.r.t. sources.
    /// Returns map source id -> gradient id
    fn backward(self, x: Id, sources: &BTreeSet<Id>) -> BTreeMap<Id, Id>;
    /// Returns iterator over data stored in backend
    fn load<T: Scalar>(self, id: Id) -> Vec<T>;
    /// Create new tensor from given operation
    fn push(self, node: Node) -> Id;
    /// Set some tensor as leaf, i. e. it no longer "requires grad"
    fn set_leaf(self, x: Id);
    /// Decrease reference count of tensor
    fn release(self, x: Id);
    /// Increase reference count of tensor
    fn retain(self, x: Id);
}
