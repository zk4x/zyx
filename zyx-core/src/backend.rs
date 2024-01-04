extern crate alloc;
use crate::{dtype::DType, node::Node, shape::Shape, tensor::Id};
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use core::iter::Iterator;

pub enum BufferView {
    F32(Box<dyn Iterator<Item = f32>>),
    I32(Box<dyn Iterator<Item = i32>>),
}

pub trait Backend: Copy {
    /// Get shape if tensor x
    fn shape(self, x: Id) -> Shape;
    /// Get dtype of tensor x
    fn dtype(self, x: Id) -> DType;
    /// Calculate derivatives of x w.r.t. sources.
    /// Returns map source id -> gradient id
    fn backward(self, x: Id, sources: &BTreeSet<Id>) -> BTreeMap<Id, Id>;
    /// Returns iterator over data stored in backend
    fn load(self, id: Id) -> BufferView;
    /// Create new tensor from given operation
    fn push(self, node: Node) -> Id;
    /// Set some tensor as leaf, i. e. it no longer "requires grad"
    fn set_leaf(self, x: Id);
    /// Decrease reference count of tensor
    fn release(self, x: Id);
    /// Increase reference count of tensor
    fn retain(self, x: Id);
}
