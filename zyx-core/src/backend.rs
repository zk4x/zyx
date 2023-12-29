extern crate alloc;
use crate::{axes::Axes, Node, shape::Shape, tensor::Id, Vec};
use core::iter::Iterator;
use crate::dtype::DType;

pub enum BufferView<'a> {
    F32(&'a dyn Iterator<Item = f32>),
    I32(&'a dyn Iterator<Item = i32>),
}

pub trait Backend: Clone {
    fn release(self, x: Id);
    fn retain(self, x: Id);
    fn shape(self, x: Id) -> Shape;
    fn dtype(self, x: Id) -> DType;
    fn backward(self, x: Id, sources: &[Id]) -> Vec<Option<Id>>;
    fn store(self, buffer: BufferView<'_>);
    fn load(&self, id: Id) -> BufferView<'_>;
    fn op(self, node: Node) -> Id;
}
