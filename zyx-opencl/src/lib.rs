use core::ffi::c_void;
use zyx_core::{axes::Axes, backend::{Backend}, Node, shape::Shape, tensor::Id};
use zyx_core::autodiff::Graph;
use zyx_core::backend::BufferView;
use zyx_core::dtype::DType;
use zyx_core::tensor::{Tensor, tensor};

pub struct OpenCL {
    mems: Vec<*mut c_void>,
    graph: zyx_core::autodiff::Graph,
}

impl OpenCL {
    pub fn new() -> OpenCL {
        OpenCL { mems: Vec::new(), graph: Graph::new() }
    }

    /// Returns f32 tensor. User is expected to cast if he needs different type.
    pub fn randn(&self, shape: impl Into<Shape>) -> Tensor<&Self> {
        let data = self.graph.leaf();
        tensor(self, data)
    }
}

impl Backend for &OpenCL {
    fn release(self, x: Id) {
        todo!()
    }

    fn retain(self, x: Id) {
        todo!()
    }

    fn shape(self, x: Id) -> Shape {
        todo!()
    }

    fn dtype(self, x: Id) -> DType {
        todo!()
    }

    fn backward(self, x: Id, sources: &[Id]) -> Vec<Option<Id>> {
        todo!()
    }

    fn store(self, buffer: BufferView<'_>) {
        todo!()
    }

    fn load(&self, x: Id) -> BufferView<'_> {
        todo!()
    }

    fn op(self, node: Node) -> Id {
        todo!()
    }
}
