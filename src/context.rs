//! # Context

extern crate alloc;
use alloc::{boxed::Box, rc::Rc, string::String, vec::Vec};
use crate::{
    node_id::NodeId,
    graph::Graph,
    shape::Shape,
    tensor::{IntoTensor, Tensor},
    OutOfMemoryError,
};
use core::cell::RefCell;

/// # Context
///
/// Context stores all data associated with tensors.
/// It stores actual values, that is device buffers,
/// directed acyclic graph of all operations executed on all tensors
/// and devices that are used to execute those operations.
#[derive(Debug)]
pub struct Context {
    graph: Rc<RefCell<Graph>>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Rc<RefCell<Graph>>> for Context {
    fn from(graph: Rc<RefCell<Graph>>) -> Self {
        Self { graph }
    }
}

impl Context {
    /// Returns Vec of strings of all nodes in this context.
    /// When you just want to print all nodes in realized graphs,
    /// consider using feature debug1.
    #[must_use]
    pub fn debug_nodes(&self) -> Vec<String> {
        self.graph.borrow_mut().debug_nodes()
    }

    /// Create new context. This context will use slow rust backend by default.
    #[must_use]
    pub fn new() -> Context {
        Self {
            graph: Rc::new(RefCell::new(Graph::default())),
        }
    }

    /// Create new context that uses `OpenCL` backend.
    /// # Errors
    /// Returns `ClError` if there was problem initializing `OpenCL`.
    #[cfg(feature = "opencl")]
    pub fn opencl() -> Result<Self, cl3::error_codes::ClError> {
        let graph = Rc::new(RefCell::new(Graph::default()));
        let device = crate::device::Device::opencl()?;
        graph.borrow_mut().devices.push(device);
        graph.borrow_mut().default_device = 1;
        Ok(Self { graph })
    }

    // TODO this can be perhaps interesting for multi platform execution,
    // but there needs to be way to convert tensors between devices and that is slow.
    /* /// Adds new device to context and makes it default.
    /// This function should be used with caution, because executing graph
    /// containing tensors that were already realized by different device will result in panic.
    pub fn on(&mut self, device: crate::device::Device) -> Self {
        let mut graph = self.graph.borrow_mut();
        graph.default_device = graph.devices.len();
        graph.devices.push(device.device);
        Self {
            rng: self.rng.clone(),
            graph: self.graph.clone(),
        }
    }*/

    #[must_use]
    pub(crate) fn from_graph(graph: Rc<RefCell<Graph>>) -> Context {
        Context { graph }
    }

    /// Create new tensor filled with value.
    #[must_use]
    pub fn full(&self, shape: impl Into<Shape>, value: f32) -> Tensor {
        let shape = shape.into();
        let mut graph = self.graph.borrow_mut();
        let temp = graph.push(crate::graph::Node::StoreF32(Box::new([value]), 1.into()));
        let data = graph.push(crate::graph::Node::Expand(temp, shape));
        graph.release(temp);
        Tensor {
            data,
            grad: None,
            graph: self.graph.clone(),
        }
    }

    /// Create new tensor filled with value.
    #[must_use]
    pub fn full_i32(&self, shape: impl Into<Shape>, value: i32) -> Tensor {
        let shape = shape.into();
        let mut graph = self.graph.borrow_mut();
        let temp = graph.push(crate::graph::Node::StoreI32(Box::new([value]), 1.into()));
        let data = graph.push(crate::graph::Node::Expand(temp, shape));
        graph.release(temp);
        Tensor {
            data,
            grad: None,
            graph: self.graph.clone(),
        }
    }

    /// Dot language string of graph
    #[must_use]
    pub fn dot_graph(&self) -> String {
        self.graph.borrow().show_graph()
    }

    /// Number of tensors stored in this context
    #[must_use]
    pub fn num_tensors(&self) -> usize {
        self.graph.borrow().num_nodes()
    }

    /// Create new ones tensor.
    #[must_use]
    pub fn ones(&self, shape: impl Into<Shape>) -> Tensor {
        self.full(shape, 1.)
    }

    /// Create new ones i32 tensor.
    #[must_use]
    pub fn ones_i32(&self, shape: impl Into<Shape>) -> Tensor {
        self.full_i32(shape, 1)
    }

    /// Create new tensor filled with values sampled from standard distribution.
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn randn(&self, shape: impl Into<Shape>) -> Tensor {
        Tensor {
            data: self.graph.borrow_mut().randn_f32(shape.into()),
            grad: None,
            graph: self.graph.clone(),
        }
    }

    /// Create new i32 tensor filled with values sampled from standard distribution.
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn randn_i32(&self, shape: impl Into<Shape>) -> Tensor {
        Tensor {
            data: self.graph.borrow_mut().randn_i32(shape.into()),
            grad: None,
            graph: self.graph.clone(),
        }
    }

    pub(crate) fn realize(&self, nodes: &[NodeId]) -> Result<(), OutOfMemoryError> {
        self.graph.borrow_mut().realize(nodes)
    }

    /// Create new tensor from data.
    #[must_use]
    pub fn tensor(&self, data: impl IntoTensor) -> Tensor {
        data.into_tensor(self)
    }

    /// Create new tensor from iterator
    #[must_use]
    pub fn tensor_from_iter_f32(
        &self,
        shape: impl Into<Shape>,
        iter: impl IntoIterator<Item = f32>,
    ) -> Tensor {
        let shape = shape.into();
        let n = shape.numel();
        Tensor {
            data: self
                .graph
                .borrow_mut()
                .tensor_from_iter_f32(shape, iter.into_iter().take(n)),
            grad: None,
            graph: self.graph.clone(),
        }
    }

    /// Create new i32 tensor from iterator
    #[must_use]
    pub fn tensor_from_iter_i32(
        &self,
        shape: impl Into<Shape>,
        iter: impl IntoIterator<Item = i32>,
    ) -> Tensor {
        let shape = shape.into();
        let n = shape.numel();
        Tensor {
            data: self
                .graph
                .borrow_mut()
                .tensor_from_iter_i32(shape, iter.into_iter().take(n)),
            grad: None,
            graph: self.graph.clone(),
        }
    }

    /// Create new i32 tensor filled with values sampled from uniform distribution.
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn uniform(&self, shape: impl Into<Shape>, range: core::ops::Range<f32>) -> Tensor {
        Tensor {
            data: self.graph.borrow_mut().uniform_f32(shape.into(), range),
            grad: None,
            graph: self.graph.clone(),
        }
    }

    /// Create new i32 tensor filled with values sampled from uniform distribution.
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn uniform_i32(&self, shape: impl Into<Shape>, range: core::ops::Range<i32>) -> Tensor {
        Tensor {
            data: self.graph.borrow_mut().uniform_i32(shape.into(), range),
            grad: None,
            graph: self.graph.clone(),
        }
    }

    /// Create new f32 tensor filled with zeros
    #[must_use]
    pub fn zeros(&self, shape: impl Into<Shape>) -> Tensor {
        self.full(shape, 0.)
    }

    /// Create new i32 tensor filled with zeros
    #[must_use]
    pub fn zeros_i32(&self, shape: impl Into<Shape>) -> Tensor {
        self.full_i32(shape, 0)
    }
}

impl IntoTensor for Tensor {
    fn into_tensor(self, _: &crate::context::Context) -> Tensor {
        self
    }
}

impl IntoTensor for &Tensor {
    fn into_tensor(self, _: &crate::context::Context) -> Tensor {
        self.clone()
    }
}

impl IntoTensor for f32 {
    fn into_tensor(self, ctx: &Context) -> Tensor {
        ctx.tensor_from_iter_f32(1, [self])
    }
}

impl<const L: usize> IntoTensor for [f32; L] {
    fn into_tensor(self, ctx: &Context) -> Tensor {
        ctx.tensor_from_iter_f32(L, self)
    }
}

impl<const L: usize, const M: usize> IntoTensor for [[f32; L]; M] {
    fn into_tensor(self, ctx: &Context) -> Tensor {
        ctx.tensor_from_iter_f32((M, L), self.into_iter().flatten())
    }
}

impl<const L: usize, const M: usize, const N: usize> IntoTensor for [[[f32; L]; M]; N] {
    fn into_tensor(self, ctx: &Context) -> Tensor {
        ctx.tensor_from_iter_f32((N, M, L), self.into_iter().flatten().flatten())
    }
}

impl IntoTensor for i32 {
    fn into_tensor(self, ctx: &Context) -> Tensor {
        ctx.tensor_from_iter_i32(1, [self])
    }
}

impl<const L: usize> IntoTensor for [i32; L] {
    fn into_tensor(self, ctx: &Context) -> Tensor {
        ctx.tensor_from_iter_i32(L, self)
    }
}

impl<const L: usize, const M: usize> IntoTensor for [[i32; L]; M] {
    fn into_tensor(self, ctx: &Context) -> Tensor {
        ctx.tensor_from_iter_i32((M, L), self.into_iter().flatten())
    }
}

impl<const L: usize, const M: usize, const N: usize> IntoTensor for [[[i32; L]; M]; N] {
    fn into_tensor(self, ctx: &Context) -> Tensor {
        ctx.tensor_from_iter_i32((N, M, L), self.into_iter().flatten().flatten())
    }
}