use crate::device::Device;
use crate::dtype::DType;
use crate::runtime::compiler::cuda::CUDA;
use crate::runtime::compiler::opencl::OpenCLCompiler;
use crate::runtime::compiler::wgpu::WGPU;
use crate::runtime::compiler::{CompiledBackend, CompilerError};
use crate::runtime::interpreter::cpu::CPU;
use crate::runtime::interpreter::{InterpretedBackend, InterpreterError};
use crate::scalar::Scalar;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::cell::OnceCell;
use core::ops::Index;
use node::Node;
use rand::rngs::SmallRng;
use crate::IntoShape;

mod compiler;
mod interpreter;
mod node;
mod view;

type TensorId = u32;

#[derive(Debug)]
pub enum ZyxError {
    CompilerError(CompilerError),
    InterpreterError(InterpreterError),
}

impl From<CompilerError> for ZyxError {
    fn from(value: CompilerError) -> Self {
        Self::CompilerError(value)
    }
}

impl From<InterpreterError> for ZyxError {
    fn from(value: InterpreterError) -> Self {
        Self::InterpreterError(value)
    }
}

struct Graph {
    rcs: Vec<u32>,
    nodes: Vec<Node>,
    // First value in each shape is rank, other values are shapes
    shapes: Vec<usize>,
    axes: Vec<usize>,
    paddings: Vec<isize>,
}

impl Graph {
    fn push_shape(&mut self, shape: impl IntoShape) -> u32 {
        let shape_id = self.shapes.len().try_into().unwrap();
        self.shapes.push(shape.rank());
        self.shapes.extend(shape.into_shape());
        return shape_id
    }

    fn shape(&self, x: TensorId) -> &[usize] {
        let mut x = x;
        let mut i = 0;
        while i < 10000 {
            let node = &self.nodes[<u32 as TryInto<usize>>::try_into(x).unwrap()];
            match node {
                Node::Const { .. } => return &[1],
                Node::Leaf { shape_id, .. }
                | Node::Reshape { shape_id, .. }
                | Node::Pad { shape_id, .. }
                | Node::Permute { shape_id, .. }
                | Node::Sum { shape_id, .. }
                | Node::Max { shape_id, .. }
                | Node::Expand { shape_id, ..
                } => return self._shape(*shape_id).into(),
                _ => x = node.parameters().next().unwrap(),
            }
        }
        panic!("Shape of {x} could not be found. This is internal bug.")
    }

    fn dtype(&self, x: TensorId) -> DType {
        // TODO
        DType::F32
    }

    fn device(&self, x: TensorId) -> Device {
        // TODO
        Device::OpenCL
    }

    fn _shape(&self, shape_id: u32) -> &[usize] {
        let id = <u32 as TryInto<usize>>::try_into(shape_id).unwrap();
        let len = self.shapes[id];
        return &self.shapes[id + 1..id + 1 + len];
    }

    fn _axes(&self, axes_id: u32) -> &[usize] {
        let id = <u32 as TryInto<usize>>::try_into(axes_id).unwrap();
        let len = self.axes[id];
        return &self.axes[id + 1..id + 1 + len];
    }

    fn _padding(&self, padding_id: u32) -> &[isize] {
        let id = <u32 as TryInto<usize>>::try_into(padding_id).unwrap();
        let len = <isize as TryInto<usize>>::try_into(self.paddings[id]).unwrap();
        return &self.paddings[id + 1..id + 1 + len];
    }

    fn is_empty(&self, x: TensorId) -> bool {
        self.rcs[<u32 as TryInto<usize>>::try_into(x).unwrap()] == 0
    }

    fn push(&mut self, node: Node) -> TensorId {
        for nid in node.parameters() {
            self.rcs[<u32 as TryInto<usize>>::try_into(nid).unwrap()] += 1;
        }
        let mut i = 0;
        let n = self.rcs.len();
        while i < n && self.rcs[i] != 0 {
            i += 1;
        }
        if i != self.rcs.len() {
            self.rcs[i] = 1;
            self.nodes[i] = node;
        } else {
            self.rcs.push(1);
            self.nodes.push(node);
        }
        return i.try_into().unwrap();
    }
}

struct Subgraph<'a> {
    nodes: BTreeMap<TensorId, Node>,
    shapes: &'a [usize],
    axes: &'a [usize],
    paddings: &'a [isize],
}

impl Subgraph<'_> {
    fn dtype(&self, tensor_id: TensorId) -> DType {
        // TODO
        DType::F32
    }

    fn shape(&self, tensor_id: TensorId) -> &[usize] {
        let mut x = tensor_id;
        let mut i = 0;
        while i < 10000 {
            let node = &self.nodes[&x];
            match node {
                Node::Const { .. } => return &[1],
                Node::Leaf { shape_id, .. }
                | Node::Reshape { shape_id, .. }
                | Node::Pad { shape_id, .. }
                | Node::Permute { shape_id, .. }
                | Node::Sum { shape_id, .. }
                | Node::Max { shape_id, .. }
                | Node::Expand { shape_id, ..
                } => return self._shape(*shape_id).into(),
                _ => x = node.parameters().next().unwrap(),
            }
        }
        panic!("Shape of {x} could not be found. This is internal bug.")
    }

    fn _shape(&self, shape_id: u32) -> &[usize] {
        let id = <u32 as TryInto<usize>>::try_into(shape_id).unwrap();
        let len = self.shapes[id];
        return &self.shapes[id + 1..id + 1 + len];
    }

    fn _axes(&self, axes_id: u32) -> &[usize] {
        let id = <u32 as TryInto<usize>>::try_into(axes_id).unwrap();
        let len = self.axes[id];
        return &self.axes[id + 1..id + 1 + len];
    }

    fn _padding(&self, padding_id: u32) -> &[isize] {
        let id = <u32 as TryInto<usize>>::try_into(padding_id).unwrap();
        let len = <isize as TryInto<usize>>::try_into(self.paddings[id]).unwrap();
        return &self.paddings[id + 1..id + 1 + len];
    }

    /// Swap movement and unary op
    /// first and second tensors must have rc == 1!
    fn swap_nodes(&mut self, first: TensorId, second: TensorId) {
        let temp;
        match self.nodes.get_mut(&first).unwrap() {
            Node::Reshape { x, .. }
            | Node::Expand { x, .. }
            | Node::Pad { x, .. }
            | Node::Permute { x, .. } => {
                temp = *x;
                *x = first;
            }
            _ => panic!("First op must be movement"),
        }
        match self.nodes.get_mut(&second).unwrap() {
            Node::Cast { x, .. }
            | Node::Neg { x, .. }
            | Node::Inv { x, .. }
            | Node::ReLU { x, .. }
            | Node::Exp { x, .. }
            | Node::Ln { x, .. }
            | Node::Sin { x, .. }
            | Node::Cos { x, .. }
            | Node::Sqrt { x, .. } => {
                *x = temp;
            }
            _ => panic!("Second op must be unary"),
        }
        // swap the two nodes
        let first_value = self.nodes.remove(&first).unwrap();
        let second_value = self.nodes.remove(&second).unwrap();
        self.nodes.insert(first, second_value);
        self.nodes.insert(second, first_value);
    }
}

impl Index<TensorId> for Subgraph<'_> {
    type Output = Node;
    fn index(&self, index: TensorId) -> &Self::Output {
        self.nodes.get(&index).unwrap()
    }
}

pub(crate) struct Runtime {
    graph: Graph,
    opencl: Option<CompiledBackend<OpenCLCompiler>>,
    cuda: Option<CompiledBackend<CUDA>>,
    wgpu: Option<CompiledBackend<WGPU>>,
    cpu: Option<InterpretedBackend<CPU>>,
    pub(crate) default_device: Device,
    pub(crate) default_device_set_by_user: bool,
    pub(crate) rng: OnceCell<SmallRng>,
}

impl Runtime {
    pub(crate) const fn new() -> Self {
        Runtime {
            graph: Graph {
                rcs: Vec::new(),
                nodes: Vec::new(),
                shapes: Vec::new(),
                axes: Vec::new(),
                paddings: Vec::new(),
            },
            opencl: None,
            cuda: None,
            wgpu: None,
            cpu: None,
            default_device: Device::CPU,
            default_device_set_by_user: false,
            rng: OnceCell::new(),
        }
    }

    /// Tries to initialize all devices and set the first
    /// successfully initialized device as the default_device in this order:
    /// 1. CUDA
    /// 2. OpenCL
    /// 3. WGPU
    /// If they all fail to initialize, then default_device
    /// is set to CPU.
    pub(crate) fn set_default_device_best(&mut self) {
        if self.default_device_set_by_user {
            return;
        }
        if self.initialize_device(Device::CUDA) {
            self.default_device = Device::CUDA;
            return;
        }
        if self.initialize_device(Device::OpenCL) {
            self.default_device = Device::OpenCL;
            return;
        }
        if self.initialize_device(Device::WGPU) {
            self.default_device = Device::WGPU;
            return;
        }
        if self.initialize_device(Device::CPU) {
            self.default_device = Device::CPU;
            return;
        }
    }

    pub(crate) fn initialize_device(&mut self, device: Device) -> bool {
        match device {
            Device::CUDA => {
                if let Ok(cuda) = CompiledBackend::initialize() {
                    self.cuda = Some(cuda);
                    true
                } else {
                    false
                }
            }
            Device::OpenCL => {
                if let Ok(opencl) = CompiledBackend::initialize() {
                    self.opencl = Some(opencl);
                    true
                } else {
                    false
                }
            }
            Device::WGPU => {
                if let Ok(wgpu) = CompiledBackend::initialize() {
                    self.wgpu = Some(wgpu);
                    true
                } else {
                    false
                }
            }
            Device::CPU => {
                if let Ok(cpu) = InterpretedBackend::initialize() {
                    self.cpu = Some(cpu);
                    true
                } else {
                    false
                }
            }
        }
    }

    pub(crate) fn retain(&mut self, x: TensorId) {
        self.graph.rcs[<u32 as TryInto<usize>>::try_into(x).unwrap()] += 1;
    }

    pub(crate) fn release(&mut self, x: TensorId) -> Result<(), ZyxError> {
        let mut params = Vec::with_capacity(10);
        params.push(x);
        while let Some(x) = params.pop() {
            let i = <u32 as TryInto<usize>>::try_into(x).unwrap();
            self.graph.rcs[i] -= 1;
            if self.graph.rcs[i] == 0 {
                params.extend(self.graph.nodes[i].parameters());
                match self.device(i.try_into().unwrap()) {
                    Device::CUDA => self.cuda.as_mut().unwrap().remove(x)?,
                    Device::OpenCL => self.opencl.as_mut().unwrap().remove(x)?,
                    Device::WGPU => self.wgpu.as_mut().unwrap().remove(x)?,
                    Device::CPU => self.cpu.as_mut().unwrap().remove(x)?,
                }
            }
            // TODO release shapes, axes, paddings, otherwise this is a memory leak
        }
        return Ok(());
    }

    #[cfg(feature = "debug1")]
    pub(crate) fn debug_graph(&self) {
        use libc_print::std_name::println;
        for (id, node) in self.graph.nodes.iter().enumerate() {
            println!("{id:>5} x{:>3} -> {node:?}", self.graph.rcs[id]);
        }
    }

    pub(crate) fn shape(&self, x: TensorId) -> &[usize] {
        return self.graph.shape(x)
    }

    pub(crate) fn dtype(&self, x: TensorId) -> DType {
        return self.graph.dtype(x)
    }

    pub(crate) fn device(&self, x: TensorId) -> Device {
        return self.graph.device(x)
    }

    pub(crate) fn relu(&mut self, x: TensorId) -> TensorId {
        return self.push(Node::ReLU { x })
    }

    pub(crate) fn exp(&mut self, x: TensorId) -> TensorId {
        return self.push(Node::Exp { x });
    }

    pub(crate) fn ln(&mut self, x: TensorId) -> TensorId {
        return self.push(Node::Ln { x });
    }

    pub(crate) fn sin(&mut self, x: TensorId) -> TensorId {
        return self.push(Node::Sin { x });
    }

    pub(crate) fn cos(&mut self, x: TensorId) -> TensorId {
        return self.push(Node::Cos { x });
    }

    pub(crate) fn sqrt(&mut self, x: TensorId) -> TensorId {
        return self.push(Node::Sqrt { x });
    }

    pub(crate) fn tanh(&mut self, x: TensorId) -> TensorId {
        return self.push(Node::Cos { x });
    }

    pub(crate) fn reshape(&mut self, x: TensorId, shape: impl IntoShape) -> TensorId {
        let shape_id = self.graph.push_shape(shape);
        return self.push(Node::Reshape { x, shape_id });
    }
}

impl Runtime {
    fn is_empty(&self, x: TensorId) -> bool {
        self.graph.rcs[<u32 as TryInto<usize>>::try_into(x).unwrap()] == 0
    }

    pub(crate) fn store<T: Scalar>(
        &mut self,
        data: &[T],
        device: Device,
        shape: impl IntoShape,
    ) -> Result<TensorId, ZyxError> {
        let node = Node::Leaf {
            shape_id: self.graph.push_shape(shape),
            dtype: T::dtype(),
            device,
        };
        let tensor_id = self.graph.push(node);
        match device {
            Device::OpenCL => {
                if self.opencl.is_none() {
                    self.opencl = Some(CompiledBackend::initialize()?);
                }
                let dev = self.opencl.as_mut().unwrap();
                dev.store(tensor_id, data)?;
            }
            Device::CUDA => {
                if self.cuda.is_none() {
                    self.cuda = Some(CompiledBackend::initialize()?);
                }
                let dev = self.cuda.as_mut().unwrap();
                dev.store(tensor_id, data)?;
            }
            Device::WGPU => {
                if self.wgpu.is_none() {
                    self.wgpu = Some(CompiledBackend::initialize()?);
                }
                let dev = self.wgpu.as_mut().unwrap();
                dev.store(tensor_id, data)?;
            }
            Device::CPU => {
                if self.cpu.is_none() {
                    self.cpu = Some(InterpretedBackend::initialize()?);
                }
                let dev = self.cpu.as_mut().unwrap();
                dev.store(tensor_id, data)?;
            }
        }
        return Ok(tensor_id);
    }

    pub(crate) fn load<T: Scalar>(&mut self, x: TensorId) -> Result<Vec<T>, ZyxError> {
        return match self.device(x) {
            Device::CUDA => {
                if !self.cuda.as_ref().unwrap().is_realized(x) {
                    self.realize(BTreeSet::from_iter([x]))?;
                }
                let length = self.shape(x).iter().product();
                Ok(self.cuda.as_mut().unwrap().load(x, length)?)
            }
            Device::OpenCL => {
                if !self.opencl.as_ref().unwrap().is_realized(x) {
                    self.realize(BTreeSet::from_iter([x]))?;
                }
                let length = self.shape(x).iter().product();
                Ok(self.opencl.as_mut().unwrap().load(x, length)?)
            }
            Device::WGPU => {
                if !self.wgpu.as_ref().unwrap().is_realized(x) {
                    self.realize(BTreeSet::from_iter([x]))?;
                }
                let length = self.shape(x).iter().product();
                Ok(self.wgpu.as_mut().unwrap().load(x, length)?)
            }
            Device::CPU => {
                if !self.cpu.as_ref().unwrap().is_realized(x) {
                    self.realize(BTreeSet::from_iter([x]))?;
                }
                let length = self.shape(x).iter().product();
                Ok(self.cpu.as_mut().unwrap().load(x, length)?)
            }
        };
    }

    fn push(&mut self, node: Node) -> TensorId {
        //std::println!("Assigned id: {id}, rcs {:?}", self.rcs);
        /*self.unrealized_nodes_count += 1;
        // This regulates caching, 256 tensors per batch seems like a good default
        if self.unrealized_nodes_count > 10000 {
            self.realize([id].into_iter().collect::<BTreeSet<Id>>())?;
            //std::println!("Num tensors: {}", self.nodes.len());
        }*/
        return self.graph.push(node);
    }

    fn is_realized(&self, x: TensorId) -> bool {
        return match self.device(x) {
            Device::CUDA => {
                if let Some(cuda) = self.cuda.as_ref() {
                    cuda.is_realized(x)
                } else {
                    false
                }
            }
            Device::OpenCL => {
                if let Some(opencl) = self.opencl.as_ref() {
                    opencl.is_realized(x)
                } else {
                    false
                }
            }
            Device::WGPU => {
                if let Some(wgpu) = self.wgpu.as_ref() {
                    wgpu.is_realized(x)
                } else {
                    false
                }
            }
            Device::CPU => {
                if let Some(cpu) = self.cpu.as_ref() {
                    cpu.is_realized(x)
                } else {
                    false
                }
            }
        };
    }

    fn realize(&mut self, tensors: BTreeSet<TensorId>) -> Result<(), ZyxError> {
        // topo search
        let mut params: Vec<TensorId> = tensors.iter().map(|tensor_id| *tensor_id).collect();
        let device = self.graph.device(params[0]);
        let mut visited = BTreeSet::new();
        while let Some(param) = params.pop() {
            if visited.insert(param) {
                if self.is_realized(param) {
                    // TODO This param should be converted to Load in subgraph
                } else {
                    params.extend(self.graph.nodes[param as usize].parameters());
                }
            }
        }
        let subgraph = visited
            .into_iter()
            .map(|id| (id, self.graph.nodes[id as usize]))
            .collect();
        let graph = Subgraph {
            shapes: &self.graph.shapes,
            axes: &self.graph.axes,
            paddings: &self.graph.paddings,
            nodes: subgraph,
        };
        match device {
            Device::OpenCL => self.opencl.as_mut().unwrap().compile_graph(graph, tensors)?,
            Device::CUDA => {
                todo!()
            }
            Device::WGPU => {
                todo!()
            }
            Device::CPU => {
                todo!()
            }
        }
        return Ok(());
    }
}
