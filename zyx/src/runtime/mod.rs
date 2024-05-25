use crate::device::Device;
use crate::dtype::DType;
use crate::runtime::compiler::cuda::CUDA;
use crate::runtime::compiler::opencl::OpenCL;
use crate::runtime::compiler::wgpu::WGPU;
use crate::runtime::compiler::{CompiledBackend, CompilerError};
use crate::runtime::interpreter::cpu::CPU;
use crate::runtime::interpreter::{InterpretedBackend, InterpreterError};
use crate::scalar::Scalar;
use crate::tensor::Tensor;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::cell::OnceCell;
use node::Node;
use rand::rngs::SmallRng;

mod compiler;
mod interpreter;
mod node;
mod view;

type TensorId = u32;

#[derive(Debug)]
pub(crate) enum RuntimeError {
    CompilerError(CompilerError),
    InterpreterError(InterpreterError),
}

impl From<CompilerError> for RuntimeError {
    fn from(value: CompilerError) -> Self {
        Self::CompilerError(value)
    }
}

impl From<InterpreterError> for RuntimeError {
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
    devices: Vec<Device>,
}

impl Graph {
    fn shape(&self, x: TensorId) -> Vec<usize> {
        let mut x = x;
        let mut i = 0;
        while i < 10000 {
        let node = & self.nodes[x as usize];
        match node {
            Node::Const { .. } => return alloc::vec![1],
            Node::Leaf { len, .. } => return alloc::vec![ * len],
            Node::Reshape { shape_id: shape, .. }
            | Node::Pad { shape_id: shape, .. }
            | Node::Permute { shape_id: shape, .. }
            | Node::Sum { shape_id: shape, .. }
            | Node::Max { shape_id: shape, .. }
            | Node::Expand { shape_id: shape, .. } => return self._shape( * shape).into(),
            _ => x = node.parameters().next().unwrap(),
            }
        }
        panic!("Shape of {x} could not be found. This is internal bug.")
    }

    fn dtype(&self, x: TensorId) -> DType {
        todo!()
    }

    fn device(&self, x: TensorId) -> Device {
        self.devices[<u32 as TryInto<usize>>::try_into(x).unwrap()]
    }

    fn _shape(&self, shape_id: u32) -> &[usize] {
        let id = <u32 as TryInto<usize>>::try_into(shape_id).unwrap();
        let len = self.shapes[id];
        return &self.shapes[id+1..id+1+len]
    }

    fn _axes(&self, axes_id: u32) -> &[usize] {
        let id = <u32 as TryInto<usize>>::try_into(axes_id).unwrap();
        let len = self.axes[id];
        return &self.axes[id+1..id+1+len]
    }

    fn _padding(&self, padding_id: u32) -> &[isize] {
        let id = <u32 as TryInto<usize>>::try_into(padding_id).unwrap();
        let len = <isize as TryInto<usize>>::try_into(self.paddings[id]).unwrap();
        return &self.paddings[id+1..id+1+len]
    }

    fn is_empty(&self, x: TensorId) -> bool {
        self.rcs[<u32 as TryInto<usize>>::try_into(x).unwrap()] == 0
    }

    fn push(&mut self, node: Node) -> Tensor {
        for nid in node.parameters() {
            self.rcs[<u32 as TryInto<usize>>::try_into(nid).unwrap()] += 1;
        }
        if let Some(i) = (0..self.rcs.len())
            .into_iter()
            .skip_while(|i| !self.is_empty(<usize as TryInto<u32>>::try_into(*i).unwrap()))
            .next()
        {
            let tensor = Tensor::from_raw(i);
            self.rcs[i] = 1;
            self.nodes[i] = node;
            return tensor
        } else {
            let tensor = Tensor::from_raw(self.rcs.len());
            self.rcs.push(1);
            self.nodes.push(node);
            return tensor
        }
    }
}

pub(crate) struct Runtime {
    graph: Graph,
    opencl: Option<CompiledBackend<OpenCL>>,
    cuda: Option<CompiledBackend<CUDA>>,
    wgpu: Option<CompiledBackend<WGPU>>,
    cpu: Option<InterpretedBackend<CPU>>,
    pub(crate) default_device: Device,
    pub(crate) default_device_set_by_user: bool,
    pub(crate) rng: OnceCell<SmallRng>,
}

impl Runtime {
    pub(crate) const fn new() -> Self {
        use rand::SeedableRng;
        Runtime {
            graph: Graph {
                rcs: Vec::new(),
                nodes: Vec::new(),
                shapes: Vec::new(),
                axes: Vec::new(),
                paddings: Vec::new(),
                devices: Vec::new(),
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
        self.graph.rcs[x as usize] += 1;
    }

    pub(crate) fn release(&mut self, x: TensorId) {
        let mut params = Vec::with_capacity(10);
        params.push(x);
        while let Some(x) = params.pop() {
            let i = <u32 as TryInto<usize>>::try_into(x).unwrap();
            self.graph.rcs[i] -= 1;
            if self.graph.rcs[i] == 0 {
                params.extend(self.graph.nodes[i].parameters());
                match self.graph.devices[i] {
                    Device::CUDA => self.cuda.as_mut().unwrap().remove(x),
                    Device::OpenCL => self.opencl.as_mut().unwrap().remove(x),
                    Device::WGPU => self.wgpu.as_mut().unwrap().remove(x),
                    Device::CPU => todo!(), //self.cpu.as_mut().unwrap().remove(x),
                }
            }
        }
    }

    pub(crate) fn shape(&self, x: TensorId) -> Vec<usize> {
        return self.graph.shape(x)
    }

    pub(crate) fn dtype(&self, x: TensorId) -> DType {
        return self.graph.dtype(x)
    }

    pub(crate) fn device(&self, x: TensorId) -> Device {
        return self.graph.device(x)
    }

    pub(crate) fn relu(&mut self, x: TensorId) -> Tensor {
        return self.push(Node::ReLU { x })
    }

    pub(crate) fn exp(&mut self, x: TensorId) -> Tensor {
        return self.push(Node::Exp { x })
    }

    pub(crate) fn ln(&mut self, x: TensorId) -> Tensor {
        return self.push(Node::Ln { x })
    }

    pub(crate) fn sin(&mut self, x: TensorId) -> Tensor {
        return self.push(Node::Sin { x })
    }

    pub(crate) fn cos(&mut self, x: TensorId) -> Tensor {
        return self.push(Node::Cos { x })
    }

    pub(crate) fn sqrt(&mut self, x: TensorId) -> Tensor {
        return self.push(Node::Sqrt { x })
    }

    pub(crate) fn tanh(&mut self, x: TensorId) -> Tensor {
        return self.push(Node::Cos { x })
    }

    pub(crate) fn reshape(&mut self, x: TensorId, shape: &[usize]) -> Tensor {
        let shape_id = self.graph.shapes.len().try_into().unwrap();
        self.graph.shapes.push(shape.len());
        self.graph.shapes.extend(shape);
        return self.push(Node::Reshape { x, shape_id })
    }
}

impl Runtime {
    fn is_empty(&self, x: TensorId) -> bool {
        self.graph.rcs[x as usize] == 0
    }

    pub(crate) fn store<T: Scalar>(
        &mut self,
        data: &[T],
        dev: Device,
    ) -> Result<Tensor, RuntimeError> {
        let node = Node::Leaf { len: data.len(), dtype: T::dtype() };
        match dev {
            Device::OpenCL => {
                if self.opencl.is_none() {
                    self.opencl = Some(CompiledBackend::initialize()?);
                }
                let dev = self.opencl.as_mut().unwrap();
                dev.store(data)?;
            }
            Device::CUDA => {
                if self.cuda.is_none() {
                    self.cuda = Some(CompiledBackend::initialize()?);
                }
                let dev = self.cuda.as_mut().unwrap();
                dev.store(data)?;
            }
            Device::WGPU => {
                if self.wgpu.is_none() {
                    self.wgpu = Some(CompiledBackend::initialize()?);
                }
                let dev = self.wgpu.as_mut().unwrap();
                dev.store(data)?;
            }
            Device::CPU => {
                if self.cpu.is_none() {
                    self.cpu = Some(InterpretedBackend::initialize()?);
                }
                let dev = self.cpu.as_mut().unwrap();
                dev.store(data)?;
            }
        }
        let tensor = self.graph.push(node);
        return Ok(tensor)
    }

    pub(crate) fn load<T: Scalar>(&mut self, x: TensorId) -> Result<Vec<T>, RuntimeError> {
        if !self.is_realized(x) {
            self.realize(BTreeSet::from_iter([x]))?;
        }
        match self.device(x) {
            Device::CUDA => self.cuda.load(x),
            Device::OpenCL => self.opencl.load(x),
            Device::WGPU => self.wgpu.load(x),
            Device::CPU => self.cpu.load(x),
        }
    }

    fn push(&mut self, node: Node) -> Tensor {
        //std::println!("Assigned id: {id}, rcs {:?}", self.rcs);
        /*self.unrealized_nodes_count += 1;
        // This regulates caching, 256 tensors per batch seems like a good default
        if self.unrealized_nodes_count > 10000 {
            self.realize([id].into_iter().collect::<BTreeSet<Id>>())?;
            //std::println!("Num tensors: {}", self.nodes.len());
        }*/
        return self.graph.push(node)
    }

    fn is_realized(&self, x: TensorId) -> bool {
        return match self.device(x) {
            Device::CUDA => if let Some(cuda) = self.cuda.as_ref() {
                cuda.is_realized(x)
            } else {
                false
            }
            Device::OpenCL => {}
            Device::WGPU => {}
            Device::CPU => {}
        }
    }

    fn realize(&mut self, tensors: BTreeSet<TensorId>) -> Result<(), RuntimeError> {
        // topo search
        let mut params: Vec<TensorId> = tensors.iter().map(|tensor_id| *tensor_id).collect();
        let device = self.graph.devices[params[0] as usize];
        let mut visited = BTreeSet::new();
        while let Some(param) = params.pop() {
            if visited.insert(param) {
                params.extend(self.graph.nodes[param as usize].parameters());
            }
        }
        match device {
            Device::OpenCL => {
                let graph = visited
                    .into_iter()
                    .map(|id| (id, self.graph.nodes[id as usize]))
                    .collect();
                self.opencl.as_mut().unwrap().compile_graph(&self.graph, graph, tensors)?;
            }
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
        return Ok(())
    }
}
