use node::Node;
use crate::device::Device;
use crate::runtime::compiler::cuda::CUDA;
use crate::runtime::compiler::opencl::OpenCL;
use crate::runtime::compiler::wgpu::WGPU;
use crate::runtime::interpreter::cpu::CPU;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::tensor::Tensor;
use crate::runtime::compiler::{CompiledBackend, CompilerError};
use crate::runtime::interpreter::{InterpretedBackend, InterpreterError};
use alloc::vec::Vec;

mod node;
mod compiler;
mod interpreter;

type TensorId = u32;

enum RuntimeError {
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

pub(crate) struct Runtime {
    rcs: Vec<u32>,
    nodes: Vec<Node>,
    shapes: Vec<usize>,
    axes: Vec<usize>,
    paddings: Vec<usize>,
    dtypes: Vec<DType>,
    devices: Vec<Device>,
    opencl: Option<CompiledBackend<OpenCL>>,
    cuda: Option<CompiledBackend<CUDA>>,
    wgpu: Option<CompiledBackend<WGPU>>,
    cpu: Option<InterpretedBackend<CPU>>,
    pub(crate) default_device: Device,
    pub(crate) default_device_set_by_user: bool,
}

impl Runtime {
    pub(crate) const fn new() -> Self {
        Runtime {
            rcs: Vec::new(),
            nodes: Vec::new(),
            shapes: Vec::new(),
            axes: Vec::new(),
            paddings: Vec::new(),
            dtypes: Vec::new(),
            devices: Vec::new(),
            opencl: None,
            cuda: None,
            wgpu: None,
            cpu: None,
            default_device: Device::CPU,
            default_device_set_by_user: false,
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
            return
        }
        if self.initialize_device(Device::CUDA) {
            self.default_device = Device::CUDA;
            return
        }
        if self.initialize_device(Device::OpenCL) {
            self.default_device = Device::OpenCL;
            return
        }
        if self.initialize_device(Device::WGPU) {
            self.default_device = Device::WGPU;
            return
        }
        if self.initialize_device(Device::CPU) {
            self.default_device = Device::CPU;
            return
        }
    }

    pub(crate) fn initialize_device(&mut self, device: Device) -> bool {
        match device {
            Device::CUDA => if let Ok(cuda) = CompiledBackend::initialize() {
                self.cuda = Some(cuda);
                true
            } else {
                false
            }
            Device::OpenCL => if let Ok(opencl) = CompiledBackend::initialize() {
                self.opencl = Some(opencl);
                true
            } else {
                false
            }
            Device::WGPU => if let Ok(wgpu) = CompiledBackend::initialize() {
                self.wgpu = Some(wgpu);
                true
            } else {
                false
            }
            Device::CPU => if let Ok(cpu) = InterpretedBackend::initialize() {
                self.cpu = Some(cpu);
                true
            } else {
                false
            }
        }
    }

    pub(crate) fn retain(&mut self, x: TensorId) {
        self.rcs[x as usize] += 1;
    }

    pub(crate) fn release(&mut self, x: TensorId) {
        let mut params = Vec::with_capacity(10);
        params.push(x);
        while let Some(x) = params.pop() {
            self.rcs[x as usize] -= 1;
            if self.rcs[x as usize] == 0 {
                params.extend(self.nodes[x as usize].parameters());
                match self.devices[x as usize] {
                    Device::CUDA => self.cuda.as_mut().unwrap().remove(x),
                    Device::OpenCL => self.opencl.as_mut().unwrap().remove(x),
                    Device::WGPU => self.wgpu.as_mut().unwrap().remove(x),
                    Device::CPU => todo!(), //self.cpu.as_mut().unwrap().remove(x),
                }
            }
        }
    }

    pub(crate) fn shape(&self, x: TensorId) -> Vec<usize> {
        todo!()
    }

    pub(crate) fn dtype(&self, x: TensorId) -> DType {
        todo!()
    }

    pub(crate) fn device(&self, x: TensorId) -> Device {
        self.devices[x as usize]
    }

    pub(crate) fn relu(&mut self, x: TensorId) -> Tensor {
        self.push(Node::ReLU(x))
    }

    pub(crate) fn exp(&mut self, x: TensorId) -> Tensor {
        self.push(Node::Exp(x))
    }

    pub(crate) fn ln(&mut self, x: TensorId) -> Tensor {
        self.push(Node::Ln(x))
    }

    pub(crate) fn sin(&mut self, x: TensorId) -> Tensor {
        self.push(Node::Sin(x))
    }

    pub(crate) fn cos(&mut self, x: TensorId) -> Tensor {
        self.push(Node::Cos(x))
    }

    pub(crate) fn sqrt(&mut self, x: TensorId) -> Tensor {
        self.push(Node::Sqrt(x))
    }

    pub(crate) fn tanh(&mut self, x: TensorId) -> Tensor {
        self.push(Node::Cos(x))
    }
}

impl Runtime {
    fn is_empty(&self, x: TensorId) -> bool {
        self.rcs[x as usize] == 0
    }

    fn store<T: Scalar>(&mut self, data: &[T], dev: Device) -> Result<Tensor, RuntimeError> {
        let node = Node::Leaf(data.len());
        let tensor = if let Some(i) = (0..self.rcs.len()).into_iter().skip_while(|i| !self.is_empty(*i as u32)).next() {
            let tensor = Tensor::from_raw(i);
            self.rcs[i] = 1;
            self.nodes[i] = node;
            self.dtypes[i] = T::dtype();
            tensor
        } else {
            let tensor = Tensor::from_raw(self.rcs.len());
            self.rcs.push(1);
            self.nodes.push(node);
            self.dtypes.push(T::dtype());
            tensor
        };
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
        Ok(tensor)
    }

    fn push(&mut self, node: Node) -> Tensor {
        todo!()
    }
}
