use crate::device::Device;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::shape::{IntoAxes, IntoShape};
use crate::tensor::TensorId;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use graph::Graph;
use interpreter::cpu::CPU;
use interpreter::{InterpretedBackend, InterpreterError};
use node::{BOp, Node, ROp, UOp};

#[cfg(any(
    feature = "cuda",
    feature = "opencl",
    feature = "wgsl",
    feature = "hsa"
))]
use compiler::{CompiledBackend, CompilerError};

#[cfg(feature = "rand")]
use rand::rngs::SmallRng;

#[cfg(feature = "cuda")]
use crate::runtime::compiler::cuda::CUDA;

#[cfg(feature = "hsa")]
use crate::runtime::compiler::hsa::HSARuntime;

#[cfg(feature = "opencl")]
use crate::runtime::compiler::opencl::OpenCLRuntime;

#[cfg(feature = "debug1")]
use std::println;

#[cfg(any(
    feature = "cuda",
    feature = "opencl",
    feature = "wgsl",
    feature = "hsa"
))]
mod compiler;
mod graph;
mod interpreter;
mod node;
//mod view;

#[derive(Debug)]
pub enum ZyxError {
    #[cfg(any(
        feature = "cuda",
        feature = "opencl",
        feature = "wgsl",
        feature = "hsa"
    ))]
    CompilerError(CompilerError),
    InterpreterError(InterpreterError),
    EmptyTensor,
}

#[cfg(any(
    feature = "cuda",
    feature = "opencl",
    feature = "wgsl",
    feature = "hsa"
))]
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

pub(crate) struct Runtime {
    graph: Graph,
    #[cfg(feature = "cuda")]
    cuda: Option<CompiledBackend<CUDA>>,
    #[cfg(feature = "hsa")]
    hsa: Option<CompiledBackend<HSARuntime>>,
    #[cfg(feature = "opencl")]
    opencl: Option<CompiledBackend<OpenCLRuntime>>,
    #[cfg(feature = "wgsl")]
    wgsl: Option<CompiledBackend<compiler::wgsl::WGSLRuntime>>,
    cpu: Option<InterpretedBackend<CPU>>,
    pub(crate) default_device: Device,
    pub(crate) default_device_set_by_user: bool,
    #[cfg(feature = "rand")]
    pub(crate) rng: core::cell::OnceCell<SmallRng>,
}

impl Runtime {
    pub(crate) const fn new() -> Self {
        Runtime {
            graph: Graph::new(),
            #[cfg(feature = "cuda")]
            cuda: None,
            #[cfg(feature = "hsa")]
            hsa: None,
            #[cfg(feature = "opencl")]
            opencl: None,
            #[cfg(feature = "wgsl")]
            wgsl: None,
            cpu: None,
            default_device: Device::CPU,
            default_device_set_by_user: false,
            #[cfg(feature = "rand")]
            rng: core::cell::OnceCell::new(),
        }
    }

    /// Tries to initialize all devices and set the first
    /// successfully initialized device as the default_device in this order:
    /// 1. CUDA
    /// 2. OpenCL
    /// 3. WGSL
    /// If they all fail to initialize, then default_device
    /// is set to CPU.
    pub(crate) fn set_default_device_best(&mut self) {
        if self.default_device_set_by_user {
            return;
        }
        #[cfg(feature = "cuda")]
        if self.initialize_device(Device::CUDA) {
            self.default_device = Device::CUDA;
            return;
        }
        #[cfg(feature = "hsa")]
        if self.initialize_device(Device::HSA) {
            self.default_device = Device::HSA;
            return;
        }
        #[cfg(feature = "opencl")]
        if self.initialize_device(Device::OpenCL) {
            self.default_device = Device::OpenCL;
            return;
        }
        #[cfg(feature = "wgsl")]
        if self.initialize_device(Device::WGSL) {
            self.default_device = Device::WGSL;
            return;
        }
        if self.initialize_device(Device::CPU) {
            self.default_device = Device::CPU;
            return;
        }
    }

    /// Returns true on successfull initialization or if device
    /// was already initialized.
    pub(crate) fn initialize_device(&mut self, device: Device) -> bool {
        match device {
            #[cfg(feature = "cuda")]
            Device::CUDA => {
                if self.cuda.is_none() {
                    if let Ok(cuda) = CompiledBackend::initialize() {
                        self.cuda = Some(cuda);
                        true
                    } else {
                        false
                    }
                } else {
                    true
                }
            }
            #[cfg(feature = "hsa")]
            Device::HSA => {
                if self.hsa.is_none() {
                    if let Ok(hsa) = CompiledBackend::initialize() {
                        self.hsa = Some(hsa);
                        true
                    } else {
                        false
                    }
                } else {
                    true
                }
            }
            #[cfg(feature = "opencl")]
            Device::OpenCL => {
                if self.opencl.is_none() {
                    if let Ok(opencl) = CompiledBackend::initialize() {
                        self.opencl = Some(opencl);
                        true
                    } else {
                        false
                    }
                } else {
                    true
                }
            }
            #[cfg(feature = "wgsl")]
            Device::WGSL => {
                if self.wgsl.is_none() {
                    if let Ok(wgsl) = CompiledBackend::initialize() {
                        self.wgsl = Some(wgsl);
                        true
                    } else {
                        false
                    }
                } else {
                    true
                }
            }
            Device::CPU => {
                if self.cpu.is_none() {
                    if let Ok(cpu) = InterpretedBackend::initialize() {
                        self.cpu = Some(cpu);
                        true
                    } else {
                        false
                    }
                } else {
                    true
                }
            }
        }
    }

    pub(crate) fn retain(&mut self, x: TensorId) {
        self.graph.retain(x);
    }

    pub(crate) fn release(&mut self, x: TensorId) -> Result<(), ZyxError> {
        let to_remove = self.graph.release(x);
        for (x, device) in to_remove {
            match device {
                #[cfg(feature = "cuda")]
                Device::CUDA => self.cuda.as_mut().unwrap().remove(x)?,
                #[cfg(feature = "hsa")]
                Device::HSA => self.hsa.as_mut().unwrap().remove(x)?,
                #[cfg(feature = "opencl")]
                Device::OpenCL => self.opencl.as_mut().unwrap().remove(x)?,
                #[cfg(feature = "wgsl")]
                Device::WGSL => self.wgsl.as_mut().unwrap().remove(x)?,
                Device::CPU => self.cpu.as_mut().unwrap().remove(x)?,
            }
        }
        return Ok(());
    }

    /*#[cfg(feature = "debug1")]
    pub(crate) fn debug_graph(&self) {
        use libc_print::std_name::println;
        for (id, node) in self.graph.nodes.iter().enumerate() {
            println!("{id:>5} x{:>3} -> {node:?}", self.graph.rcs[id]);
        }
    }*/

    pub(crate) fn shape(&self, x: TensorId) -> &[usize] {
        return self.graph.shape(x);
    }

    pub(crate) fn dtype(&self, x: TensorId) -> DType {
        return self.graph.dtype(x);
    }

    pub(crate) fn device(&self, x: TensorId) -> Device {
        return self.graph.device(x);
    }

    pub(crate) fn cast(&mut self, x: TensorId, dtype: DType) -> TensorId {
        return self.graph.push(Node::Unary {
            x,
            uop: UOp::Cast(dtype),
        });
    }

    pub(crate) fn reciprocal(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Inv });
    }

    pub(crate) fn neg(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Neg });
    }

    pub(crate) fn relu(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::ReLU });
    }

    pub(crate) fn exp(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Exp });
    }

    pub(crate) fn ln(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Ln });
    }

    pub(crate) fn sin(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Sin });
    }

    pub(crate) fn cos(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Cos });
    }

    pub(crate) fn sqrt(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Sqrt });
    }

    pub(crate) fn tanh(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Tanh });
    }

    pub(crate) fn reshape(&mut self, x: TensorId, shape: impl IntoShape) -> TensorId {
        return self.graph.push(Node::Reshape {
            x,
            shape: shape.into_shape().collect(),
        });
    }

    pub(crate) fn expand(&mut self, x: TensorId, shape: impl IntoShape) -> TensorId {
        return self.graph.push(Node::Expand {
            x,
            shape: shape.into_shape().collect(),
        });
    }

    pub(crate) fn permute(&mut self, x: TensorId, axes: impl IntoAxes) -> TensorId {
        let shape: Vec<usize> = self.shape(x).permute(axes.clone()).collect();
        return self.graph.push(Node::Permute {
            x,
            axes: axes.into_axes(shape.len()).collect(),
            shape,
        });
    }

    pub(crate) fn sum(&mut self, x: TensorId, axes: impl IntoAxes) -> TensorId {
        let shape: Vec<usize> = self.shape(x).reduce(axes.clone()).collect();
        return self.graph.push(Node::Reduce {
            x,
            axes: axes.into_axes(shape.len()).collect(),
            shape,
            rop: ROp::Sum,
        });
    }

    pub(crate) fn add(&mut self, x: TensorId, y: TensorId) -> TensorId {
        return self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Add,
        });
    }

    pub(crate) fn sub(&mut self, x: TensorId, y: TensorId) -> TensorId {
        return self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Sub,
        });
    }

    pub(crate) fn mul(&mut self, x: TensorId, y: TensorId) -> TensorId {
        return self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Mul,
        });
    }

    pub(crate) fn div(&mut self, x: TensorId, y: TensorId) -> TensorId {
        return self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Div,
        });
    }

    pub(crate) fn pow(&mut self, x: TensorId, y: TensorId) -> TensorId {
        return self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Pow,
        });
    }

    pub(crate) fn cmplt(&mut self, x: TensorId, y: TensorId) -> TensorId {
        return self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Cmplt,
        });
    }

    pub(crate) fn where_(&mut self, x: TensorId, y: TensorId, z: TensorId) -> TensorId {
        return self.graph.push(Node::Where { x, y, z });
    }
}

impl Runtime {
    pub(crate) fn store<T: Scalar>(
        &mut self,
        data: &[T],
        device: Device,
        shape: impl IntoShape,
    ) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug1")]
        println!("Storing {data:?} to {device:?} device with shape {shape:?}.");
        let node = Node::Leaf {
            shape: shape.into_shape().collect(),
            dtype: T::dtype(),
            device,
        };
        let tensor_id = self.graph.push(node);
        match device {
            #[cfg(feature = "cuda")]
            Device::CUDA => {
                if self.cuda.is_none() {
                    self.cuda = Some(CompiledBackend::initialize()?);
                }
                let dev = self.cuda.as_mut().unwrap();
                dev.store(tensor_id, data)?;
            }
            #[cfg(feature = "hsa")]
            Device::HSA => {
                if self.hsa.is_none() {
                    self.hsa = Some(CompiledBackend::initialize()?);
                }
                let dev = self.hsa.as_mut().unwrap();
                dev.store(tensor_id, data)?;
            }
            #[cfg(feature = "opencl")]
            Device::OpenCL => {
                if self.opencl.is_none() {
                    self.opencl = Some(CompiledBackend::initialize()?);
                }
                let dev = self.opencl.as_mut().unwrap();
                dev.store(tensor_id, data)?;
            }
            #[cfg(feature = "wgsl")]
            Device::WGSL => {
                if self.wgsl.is_none() {
                    self.wgsl = Some(CompiledBackend::initialize()?);
                }
                let dev = self.wgsl.as_mut().unwrap();
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
            #[cfg(feature = "cuda")]
            Device::CUDA => {
                if !self.cuda.as_ref().unwrap().is_realized(x) {
                    self.realize(BTreeSet::from_iter([x]))?;
                }
                let length = self.shape(x).iter().product();
                Ok(self.cuda.as_mut().unwrap().load(x, length)?)
            }
            #[cfg(feature = "hsa")]
            Device::HSA => {
                if !self.hsa.as_ref().unwrap().is_realized(x) {
                    self.realize(BTreeSet::from_iter([x]))?;
                }
                let length = self.shape(x).iter().product();
                Ok(self.hsa.as_mut().unwrap().load(x, length)?)
            }
            #[cfg(feature = "opencl")]
            Device::OpenCL => {
                if !self.opencl.as_ref().unwrap().is_realized(x) {
                    self.realize(BTreeSet::from_iter([x]))?;
                }
                let length = self.shape(x).iter().product();
                Ok(self.opencl.as_mut().unwrap().load(x, length)?)
            }
            #[cfg(feature = "wgsl")]
            Device::WGSL => {
                if !self.wgsl.as_ref().unwrap().is_realized(x) {
                    self.realize(BTreeSet::from_iter([x]))?;
                }
                let length = self.shape(x).iter().product();
                Ok(self.wgsl.as_mut().unwrap().load(x, length)?)
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

    fn is_realized(&self, x: TensorId) -> bool {
        return match self.device(x) {
            #[cfg(feature = "cuda")]
            Device::CUDA => {
                if let Some(cuda) = self.cuda.as_ref() {
                    cuda.is_realized(x)
                } else {
                    false
                }
            }
            #[cfg(feature = "hsa")]
            Device::HSA => {
                if let Some(hsa) = self.hsa.as_ref() {
                    hsa.is_realized(x)
                } else {
                    false
                }
            }
            #[cfg(feature = "opencl")]
            Device::OpenCL => {
                if let Some(opencl) = self.opencl.as_ref() {
                    opencl.is_realized(x)
                } else {
                    false
                }
            }
            #[cfg(feature = "wgsl")]
            Device::WGSL => {
                if let Some(wgsl) = self.wgsl.as_ref() {
                    wgsl.is_realized(x)
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

    pub(crate) fn realize(&mut self, tensors: BTreeSet<TensorId>) -> Result<(), ZyxError> {
        if tensors.len() == 0 {
            return Ok(());
        }
        let device = self.device(*tensors.first().unwrap());
        let graph = self
            .graph
            .realize_graph(&tensors, |id| self.is_realized(id));
        match device {
            #[cfg(feature = "cuda")]
            Device::CUDA => {
                self.cuda.as_mut().unwrap().compile_graph(&graph, tensors)?;
                self.cuda.as_mut().unwrap().launch_graph(&graph)?;
                return Ok(());
            }
            #[cfg(feature = "hsa")]
            Device::HSA => {
                self.hsa.as_mut().unwrap().compile_graph(&graph, tensors)?;
                self.hsa.as_mut().unwrap().launch_graph(&graph)?;
                return Ok(());
            }
            #[cfg(feature = "opencl")]
            Device::OpenCL => {
                self.opencl
                    .as_mut()
                    .unwrap()
                    .compile_graph(&graph, tensors)?;
                self.opencl.as_mut().unwrap().launch_graph(&graph)?;
                return Ok(());
            }
            #[cfg(feature = "wgsl")]
            Device::WGSL => {
                self.wgsl.as_mut().unwrap().compile_graph(&graph, tensors)?;
                self.wgsl.as_mut().unwrap().launch_graph(&graph)?;
                return Ok(());
            }
            Device::CPU => {
                self.cpu
                    .as_mut()
                    .unwrap()
                    .interpret_graph(&graph, &tensors)?;
                return Ok(());
            }
        }
    }
}
