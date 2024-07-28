use crate::device::Device;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::tensor::TensorId;
use alloc::vec;
use alloc::{
    collections::{btree_map::Entry, BTreeMap, BTreeSet},
    vec::Vec,
};
use custom::{
    cpu::CPU,
    {CustomError, InterpretedBackend},
};
use graph::Graph;
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

#[cfg(feature = "half")]
use half::{bf16, f16};

#[cfg(feature = "complex")]
use num_complex::Complex;

#[cfg(any(
    feature = "cuda",
    feature = "opencl",
    feature = "wgsl",
    feature = "hsa"
))]
mod compiler;
mod custom;
mod graph;
mod node;

fn permute(shape: &[usize], axes: &[usize]) -> Vec<usize> {
    axes.iter().map(|a| shape[*a]).collect()
}

fn reduce(shape: &[usize], axes: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(i, d)| if axes.contains(&i) { None } else { Some(d) })
        .collect()
}

#[derive(Debug)]
pub enum ZyxError {
    #[cfg(any(
        feature = "cuda",
        feature = "opencl",
        feature = "wgsl",
        feature = "hsa"
    ))]
    CompilerError(CompilerError),
    CustomError(CustomError),
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

impl From<CustomError> for ZyxError {
    fn from(value: CustomError) -> Self {
        Self::CustomError(value)
    }
}

pub(crate) struct Runtime {
    graph: Graph,
    #[cfg(feature = "cuda")]
    cuda: Option<CompiledBackend<compiler::cuda::CUDARuntime>>,
    #[cfg(feature = "hsa")]
    hsa: Option<CompiledBackend<compiler::hsa::HSARuntime>>,
    #[cfg(feature = "opencl")]
    opencl: Option<CompiledBackend<compiler::opencl::OpenCLRuntime>>,
    #[cfg(feature = "wgsl")]
    wgsl: Option<CompiledBackend<compiler::wgsl::WGSLRuntime>>,
    cpu: Option<InterpretedBackend<CPU>>,
    pub(crate) default_device: Device,
    pub(crate) default_device_set_by_user: bool,
    #[cfg(feature = "rand")]
    pub(crate) rng: core::cell::OnceCell<SmallRng>,
    training: bool,
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
            training: false,
        }
    }

    /// If default device was not set by the user, this function
    /// tries to initialize all devices and set the first
    /// successfully initialized device as the default_device in this order:
    /// 1. CUDA
    /// 2. HSA
    /// 3. OpenCL
    /// 4. WGSL
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

    // Initialization
    pub(crate) fn full(
        &mut self,
        shape: Vec<usize>,
        value: impl Scalar,
    ) -> Result<TensorId, ZyxError> {
        let one = self.store(vec![value], vec![1])?;
        let expanded = self.expand(one, shape);
        self.release(one)?;
        return Ok(expanded);
    }

    pub(crate) fn ones(&mut self, shape: Vec<usize>, dtype: DType) -> Result<TensorId, ZyxError> {
        return match dtype {
            #[cfg(feature = "half")]
            DType::BF16 => self.full(shape, bf16::ONE),
            #[cfg(feature = "half")]
            DType::F16 => self.full(shape, f16::ONE),
            DType::F32 => self.full(shape, 1f32),
            DType::F64 => self.full(shape, 1f64),
            #[cfg(feature = "complex")]
            DType::CF32 => self.full(shape, Complex::new(1f32, 0.)),
            #[cfg(feature = "complex")]
            DType::CF64 => self.full(shape, Complex::new(1f64, 0.)),
            DType::U8 => self.full(shape, 1u8),
            DType::I8 => self.full(shape, 1i8),
            DType::I16 => self.full(shape, 1i16),
            DType::I32 => self.full(shape, 1i32),
            DType::I64 => self.full(shape, 1i64),
            DType::Bool => self.full(shape, true),
        };
    }

    pub(crate) fn zeros(&mut self, shape: Vec<usize>, dtype: DType) -> Result<TensorId, ZyxError> {
        return match dtype {
            #[cfg(feature = "half")]
            DType::BF16 => self.full(shape, bf16::ZERO),
            #[cfg(feature = "half")]
            DType::F16 => self.full(shape, f16::ZERO),
            DType::F32 => self.full(shape, 0f32),
            DType::F64 => self.full(shape, 0f64),
            #[cfg(feature = "complex")]
            DType::CF32 => self.full(shape, Complex::new(0f32, 0.)),
            #[cfg(feature = "complex")]
            DType::CF64 => self.full(shape, Complex::new(0f64, 0.)),
            DType::U8 => self.full(shape, 0u8),
            DType::I8 => self.full(shape, 0i8),
            DType::I16 => self.full(shape, 0i16),
            DType::I32 => self.full(shape, 0i32),
            DType::I64 => self.full(shape, 0i64),
            DType::Bool => self.full(shape, false),
        };
    }

    // Unary ops
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

    pub(crate) fn nonzero(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary {
            x,
            uop: UOp::Nonzero,
        });
    }

    pub(crate) fn not(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Not });
    }

    pub(crate) fn reshape(&mut self, x: TensorId, shape: Vec<usize>) -> TensorId {
        return self.graph.push(Node::Reshape { x, shape });
    }

    pub(crate) fn expand(&mut self, x: TensorId, shape: Vec<usize>) -> TensorId {
        return self.graph.push(Node::Expand { x, shape });
    }

    pub(crate) fn permute(&mut self, x: TensorId, axes: Vec<usize>) -> TensorId {
        let shape = permute(self.shape(x), &axes);
        return self.graph.push(Node::Permute { x, axes, shape });
    }

    pub(crate) fn sum(&mut self, x: TensorId, axes: Vec<usize>) -> TensorId {
        let shape = reduce(self.shape(x), &axes);
        return self.graph.push(Node::Reduce {
            x,
            axes,
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
}

impl Runtime {
    pub(crate) fn store<T: Scalar>(
        &mut self,
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> Result<TensorId, ZyxError> {
        self.set_default_device_best();
        let device = self.default_device;
        //#[cfg(feature = "debug1")]
        //println!("Storing {data:?} to {device:?} device with shape {shape:?}.");
        let tensor_id = self.graph.push(Node::Leaf {
            shape,
            dtype: T::dtype(),
            device,
        });
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

    pub(crate) fn realize(&mut self, tensors: BTreeSet<TensorId>) -> Result<(), ZyxError> {
        if tensors.len() == 0 {
            return Ok(());
        }
        let device = self.device(*tensors.first().unwrap());
        let graph = match device {
            #[cfg(feature = "cuda")]
            Device::CUDA => self
                .graph
                .realize_graph(&tensors, |id| self.cuda.as_ref().unwrap().is_realized(id)),
            #[cfg(feature = "hsa")]
            Device::HSA => self
                .graph
                .realize_graph(&tensors, |id| self.hsa.as_ref().unwrap().is_realized(id)),
            #[cfg(feature = "opencl")]
            Device::OpenCL => self
                .graph
                .realize_graph(&tensors, |id| self.opencl.as_ref().unwrap().is_realized(id)),
            #[cfg(feature = "wgsl")]
            Device::WGSL => self
                .graph
                .realize_graph(&tensors, |id| self.wgsl.as_ref().unwrap().is_realized(id)),
            Device::CPU => self
                .graph
                .realize_graph(&tensors, |id| self.cpu.as_ref().unwrap().is_realized(id)),
        };
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
                    .interpret_graph(graph, &tensors)?;
                return Ok(());
            }
        }
    }

    pub(crate) fn backward(
        &mut self,
        x: TensorId,
        sources: BTreeSet<TensorId>,
    ) -> Result<BTreeMap<TensorId, TensorId>, ZyxError> {
        let topo = self.graph.build_topo(x, &sources);
        //std::println!("Topo: {topo:?}");

        let req_grad: BTreeSet<TensorId> = topo
            .iter()
            .copied()
            .chain(sources.iter().copied())
            .collect();
        // Node -> Grad
        let mut grads: BTreeMap<TensorId, TensorId> = BTreeMap::new();
        // Initial gradient of ones
        let grad1 = self.ones(vec![1], self.dtype(x))?;
        let sh = self.shape(x).into();
        grads.insert(
            x,
            self.graph.push(Node::Expand {
                x: grad1,
                shape: sh,
            }),
        );
        self.release(grad1)?;
        //std::println!("{:?}", self.nodes.last().unwrap());

        fn insert_or_add_grad(
            r: &mut Runtime,
            grads: &mut BTreeMap<TensorId, TensorId>,
            x: TensorId,
            grad: TensorId,
        ) {
            match grads.entry(x) {
                Entry::Vacant(e) => {
                    e.insert(grad);
                }
                Entry::Occupied(e) => {
                    let (k, prev_grad) = e.remove_entry();
                    grads.insert(
                        k,
                        r.graph.push(Node::Binary {
                            x: prev_grad,
                            y: grad,
                            bop: BOp::Add,
                        }),
                    );
                    // These can never fail as it just decreses ref count,
                    // there is no deallocation.
                    r.release(prev_grad).unwrap();
                    r.release(grad).unwrap();
                }
            }
        }

        // All releases that cannot fail use unwrap to catch incorrect refcounts immediatelly.
        // reverse gradient calculation
        for nid in topo {
            let grad = grads[&nid];
            match self.graph[nid] {
                //Node::Const(..) | Node::Detach(..) => {}
                Node::Leaf { .. } => {}
                Node::Binary { x, y, bop } => match bop {
                    BOp::Add => {
                        if req_grad.contains(&x) {
                            self.retain(grad);
                            insert_or_add_grad(self, &mut grads, x, grad);
                        }
                        if req_grad.contains(&y) {
                            self.retain(grad);
                            insert_or_add_grad(self, &mut grads, y, grad);
                        }
                    }
                    BOp::Sub => {
                        if req_grad.contains(&x) {
                            self.retain(grad);
                            insert_or_add_grad(self, &mut grads, x, grad);
                        }
                        if req_grad.contains(&y) {
                            let grad = self.neg(grad);
                            insert_or_add_grad(self, &mut grads, y, grad);
                        }
                    }
                    BOp::Mul => {
                        if req_grad.contains(&x) {
                            let graph = self.mul(y, grad);
                            insert_or_add_grad(self, &mut grads, x, grad);
                        }
                        if req_grad.contains(&y) {
                            let grad = self.mul(x, grad);
                            insert_or_add_grad(self, &mut grads, y, grad);
                        }
                    }
                    BOp::Div => {
                        if req_grad.contains(&x) {
                            grads.insert(x, self.div(grad, y));
                            insert_or_add_grad(self, &mut grads, x, grad);
                        }
                        if req_grad.contains(&y) {
                            // -grad*x/(y^2)
                            let dtype = self.dtype(y);
                            let two_temp = self.ones(vec![1], dtype)?;
                            let two = self.add(two_temp, two_temp);
                            self.release(two_temp).unwrap();
                            let two_e = self.expand(two, self.shape(y).into());
                            self.release(two).unwrap();
                            let two_2 = self.pow(y, two_e);
                            self.release(two_e).unwrap();
                            let temp = self.mul(x, grad);
                            let temp_neg = self.neg(temp);
                            self.release(temp).unwrap();
                            let y_grad = self.div(temp_neg, two_2);
                            self.release(temp_neg).unwrap();
                            self.release(two_2).unwrap();
                            grads.insert(y, y_grad);
                            insert_or_add_grad(self, &mut grads, y, grad);
                        }
                    }
                    BOp::Pow => {
                        if req_grad.contains(&x) {
                            // grad * y * x.pow(y-1)
                            let ones = self.ones(self.shape(y).into(), self.dtype(y))?;
                            let y_1 = self.sub(y, ones);
                            self.release(ones).unwrap();
                            let pow_y_1 = self.pow(x, y_1);
                            self.release(y_1).unwrap();
                            let y_mul = self.mul(y, pow_y_1);
                            self.release(pow_y_1).unwrap();
                            let x_grad = self.mul(grad, y_mul);
                            self.release(y_mul).unwrap();
                            insert_or_add_grad(self, &mut grads, x, x_grad);
                        }
                        if req_grad.contains(&y) {
                            // grad * x.pow(y) * ln(x)
                            let temp1 = self.ln(x);
                            let temp2 = self.mul(nid, temp1);
                            self.release(temp1).unwrap();
                            let y_grad = self.mul(grad, temp2);
                            self.release(temp2).unwrap();
                            insert_or_add_grad(self, &mut grads, y, y_grad);
                        }
                    }
                    BOp::Cmplt => {
                        panic!(
                            "Compare less than (cmplt, operator <) is not a differentiable operation."
                        );
                    }
                    BOp::Max => {
                        todo!("Max backward.");
                    }
                },
                Node::Unary { x, uop } => match uop {
                    UOp::Noop => {
                        panic!()
                    }
                    UOp::Inv => {
                        // -1/(x*x)
                        let x_2 = self.mul(x, x);
                        let x_2_inv = self.reciprocal(x_2);
                        self.release(x_2).unwrap();
                        let x_grad = self.neg(x_2);
                        insert_or_add_grad(self, &mut grads, x, x_grad);
                    }
                    UOp::ReLU => {
                        let zeros = self.zeros(self.shape(x).into(), self.dtype(x))?;
                        let zl = self.cmplt(zeros, x);
                        self.release(zeros).unwrap();
                        let x_grad = self.mul(zl, grad);
                        self.release(zl).unwrap();
                        insert_or_add_grad(self, &mut grads, x, x_grad);
                    }
                    UOp::Exp => {
                        let grad = self.mul(nid, grad);
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Ln => {
                        let grad = self.div(grad, x);
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Sin => {
                        let x_temp = self.cos(x);
                        let grad = self.mul(x_temp, grad);
                        self.release(x_temp).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Cos => {
                        let x_temp1 = self.sin(x);
                        let x_temp = self.neg(x_temp1);
                        self.release(x_temp1).unwrap();
                        let grad = self.mul(x_temp, grad);
                        self.release(x_temp).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Sqrt => {
                        // x_grad = grad/(2*sqrt(x))
                        let sqrt_x = self.sqrt(x);
                        let sqrtx_2 = self.add(sqrt_x, sqrt_x);
                        self.release(sqrt_x).unwrap();
                        let grad = self.div(grad, sqrtx_2);
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Cast(_) => {
                        let grad = self.cast(grad, self.dtype(x));
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Neg => {
                        let grad = self.neg(grad);
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Tanh => {
                        // 1 - tanh^2(x)
                        let tanh_x_2 = self.mul(nid, nid);
                        let ones = self.ones(self.shape(x).into(), self.dtype(x))?;
                        let grad = self.sub(ones, tanh_x_2);
                        self.release(ones).unwrap();
                        self.release(tanh_x_2).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Not => {
                        todo!("Not backward")
                        //self.x.e(TernaryOps.WHERE, grad_output, grad_output.const(0)) if self.needs_input_grad[1] else None,
                        //self.x.e(TernaryOps.WHERE, grad_output.const(0), grad_output) if self.needs_input_grad[2] else None
                    }
                    UOp::Nonzero => {
                        todo!("Nonzero backward")
                    }
                },
                Node::Reshape { x, .. } => {
                    let grad = self.reshape(grad, self.shape(x).into());
                    insert_or_add_grad(self, &mut grads, x, grad);
                }
                Node::Expand { x, ref shape } => {
                    let mut vec = shape.clone();
                    while vec.len() < shape.len() {
                        vec.insert(0, 1);
                    }
                    let expand_axes: Vec<usize> = vec
                        .into_iter()
                        .zip(shape)
                        .enumerate()
                        .filter_map(|(a, (d, e))| if d == *e { None } else { Some(a) })
                        .collect();
                    let temp = self.sum(grad, expand_axes);
                    let grad = self.reshape(temp, self.shape(x).into());
                    self.release(temp).unwrap();
                    insert_or_add_grad(self, &mut grads, x, grad);
                }
                Node::Permute { x, ref axes, .. } => {
                    let mut axes: Vec<(usize, usize)> = axes.iter().copied().enumerate().collect();
                    axes.sort_by_key(|(_, v)| *v);
                    let argsort_axes = axes.iter().map(|(k, _)| *k).collect();
                    let grad = self.permute(grad, argsort_axes);
                    insert_or_add_grad(self, &mut grads, x, grad);
                }
                Node::Pad { x, ref pad, .. } => {
                    todo!("Padding backward");
                    /*let sh = get_shape(&self.nodes, x).clone();
                    let inv_padding = padding.iter().map(|(lp, rp)| (-lp, -rp)).collect();
                    let grad = self.push(Node::Pad(grad, inv_padding, sh))?;
                    insert_or_add_grad(self, &mut grads, x, grad)?;*/
                }
                Node::Reduce { x, rop, .. } => match rop {
                    ROp::Sum => {
                        let grad = self.expand(grad, self.shape(x).into());
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    ROp::Max => {
                        // x_grad = (1 - (x < z.expand(x.shape()))) * grad
                        let x_shape: Vec<usize> = self.shape(x).into();
                        let z_temp = self.expand(nid, x_shape.clone());
                        let cmp_t = self.cmplt(x, z_temp);
                        self.release(z_temp).unwrap();
                        let ones = self.zeros(x_shape, self.dtype(x))?;
                        let max_1s = self.sub(ones, cmp_t);
                        self.release(ones).unwrap();
                        self.release(cmp_t).unwrap();
                        let grad = self.mul(max_1s, grad);
                        self.release(max_1s).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                },
            }
        }
        let mut res = BTreeMap::new();
        for (k, v) in grads.into_iter() {
            if sources.contains(&k) {
                res.insert(k, v);
            } else {
                self.release(v)?;
            }
        }
        return Ok(res);
    }
}
