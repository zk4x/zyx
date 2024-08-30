use crate::dtype::{Constant, DType};
use crate::index_map::IndexMap;
use crate::scalar::Scalar;
use crate::shape::Dimension;
use crate::tensor::TensorId;
use backend::cuda::{
    initialize_cuda_backend, CUDABuffer, CUDADevice, CUDAError, CUDAMemoryPool, CUDAProgram,
};
use backend::hip::{
    initialize_hip_backend, HIPBuffer, HIPDevice, HIPError, HIPMemoryPool, HIPProgram,
};
use backend::opencl::{
    initialize_opencl_backend, OpenCLBuffer, OpenCLDevice, OpenCLError, OpenCLMemoryPool,
    OpenCLProgram,
};
use backend::DeviceInfo;
use graph::Graph;
use ir::IRKernel;
use node::{BOp, Node, ROp, UOp};
use scheduler::CompiledGraph;
use std::fmt::Display;
use std::{
    collections::{btree_map::Entry, BTreeMap, BTreeSet},
    vec,
    vec::Vec,
};
use view::View;

pub use backend::cuda::CUDAConfig;
pub use backend::hip::HIPConfig;
pub use backend::opencl::OpenCLConfig;

#[cfg(feature = "rand")]
use rand::rngs::SmallRng;

#[cfg(feature = "half")]
use half::{bf16, f16};

#[cfg(feature = "complex")]
use num_complex::Complex;

mod backend;
mod graph;
mod ir;
mod node;
mod scheduler;
mod view;

// This is the whole global state of zyx
pub(crate) struct Runtime {
    // Current graph of tensor operations as nodes
    graph: Graph,
    // Random number generator
    #[cfg(feature = "rand")]
    rng: std::cell::OnceCell<SmallRng>,
    // Are we in training mode?
    pub(crate) training: bool,
    devices: Vec<Device>,
    memory_pools: Vec<MemoryPool>,
    // Cache for compiled graphs
    compiled_graph_cache: BTreeMap<Graph, CompiledGraph>,
    // Cache which maps IRKernel to device and program id on the device
    ir_kernel_cache: BTreeMap<IRKernel, (DeviceId, usize)>,
}

#[cfg_attr(feature = "py", pyo3::pyclass)]
#[derive(serde::Deserialize, Debug)]
pub struct BackendConfig {
    pub cuda: CUDAConfig,
    pub hip: HIPConfig,
    pub opencl: OpenCLConfig,
}

impl Default for BackendConfig {
    fn default() -> Self {
        BackendConfig {
            cuda: CUDAConfig::default(),
            hip: HIPConfig::default(),
            opencl: OpenCLConfig::default(),
        }
    }
}

type MemoryPoolId = usize;
type DeviceId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct BufferId {
    memory_pool_id: usize,
    buffer_id: usize,
}

#[derive(Debug)]
enum Device {
    CUDA {
        device: CUDADevice,
        memory_pool_id: MemoryPoolId,
        // Program and tensors passed as arguments for the program and if arguments are read only
        programs: Vec<CUDAProgram>,
    },
    HIP {
        device: HIPDevice,
        memory_pool_id: MemoryPoolId,
        // Program and tensors passed as arguments for the program and if arguments are read only
        programs: Vec<HIPProgram>,
    },
    OpenCL {
        device: OpenCLDevice,
        memory_pool_id: MemoryPoolId,
        // Program and tensors passed as arguments for the program and if arguments are read only
        programs: Vec<OpenCLProgram>,
    },
}

enum MemoryPool {
    CUDA {
        memory_pool: CUDAMemoryPool,
        buffers: IndexMap<CUDABuffer>,
    },
    HIP {
        memory_pool: HIPMemoryPool,
        buffers: IndexMap<HIPBuffer>,
    },
    OpenCL {
        memory_pool: OpenCLMemoryPool,
        buffers: IndexMap<OpenCLBuffer>,
    },
}

impl Runtime {
    #[must_use]
    pub(crate) const fn new() -> Self {
        Runtime {
            graph: Graph::new(),
            #[cfg(feature = "rand")]
            rng: core::cell::OnceCell::new(),
            training: false,
            compiled_graph_cache: BTreeMap::new(),
            devices: Vec::new(),
            memory_pools: Vec::new(),
            ir_kernel_cache: BTreeMap::new(),
        }
    }

    pub(crate) fn retain(&mut self, x: TensorId) {
        self.graph.retain(x);
    }

    pub(crate) fn release(&mut self, x: TensorId) -> Result<(), ZyxError> {
        let to_remove = self.graph.release(x);
        self.deallocate_tensors(to_remove)
    }

    #[cfg(feature = "rand")]
    pub(crate) fn manual_seed(&mut self, seed: u64) {
        use rand::SeedableRng;
        self.rng = std::cell::OnceCell::from(SmallRng::seed_from_u64(seed));
    }

    /// Creates dot plot of graph between given tensors
    #[must_use]
    pub(crate) fn plot_dot_graph(&self, tensors: &BTreeSet<TensorId>) -> String {
        self.graph.plot_dot_graph(tensors)
    }

    #[must_use]
    pub(crate) fn shape(&self, x: TensorId) -> &[usize] {
        return self.graph.shape(x);
    }

    #[must_use]
    pub(crate) fn dtype(&self, x: TensorId) -> DType {
        return self.graph.dtype(x);
    }

    #[cfg(feature = "rand")]
    #[must_use]
    pub(crate) fn uniform<T: Scalar>(
        &mut self,
        shape: Vec<Dimension>,
        start: T,
        end: T,
    ) -> Result<TensorId, ZyxError> {
        use rand::{distributions::Uniform, Rng, SeedableRng};
        const SEED: u64 = 69420;
        // Pass in few numbers generated randomly on cpu and then add
        // some nodes for bitshifts and such.
        let n: usize = shape.iter().product();
        match T::dtype() {
            #[cfg(feature = "half")]
            DType::BF16 => todo!(),
            #[cfg(feature = "half")]
            DType::F16 => todo!(),
            DType::F32 => {
                let range = Uniform::new(start.cast::<f32>(), end.cast::<f32>());
                self.rng.get_or_init(|| SmallRng::seed_from_u64(SEED));
                let rng = self.rng.get_mut().unwrap();
                let data: Vec<f32> = (0..n).map(|_| rng.sample(&range)).collect();
                self.temp(shape, &data)
            }
            DType::F64 => todo!(),
            #[cfg(feature = "complex")]
            DType::CF32 => todo!(),
            #[cfg(feature = "complex")]
            DType::CF64 => todo!(),
            DType::U8 => {
                let range = Uniform::new(start.cast::<u8>(), end.cast::<u8>());
                self.rng.get_or_init(|| SmallRng::seed_from_u64(SEED));
                let rng = self.rng.get_mut().unwrap();
                let data: Vec<u8> = (0..n).map(|_| rng.sample(&range)).collect();
                self.temp(shape, &data)
            }
            DType::I8 => todo!(),
            DType::I16 => todo!(),
            DType::I32 => todo!(),
            DType::I64 => todo!(),
            DType::Bool => todo!(),
        }
    }

    pub(crate) fn temp<T: Scalar>(
        &mut self,
        shape: Vec<Dimension>,
        data: &[T],
    ) -> Result<TensorId, ZyxError> {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        if data.len() == 1 {
            Ok(self.graph.push(Node::Const {
                value: Constant::new(data[0]),
            }))
        } else {
            let id = self
                .graph
                .push_wshape_and_dtype(Node::Leaf, shape, T::dtype());
            self.initialize_backends()?;
            let bytes = data.len() * T::byte_size();
            // Put it into memory pool with fastest device out of memory pools with enough free capacity
            let mem_pools: Vec<usize> = self
                .memory_pools
                .iter()
                .enumerate()
                .filter_map(|(id, mp)| {
                    if mp.free_bytes() > bytes {
                        Some(id)
                    } else {
                        None
                    }
                })
                .collect();
            if mem_pools.is_empty() {
                return Err(ZyxError::AllocationError);
            }
            // Pick memory pool with fastest device
            let mut memory_pool_id = mem_pools[0];
            let mut max_compute = 0;
            for dev in &self.devices {
                if dev.compute() > max_compute && mem_pools.contains(&dev.memory_pool_id()) {
                    max_compute = dev.compute();
                    memory_pool_id = dev.memory_pool_id();
                }
            }
            // Search for first memory pool where we can put this tensor
            let buffer_id = match &mut self.memory_pools[memory_pool_id] {
                MemoryPool::CUDA {
                    memory_pool,
                    buffers,
                } => {
                    let buffer_id = buffers.push(memory_pool.allocate(bytes)?);
                    let ptr: *const u8 = data.as_ptr().cast();
                    memory_pool.host_to_pool(
                        unsafe { std::slice::from_raw_parts(ptr, bytes) },
                        &mut buffers[buffer_id],
                    )?;
                    BufferId {
                        memory_pool_id,
                        buffer_id,
                    }
                }
                MemoryPool::HIP {
                    memory_pool,
                    buffers,
                } => {
                    let buffer_id = buffers.push(memory_pool.allocate(bytes)?);
                    let ptr: *const u8 = data.as_ptr().cast();
                    memory_pool.host_to_pool(
                        unsafe { std::slice::from_raw_parts(ptr, bytes) },
                        &mut buffers[buffer_id],
                    )?;
                    BufferId {
                        memory_pool_id,
                        buffer_id,
                    }
                }
                MemoryPool::OpenCL {
                    memory_pool,
                    buffers,
                } => {
                    let buffer_id = buffers.push(memory_pool.allocate(bytes)?);
                    let ptr: *const u8 = data.as_ptr().cast();
                    memory_pool.host_to_pool(
                        unsafe { std::slice::from_raw_parts(ptr, bytes) },
                        &mut buffers[buffer_id],
                    )?;
                    BufferId {
                        memory_pool_id,
                        buffer_id,
                    }
                }
            };
            self.graph
                .tensor_buffer_map
                .insert((id, View::new(self.shape(id))), buffer_id);
            Ok(id)
        }
    }

    // Initialization
    #[must_use]
    pub(crate) fn full(&mut self, shape: Vec<usize>, value: impl Scalar) -> TensorId {
        let one = self.graph.push(Node::Const {
            value: Constant::new(value),
        });
        let expanded = self.expand(one, shape);
        self.release(one).unwrap();
        return expanded;
    }

    #[must_use]
    pub(crate) fn ones(&mut self, shape: Vec<usize>, dtype: DType) -> TensorId {
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

    #[must_use]
    pub(crate) fn zeros(&mut self, shape: Vec<usize>, dtype: DType) -> TensorId {
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
    #[must_use]
    pub(crate) fn cast(&mut self, x: TensorId, dtype: DType) -> TensorId {
        if dtype == self.dtype(x) {
            self.retain(x);
            return x;
        }
        return self.graph.push_wdtype(
            Node::Unary {
                x,
                uop: UOp::Cast(dtype),
            },
            dtype,
        );
    }

    #[must_use]
    pub(crate) fn reciprocal(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Inv });
    }

    #[must_use]
    pub(crate) fn neg(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Neg });
    }

    #[must_use]
    pub(crate) fn relu(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::ReLU });
    }

    #[must_use]
    pub(crate) fn exp2(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Exp2 });
    }

    #[must_use]
    pub(crate) fn log2(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Log2 });
    }

    #[must_use]
    pub(crate) fn inv(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Inv });
    }

    #[must_use]
    pub(crate) fn sin(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Sin });
    }

    #[must_use]
    pub(crate) fn cos(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Cos });
    }

    #[must_use]
    pub(crate) fn sqrt(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Sqrt });
    }

    #[must_use]
    pub(crate) fn nonzero(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary {
            x,
            uop: UOp::Nonzero,
        });
    }

    #[must_use]
    pub(crate) fn not(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Not });
    }

    #[must_use]
    pub(crate) fn reshape(&mut self, x: TensorId, shape: Vec<usize>) -> TensorId {
        if &shape == self.shape(x) {
            self.retain(x);
            return x;
        }
        self.graph.push_wshape(Node::Reshape { x }, shape)
    }

    #[must_use]
    pub(crate) fn expand(&mut self, x: TensorId, shape: Vec<usize>) -> TensorId {
        if &shape == self.shape(x) {
            self.retain(x);
            return x;
        }
        self.graph.push_wshape(Node::Expand { x }, shape)
    }

    #[must_use]
    pub(crate) fn permute(&mut self, x: TensorId, axes: Vec<usize>) -> TensorId {
        let shape = permute(self.shape(x), &axes);
        self.graph.push_wshape(Node::Permute { x, axes }, shape)
    }

    #[must_use]
    pub(crate) fn pad_zeros(&mut self, x: TensorId, padding: Vec<(isize, isize)>) -> TensorId {
        let mut shape: Vec<usize> = self.shape(x).into();
        //println!("Self shape: {shape:?}, padding: {padding:?}");
        let mut i = 0;
        for d in shape.iter_mut().rev() {
            *d = (*d as isize + padding[i].0 + padding[i].1) as usize;
            i += 1;
            if i >= padding.len() {
                break;
            }
        }
        //println!("Result {shape:?}");
        self.graph.push_wshape(Node::Pad { x, padding }, shape)
    }

    #[must_use]
    pub(crate) fn sum_reduce(&mut self, x: TensorId, axes: Vec<usize>) -> TensorId {
        let shape = reduce(self.shape(x), &axes);
        self.graph.push_wshape(
            Node::Reduce {
                x,
                axes,
                rop: ROp::Sum,
            },
            shape,
        )
    }

    #[must_use]
    pub(crate) fn max_reduce(&mut self, x: TensorId, axes: Vec<usize>) -> TensorId {
        let shape = reduce(self.shape(x), &axes);
        self.graph.push_wshape(
            Node::Reduce {
                x,
                axes,
                rop: ROp::Max,
            },
            shape,
        )
    }

    #[must_use]
    pub(crate) fn add(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Add,
        })
    }

    #[must_use]
    pub(crate) fn sub(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Sub,
        })
    }

    #[must_use]
    pub(crate) fn mul(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Mul,
        })
    }

    #[must_use]
    pub(crate) fn div(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Div,
        })
    }

    #[must_use]
    pub(crate) fn pow(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Pow,
        })
    }

    #[must_use]
    pub(crate) fn cmplt(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Cmplt,
        })
    }

    #[must_use]
    pub(crate) fn maximum(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Max,
        })
    }
}

impl Runtime {
    // Initializes all available devices, creating a device for each compute
    // device and a memory pool for each physical memory.
    // Does nothing if devices were already initialized.
    // Returns error if all devices failed to initialize
    // DeviceParameters allows to disable some devices if requested
    fn initialize_backends(&mut self) -> Result<(), ZyxError> {
        if !self.devices.is_empty() {
            return Ok(());
        }

        // Search through config directories and find zyx/backend_config.json
        // If not found or failed to parse, use defaults.
        let backend_config = xdg::BaseDirectories::new()
            .map_err(|e| {
                if let Ok(_) = std::env::var("DEBUG_DEV") {
                    println!(
                        "Failed to find config directories for backend_config.json, using defaults: {e}"
                    );
                }
            })
            .ok()
            .map(|bd| {
                let mut dirs = bd.get_config_dirs();
                dirs.push(bd.get_config_home());
                dirs
            })
            .map(|paths| {
                paths.into_iter().find_map(|mut path| {
                    path.push("zyx/backend_config.json");
                    std::fs::read_to_string(&path)
                        .map_err(|e| {
                            if let Ok(_) = std::env::var("DEBUG_DEV") {
                                println!("Failed to read backend_config.json at {path:?}, using defaults: {e}");
                            }
                        })
                        .ok()
                })
            })
            .flatten()
            .map(|file| {
                serde_json::from_str(&file)
                    .map_err(|e| {
                        if let Ok(_) = std::env::var("DEBUG_DEV") {
                            println!("Failed to parse backend_config.json, using defaults: {e}");
                        }
                    })
                    .ok()
            })
            .flatten()
            .unwrap_or_else(|| BackendConfig::default());

        if let Ok((memory_pools, devices)) = initialize_cuda_backend(&backend_config.cuda) {
            let n = self.memory_pools.len();
            self.memory_pools
                .extend(memory_pools.into_iter().map(|m| MemoryPool::CUDA {
                    memory_pool: m,
                    buffers: IndexMap::new(),
                }));
            self.devices
                .extend(devices.into_iter().map(|device| Device::CUDA {
                    memory_pool_id: device.memory_pool_id() + n,
                    device,
                    programs: Vec::new(),
                }));
        }
        if let Ok((memory_pools, devices)) = initialize_hip_backend(&backend_config.hip) {
            let n = self.memory_pools.len();
            self.memory_pools
                .extend(memory_pools.into_iter().map(|m| MemoryPool::HIP {
                    memory_pool: m,
                    buffers: IndexMap::new(),
                }));
            self.devices
                .extend(devices.into_iter().map(|device| Device::HIP {
                    memory_pool_id: device.memory_pool_id() + n,
                    device,
                    programs: Vec::new(),
                }));
        }
        if let Ok((memory_pools, devices)) = initialize_opencl_backend(&backend_config.opencl) {
            let n = self.memory_pools.len();
            self.memory_pools
                .extend(memory_pools.into_iter().map(|m| MemoryPool::OpenCL {
                    memory_pool: m,
                    buffers: IndexMap::new(),
                }));
            self.devices
                .extend(devices.into_iter().map(|device| Device::OpenCL {
                    memory_pool_id: device.memory_pool_id() + n,
                    device,
                    programs: Vec::new(),
                }));
        }
        if self.devices.is_empty() {
            return Err(ZyxError::NoBackendAvailable);
        }
        Ok(())
    }

    pub(crate) fn load<T: Scalar>(&mut self, x: TensorId) -> Result<Vec<T>, ZyxError> {
        // Check if tensor is evaluated
        if self
            .graph
            .tensor_buffer_map
            .iter()
            .all(|((id, _), _)| *id != x)
        {
            self.realize(BTreeSet::from([x]))?;
        }
        // If at least part of tensor exists in some device, there must be
        // the rest of the tensor in other devices
        let n: usize = self.shape(x).iter().product();
        let mut data: Vec<T> = Vec::with_capacity(n);
        for ((tensor_id, view), buffer_id) in &self.graph.tensor_buffer_map {
            if *tensor_id == x {
                if view.numel() == n {
                    let slice = unsafe {
                        std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), n * T::byte_size())
                    };
                    match &mut self.memory_pools[buffer_id.memory_pool_id] {
                        MemoryPool::CUDA {
                            memory_pool,
                            buffers,
                        } => {
                            memory_pool.pool_to_host(&buffers[buffer_id.buffer_id], slice)?;
                        }
                        MemoryPool::HIP {
                            memory_pool,
                            buffers,
                        } => {
                            memory_pool.pool_to_host(&buffers[buffer_id.buffer_id], slice)?;
                        }
                        MemoryPool::OpenCL {
                            memory_pool,
                            buffers,
                        } => {
                            memory_pool.pool_to_host(&buffers[buffer_id.buffer_id], slice)?;
                        }
                    }
                    break;
                } else {
                    todo!()
                }
            }
        }
        unsafe { data.set_len(n) };
        // for each device where tensor is stored load it
        Ok(data)
    }

    pub(crate) fn realize(&mut self, tensors: BTreeSet<TensorId>) -> Result<(), ZyxError> {
        // Runs in O(4n) where n = self.graph.len(),
        // first pass for visited nodes, second pass for outisde_rcs, third pass for order,
        // fourth pass for to_delete and new_leafs
        // Could possibly be optimized a bit
        if tensors.len() == 0 {
            return Ok(());
        }
        if tensors.iter().all(|tensor| {
            self.graph
                .tensor_buffer_map
                .iter()
                .any(|((t, _), _)| tensor == t)
        }) {
            return Ok(());
        }
        if self.devices.is_empty() {
            self.initialize_backends()?;
        }
        // Get rcs of nodes outside of realized graph
        let (graph, outside_nodes, order) = self.graph.realize_graph(&tensors, |x| {
            self.graph
                .tensor_buffer_map
                .iter()
                .any(|((id, _), _)| *id == x)
        });
        // Which parts of graph are no longer needed and can be deleted and which nodes will be new leafs?
        // New leafs never store data, so we can deallocate them if they are allocated.
        let mut to_delete = BTreeSet::new();
        let mut new_leafs = BTreeSet::new();
        //println!("Graph: {:?}", graph);
        //println!("Outside nodes: {outside_nodes:?}");
        //println!("Order: {order:?}");
        for tensor in &order {
            if matches!(self.graph[*tensor], Node::Leaf | Node::Const { .. }) {
                if !outside_nodes.contains(tensor) {
                    to_delete.insert(*tensor);
                    continue;
                }
            } else {
                if self.graph[*tensor]
                    .parameters()
                    .all(|tensor| to_delete.contains(&tensor))
                    && !outside_nodes.contains(tensor)
                {
                    to_delete.insert(*tensor);
                } else if self.graph[*tensor]
                    .parameters()
                    .any(|tensor| to_delete.contains(&tensor))
                {
                    for param in self.graph[*tensor].parameters() {
                        if to_delete.contains(&param) {
                            new_leafs.insert(param);
                        }
                    }
                }
            }
        }
        //println!("New leafs: {new_leafs:?}");
        // Compile and launch
        if !self.compiled_graph_cache.contains_key(&graph) {
            // Also realize nodes that will become new leafs... but why?
            //tensors.extend(&new_leafs);
            let compiled_graph = self.compile_graph(graph.clone(), &tensors)?;
            self.compiled_graph_cache
                .insert(graph.clone(), compiled_graph);
        }
        self.launch_graph(&graph)?;
        self.deallocate_tensors(to_delete.clone())?;
        // Remove evaluated part of graph unless needed for backpropagation
        for tensor in new_leafs {
            self.graph.add_shape_dtype(tensor);
            self.graph[tensor] = Node::Leaf;
            to_delete.remove(&tensor);
        }
        //println!("To delete: {to_delete:?}");
        // Delete the node, but do not use release function, just remove it from graph.nodes
        // and deallocate it from device
        self.graph.delete_tensors(&to_delete);
        return Ok(());
    }

    fn deallocate_tensors(&mut self, to_remove: BTreeSet<TensorId>) -> Result<(), ZyxError> {
        let mut buffers: Vec<BufferId> = Vec::new();
        for tensor in to_remove {
            for (_, buffer_id) in self
                .graph
                .tensor_buffer_map
                .iter()
                .filter(|((t, _), _)| *t == tensor)
            {
                buffers.push(*buffer_id);
            }
        }
        self.graph
            .tensor_buffer_map
            .retain(|_, b| !buffers.contains(b));
        for buffer in buffers {
            match &mut self.memory_pools[buffer.memory_pool_id] {
                MemoryPool::CUDA {
                    memory_pool,
                    buffers,
                } => {
                    let buffer = buffers.remove(buffer.buffer_id).unwrap();
                    memory_pool.deallocate(buffer)?;
                }
                MemoryPool::HIP {
                    memory_pool,
                    buffers,
                } => {
                    let buffer = buffers.remove(buffer.buffer_id).unwrap();
                    memory_pool.deallocate(buffer)?;
                }
                MemoryPool::OpenCL {
                    memory_pool,
                    buffers,
                } => {
                    let buffer = buffers.remove(buffer.buffer_id).unwrap();
                    memory_pool.deallocate(buffer)?;
                }
            }
        }
        return Ok(());
    }

    pub(crate) fn backward(
        &mut self,
        x: TensorId,
        sources: BTreeSet<TensorId>,
    ) -> BTreeMap<TensorId, TensorId> {
        // Does not allocate new tensors, only constant and op nodes
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
        let grad1 = self.ones(vec![1], self.dtype(x));
        let sh: Vec<Dimension> = self.shape(x).into();
        grads.insert(x, self.graph.push_wshape(Node::Expand { x: grad1 }, sh));
        self.release(grad1).unwrap();
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
                Node::Const { .. } | Node::Leaf { .. } => {}
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
                            let grad = self.mul(y, grad);
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
                            let two_temp = self.ones(vec![1], dtype);
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
                            let ones = self.ones(self.shape(y).into(), self.dtype(y));
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
                            let temp1 = self.log2(x);
                            let temp2 = self.mul(nid, temp1);
                            self.release(temp1).unwrap();
                            let y_grad = self.mul(grad, temp2);
                            self.release(temp2).unwrap();
                            insert_or_add_grad(self, &mut grads, y, y_grad);
                        }
                    }
                    BOp::Cmplt | BOp::Cmpgt => {
                        panic!("Comparison is not a differentiable operation.");
                    }
                    BOp::Max => {
                        todo!("Max backward.");
                    }
                    BOp::Or => {
                        todo!("Or backward.");
                    }
                },
                Node::Unary { x, uop } => match uop {
                    UOp::Inv => {
                        // -1/(x*x)
                        let x_2_inv = self.mul(nid, nid);
                        let x_grad = self.neg(x_2_inv);
                        self.release(x_2_inv).unwrap();
                        insert_or_add_grad(self, &mut grads, x, x_grad);
                    }
                    UOp::ReLU => {
                        let zeros = self.zeros(self.shape(x).into(), self.dtype(x));
                        let zl = self.cmplt(zeros, x);
                        self.release(zeros).unwrap();
                        let x_grad = self.mul(zl, grad);
                        self.release(zl).unwrap();
                        insert_or_add_grad(self, &mut grads, x, x_grad);
                    }
                    UOp::Exp2 => {
                        let temp = self.full(self.shape(x).into(), std::f64::consts::E.log2());
                        let temp2 = self.mul(nid, temp);
                        self.release(temp).unwrap();
                        let grad = self.mul(nid, temp2);
                        self.release(temp2).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Log2 => {
                        let temp = self.full(self.shape(x).into(), std::f64::consts::E.log2());
                        let temp2 = self.mul(x, temp);
                        self.release(temp).unwrap();
                        let grad = self.div(grad, temp2);
                        self.release(temp2).unwrap();
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
                    /*UOp::Tanh => {
                        // 1 - tanh^2(x)
                        let tanh_x_2 = self.mul(nid, nid);
                        let ones = self.ones(self.shape(x).into(), self.dtype(x));
                        let grad = self.sub(ones, tanh_x_2);
                        self.release(ones).unwrap();
                        self.release(tanh_x_2).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }*/
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
                Node::Expand { x } => {
                    let shape = self.graph.shape(nid);
                    let mut vec: Vec<Dimension> = shape.into();
                    while vec.len() < shape.len() {
                        vec.insert(0, 1);
                    }
                    let expand_axes: Vec<usize> = vec
                        .into_iter()
                        .zip(shape)
                        .enumerate()
                        .filter_map(|(a, (d, e))| if d == *e { None } else { Some(a) })
                        .collect();
                    let temp = self.sum_reduce(grad, expand_axes);
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
                Node::Pad { x, ref padding, .. } => {
                    let inv_padding = padding.iter().map(|(lp, rp)| (-lp, -rp)).collect();
                    let grad = self.pad_zeros(grad, inv_padding);
                    insert_or_add_grad(self, &mut grads, x, grad);
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
                        let ones = self.zeros(x_shape, self.dtype(x));
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
                self.release(v).unwrap();
            }
        }
        return res;
    }
}

impl MemoryPool {
    fn free_bytes(&self) -> usize {
        match self {
            MemoryPool::CUDA { memory_pool, .. } => memory_pool.free_bytes(),
            MemoryPool::HIP { memory_pool, .. } => memory_pool.free_bytes(),
            MemoryPool::OpenCL { memory_pool, .. } => memory_pool.free_bytes(),
        }
    }
}

fn permute(shape: &[usize], axes: &[usize]) -> Vec<usize> {
    axes.iter().map(|a| shape[*a]).collect()
}

fn reduce(shape: &[usize], axes: &[usize]) -> Vec<usize> {
    let res: Vec<usize> = shape
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(i, d)| if axes.contains(&i) { None } else { Some(d) })
        .collect();
    if res.is_empty() {
        vec![1]
    } else {
        res
    }
}

#[derive(Debug)]
pub enum ZyxError {
    EmptyTensor,
    BackendConfig(&'static str),
    WrongDType(&'static str),
    NoBackendAvailable,
    AllocationError,
    CUDAError(CUDAError),
    HIPError(HIPError),
    OpenCLError(OpenCLError),
}

impl From<CUDAError> for ZyxError {
    fn from(value: CUDAError) -> Self {
        ZyxError::CUDAError(value)
    }
}

impl From<HIPError> for ZyxError {
    fn from(value: HIPError) -> Self {
        ZyxError::HIPError(value)
    }
}

impl From<OpenCLError> for ZyxError {
    fn from(value: OpenCLError) -> Self {
        ZyxError::OpenCLError(value)
    }
}

impl Device {
    fn compute(&self) -> u128 {
        match self {
            Device::CUDA {
                device,
                memory_pool_id: _,
                programs: _,
            } => device.info().compute,
            Device::HIP {
                device,
                memory_pool_id: _,
                programs: _,
            } => device.info().compute,
            Device::OpenCL {
                device,
                memory_pool_id: _,
                programs: _,
            } => device.info().compute,
        }
    }

    fn memory_pool_id(&self) -> MemoryPoolId {
        match self {
            Device::CUDA { memory_pool_id, .. } => *memory_pool_id,
            Device::HIP { memory_pool_id, .. } => *memory_pool_id,
            Device::OpenCL { memory_pool_id, .. } => *memory_pool_id,
        }
    }

    fn info(&self) -> &DeviceInfo {
        match self {
            Device::CUDA { device, .. } => device.info(),
            Device::HIP { device, .. } => device.info(),
            Device::OpenCL { device, .. } => device.info(),
        }
    }
}

impl Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::CUDA {
                device: _,
                memory_pool_id,
                programs: _,
            } => f.write_fmt(format_args!(
                "Device {{ memory_pool_id: {memory_pool_id} }})"
            )),
            Device::HIP {
                device: _,
                memory_pool_id,
                programs: _,
            } => f.write_fmt(format_args!(
                "Device {{ memory_pool_id: {memory_pool_id} }})"
            )),
            Device::OpenCL {
                device: _,
                memory_pool_id,
                programs: _,
            } => f.write_fmt(format_args!(
                "Device {{ memory_pool_id: {memory_pool_id} }})"
            )),
        }
    }
}
