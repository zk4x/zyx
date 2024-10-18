use crate::dtype::{Constant, DType};
use crate::scalar::Scalar;
use crate::shape::{permute, reduce, Dimension};
use crate::tensor::TensorId;
use backend::{
    BufferId, CUDAConfig, CUDAError, Device, DeviceId, DeviceInfo, HIPConfig, HIPError, MemoryPool,
    OpenCLConfig, OpenCLError, VulkanConfig, VulkanError,
};
#[cfg(feature = "wgsl")]
use backend::{WGSLConfig, WGSLError};
use graph::Graph;
use ir::IRKernel;
use node::{BOp, Node, ROp, UOp};
use scheduler::{CompiledGraph, Kernel, KernelOptimizer};
use std::path::PathBuf;
use std::{
    collections::{btree_map::Entry, BTreeMap, BTreeSet},
    vec,
    vec::Vec,
};
use view::View;

#[cfg(feature = "rand")]
use rand::rngs::SmallRng;

use half::{bf16, f16};

#[cfg(feature = "complex")]
use num_complex::Complex;

mod backend;
mod graph;
mod ir;
mod node;
mod scheduler;
mod view;

/// Device configuration
#[cfg_attr(feature = "py", pyo3::pyclass)]
#[derive(serde::Deserialize, Debug, Default)]
pub struct DeviceConfig {
    /// CUDA configuration
    pub cuda: CUDAConfig,
    /// HIP configuration
    pub hip: HIPConfig,
    /// OpenCL configuration
    pub opencl: OpenCLConfig,
    /// Vulkan configuration
    pub vulkan: VulkanConfig,
    /// WGSL configuration
    #[cfg(feature = "wgsl")]
    pub wgsl: WGSLConfig,
}

// This is the whole global state of zyx
pub(super) struct Runtime {
    // Current graph of tensor operations as nodes
    graph: Graph,
    // Random number generator
    #[cfg(feature = "rand")]
    pub(super) rng: std::cell::OnceCell<SmallRng>,
    // Cache for compiled graphs
    compiled_graph_cache: BTreeMap<Graph, CompiledGraph>,
    memory_pools: Vec<MemoryPool>,
    // Where are tensors stored
    tensor_buffer_map: BTreeMap<(TensorId, View), BufferId>,
    devices: Vec<Device>,
    // Cache which maps IRKernel to device and program id on the device
    ir_kernel_cache: BTreeMap<IRKernel, (DeviceId, usize)>,
    optimizer_cache: BTreeMap<(Kernel, DeviceInfo), KernelOptimizer>,
    config_dir: Option<PathBuf>, // Why the hell isn't PathBuf::new const?????
    // Are we in training mode?
    pub(super) training: bool,
    pub(super) search_iterations: usize,
    pub(super) debug: u32,
}

impl Runtime {
    #[must_use]
    pub(super) const fn new() -> Self {
        Runtime {
            compiled_graph_cache: BTreeMap::new(),
            tensor_buffer_map: BTreeMap::new(),
            graph: Graph::new(),
            ir_kernel_cache: BTreeMap::new(),
            devices: Vec::new(),
            memory_pools: Vec::new(),
            #[cfg(feature = "rand")]
            rng: core::cell::OnceCell::new(),
            config_dir: None,
            optimizer_cache: BTreeMap::new(),
            training: false,
            search_iterations: 5,
            debug: 0,
        }
    }

    pub(super) fn retain(&mut self, x: TensorId) {
        self.graph.retain(x);
    }

    pub(super) fn release(&mut self, x: TensorId) -> Result<(), ZyxError> {
        let to_remove = self.graph.release(x);
        self.deallocate_tensors(to_remove)?;
        // TODO Check the number of tensors. If there are no tensors remaining, deinitialize the runtime,
        // since rust does not implement drop for us.
        if self.graph.is_empty() && self.tensor_buffer_map.is_empty() {
            self.deinitialize()?;
        }
        Ok(())
    }

    /// This function deinitializes the whole runtime, deallocates all allocated memory and deallocates all caches
    /// It does not reset the rng and it does not change debug, search, training and config_dir fields
    fn deinitialize(&mut self) -> Result<(), ZyxError> {
        //println!("Deinitialize");
        // drop compiled graph cache
        self.compiled_graph_cache = BTreeMap::new();
        // drop tensor buffer_map
        self.tensor_buffer_map = BTreeMap::new();
        // drop graph
        self.graph = Graph::new();
        // drop ir kernel cache
        self.ir_kernel_cache = BTreeMap::new();
        // drop devices
        while let Some(dev) = self.devices.pop() {
            dev.deinitialize()?;
        }
        // drop memory pools
        while let Some(mp) = self.memory_pools.pop() {
            mp.deinitialize()?;
        }
        Ok(())
    }

    #[cfg(feature = "rand")]
    pub(super) fn manual_seed(&mut self, seed: u64) {
        use rand::SeedableRng;
        self.rng = std::cell::OnceCell::from(SmallRng::seed_from_u64(seed));
    }

    /// Creates dot plot of graph between given tensors
    #[must_use]
    pub(super) fn plot_dot_graph(&self, tensors: &BTreeSet<TensorId>) -> String {
        //println!("Tensor storage {:?}", self.tensor_buffer_map);
        self.graph.plot_dot_graph(tensors)
    }

    #[must_use]
    pub(super) fn shape(&self, x: TensorId) -> &[usize] {
        return self.graph.shape(x);
    }

    #[must_use]
    pub(super) fn dtype(&self, x: TensorId) -> DType {
        return self.graph.dtype(x);
    }

    pub(super) fn variable<T: Scalar>(
        &mut self,
        shape: Vec<Dimension>,
        data: &[T],
    ) -> Result<TensorId, ZyxError> {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        let id = self
            .graph
            .push_wshape_and_dtype(Node::Leaf, shape, T::dtype());
        self.initialize_devices()?;
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
        let buffer_id = self.memory_pools[memory_pool_id].allocate(bytes)?;
        self.memory_pools[memory_pool_id].host_to_pool(&data, buffer_id)?;
        self.tensor_buffer_map.insert(
            (id, View::new(self.shape(id))),
            BufferId {
                memory_pool_id,
                buffer_id,
            },
        );
        Ok(id)
    }

    #[must_use]
    pub(super) fn constant(&mut self, value: impl Scalar) -> TensorId {
        self.graph.push(Node::Const {
            value: Constant::new(value),
        })
    }

    // Initialization
    #[must_use]
    pub(super) fn full(
        &mut self,
        shape: Vec<usize>,
        value: impl Scalar,
    ) -> Result<TensorId, ZyxError> {
        let x = self.variable(vec![1], &[value])?;
        let expanded = self.expand(x, shape);
        self.release(x)?;
        return Ok(expanded);
    }

    #[must_use]
    pub(super) fn ones(&mut self, shape: Vec<usize>, dtype: DType) -> TensorId {
        let x = match dtype {
            DType::BF16 => self.constant(bf16::ONE),
            DType::F8 => todo!(),
            DType::F16 => self.constant(f16::ONE),
            DType::F32 => self.constant(1f32),
            DType::F64 => self.constant(1f64),
            #[cfg(feature = "complex")]
            DType::CF32 => self.constant(Complex::new(1f32, 0.)),
            #[cfg(feature = "complex")]
            DType::CF64 => self.constant(Complex::new(1f64, 0.)),
            DType::U8 => self.constant(1u8),
            DType::U32 => self.constant(1u32),
            DType::I8 => self.constant(1i8),
            DType::I16 => self.constant(1i16),
            DType::I32 => self.constant(1i32),
            DType::I64 => self.constant(1i64),
            DType::Bool => self.constant(true),
        };
        let expanded = self.expand(x, shape);
        self.release(x).unwrap();
        return expanded;
    }

    #[must_use]
    pub(super) fn zeros(&mut self, shape: Vec<usize>, dtype: DType) -> TensorId {
        let x = match dtype {
            DType::BF16 => self.constant(bf16::ZERO),
            DType::F8 => todo!(),
            DType::F16 => self.constant(f16::ZERO),
            DType::F32 => self.constant(0f32),
            DType::F64 => self.constant(0f64),
            #[cfg(feature = "complex")]
            DType::CF32 => self.constant(Complex::new(0f32, 0.)),
            #[cfg(feature = "complex")]
            DType::CF64 => self.constant(Complex::new(0f64, 0.)),
            DType::U8 => self.constant(0u8),
            DType::U32 => self.constant(0u32),
            DType::I8 => self.constant(0i8),
            DType::I16 => self.constant(0i16),
            DType::I32 => self.constant(0i32),
            DType::I64 => self.constant(0i64),
            DType::Bool => self.constant(false),
        };
        let expanded = self.expand(x, shape);
        self.release(x).unwrap();
        return expanded;
    }

    // Unary ops
    #[must_use]
    pub(super) fn cast(&mut self, x: TensorId, dtype: DType) -> TensorId {
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

    /// Bitcast self to other type, currently immediatelly realizes the tensor
    #[must_use]
    pub(super) unsafe fn bitcast(
        &mut self,
        x: TensorId,
        dtype: DType,
    ) -> Result<TensorId, ZyxError> {
        if dtype == self.dtype(x) {
            self.retain(x);
            return Ok(x);
        }
        self.realize(BTreeSet::from([x]))?;
        let mut shape = self.shape(x).to_vec();
        let old_k = (x, View::new(&shape));
        // We create a new pointer in tensor_buffer_map to the same buffer
        // and create a new Leaf in graph
        //self.tensor_buffer_map.find();
        let cd = dtype.byte_size() / self.dtype(x).byte_size();
        if let Some(d) = shape.last_mut() {
            if *d % cd != 0 {
                return Err(ZyxError::DTypeError("Can't bitcast due to tensor's last dimension not being correct multiple of dtype.".into()));
            }
            *d = *d / cd;
        }
        let id = self
            .graph
            .push_wshape_and_dtype(Node::Leaf, shape.clone(), dtype);
        if let Some((_, bid)) = self.tensor_buffer_map.iter().find(|(k, _)| *k == &old_k) {
            //println!("Bitcast {x}, res {id}, new shape {shape:?} buffer id {bid:?}");
            self.tensor_buffer_map.insert((id, View::new(&shape)), *bid);
        } else {
            panic!("Tensor sharded across multiple devices can't be currently bitcasted. Internal bug.");
        }
        //println!("TBM:\n{:?}", self.tensor_buffer_map);
        Ok(id)
    }

    #[must_use]
    pub(super) fn reciprocal(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Inv })
    }

    #[must_use]
    pub(super) fn neg(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Neg })
    }

    #[must_use]
    pub(super) fn relu(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::ReLU })
    }

    #[must_use]
    pub(super) fn exp2(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Exp2 })
    }

    #[must_use]
    pub(super) fn log2(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Log2 })
    }

    #[must_use]
    pub(super) fn inv(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Inv })
    }

    #[must_use]
    pub(super) fn sin(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Sin })
    }

    #[must_use]
    pub(super) fn cos(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Cos })
    }

    #[must_use]
    pub(super) fn sqrt(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Sqrt })
    }

    #[must_use]
    pub(super) fn not(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Not })
    }

    #[must_use]
    pub(super) fn reshape(&mut self, x: TensorId, shape: Vec<usize>) -> TensorId {
        //println!("reshaping to {shape:?}, {:?}", self.shape(x));
        let sh = self.shape(x);
        assert_eq!(
            shape.iter().product::<usize>(),
            sh.iter().product::<usize>()
        );
        if &shape == sh {
            self.retain(x);
            return x;
        }
        self.graph.push_wshape(Node::Reshape { x }, shape)
    }

    #[must_use]
    pub(super) fn expand(&mut self, x: TensorId, shape: Vec<usize>) -> TensorId {
        let sh = self.shape(x);
        if &shape == sh {
            self.retain(x);
            return x;
        }
        if shape.len() > sh.len() {
            let sh: Vec<usize> = std::iter::repeat(1)
                .take(shape.len() - sh.len())
                .chain(sh.iter().copied())
                .collect();
            assert_eq!(shape.len(), sh.len());
            let y = self.reshape(x, sh);
            let x = self.graph.push_wshape(Node::Expand { x: y }, shape);
            self.release(y).unwrap();
            return x;
        }
        assert_eq!(shape.len(), sh.len());
        self.graph.push_wshape(Node::Expand { x }, shape)
    }

    #[must_use]
    pub(super) fn permute(&mut self, x: TensorId, axes: Vec<usize>) -> TensorId {
        if axes.len() < 2 || axes == (0..axes.len()).collect::<Vec<usize>>() {
            self.retain(x);
            return x;
        }
        let shape = permute(self.shape(x), &axes);
        self.graph.push_wshape(Node::Permute { x, axes }, shape)
    }

    #[must_use]
    pub(super) fn pad_zeros(&mut self, x: TensorId, padding: Vec<(isize, isize)>) -> TensorId {
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
    pub(super) fn sum_reduce(&mut self, x: TensorId, mut axes: Vec<usize>) -> TensorId {
        let sh = self.shape(x);
        if axes.is_empty() {
            axes = (0..sh.len()).collect();
        };
        let shape = reduce(sh, &axes);
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
    pub(super) fn max_reduce(&mut self, x: TensorId, mut axes: Vec<usize>) -> TensorId {
        let sh = self.shape(x);
        if axes.is_empty() {
            axes = (0..sh.len()).collect();
        };
        let shape = reduce(sh, &axes);
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
    pub(super) fn add(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Add,
        })
    }

    #[must_use]
    pub(super) fn sub(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Sub,
        })
    }

    #[must_use]
    pub(super) fn mul(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Mul,
        })
    }

    #[must_use]
    pub(super) fn div(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Div,
        })
    }

    #[must_use]
    pub(super) fn and(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::And,
        })
    }

    #[must_use]
    pub(super) fn or(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary { x, y, bop: BOp::Or })
    }

    #[must_use]
    pub(super) fn bitor(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::BitOr,
        })
    }

    #[must_use]
    pub(super) fn bitxor(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::BitXor,
        })
    }

    #[must_use]
    pub(super) fn bitand(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::BitAnd,
        })
    }

    #[must_use]
    pub(super) fn pow(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Pow,
        })
    }

    #[must_use]
    pub(super) fn cmplt(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Cmplt,
        })
    }

    #[must_use]
    pub(super) fn cmpgt(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Cmpgt,
        })
    }

    #[must_use]
    pub(super) fn not_eq(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::NotEq,
        })
    }

    #[must_use]
    pub(super) fn maximum(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary {
            x,
            y,
            bop: BOp::Max,
        })
    }
}

impl Runtime {
    pub(super) fn debug_dev(&self) -> bool {
        self.debug % 2 == 1
    }

    fn debug_perf(&self) -> bool {
        (self.debug >> 1) % 2 == 1
    }

    fn debug_sched(&self) -> bool {
        (self.debug >> 2) % 2 == 1
    }

    fn debug_ir(&self) -> bool {
        (self.debug >> 3) % 2 == 1
    }

    fn debug_asm(&self) -> bool {
        (self.debug >> 4) % 2 == 1
    }

    /// Loads data with beginning elements of the tensor x.
    /// If data.len() == x.numel(), then it loads the whole tensor.
    pub(super) fn load<T: Scalar>(&mut self, x: TensorId, data: &mut [T]) -> Result<(), ZyxError> {
        let n: usize = self.shape(x).iter().product();
        assert!(data.len() <= n, "Return buffer is bigger than tensor");
        // Check if tensor is evaluated
        if self.tensor_buffer_map.iter().all(|((id, _), _)| *id != x) {
            self.realize(BTreeSet::from([x]))?;
        }
        // If at least part of tensor exists in some device, there must be
        // the rest of the tensor in other devices
        for ((tensor_id, view), buffer_id) in &self.tensor_buffer_map {
            if *tensor_id == x {
                if view.numel() == n {
                    self.memory_pools[buffer_id.memory_pool_id]
                        .pool_to_host(buffer_id.buffer_id, data)?;
                    break;
                } else {
                    todo!()
                }
            }
        }
        //println!("{data:?}, {}", data.len());
        // for each device where tensor is stored load it
        Ok(())
    }

    pub(super) fn realize(&mut self, tensors: BTreeSet<TensorId>) -> Result<(), ZyxError> {
        //let timer = backend::Timer::new();
        // Runs in O(4n) where n = self.graph.len(),
        // first pass for visited nodes, second pass for outisde_rcs, third pass for order,
        // fourth pass for to_delete and new_leafs
        // Could possibly be optimized a bit
        if tensors.len() == 0 {
            return Ok(());
        }
        if tensors
            .iter()
            .all(|tensor| self.tensor_buffer_map.iter().any(|((t, _), _)| tensor == t))
        {
            return Ok(());
        }
        if self.devices.is_empty() {
            self.initialize_devices()?;
        }
        // Get rcs of nodes outside of realized graph
        let (mut graph, outside_nodes, order) = self.graph.realize_graph(tensors.clone(), |x| {
            self.tensor_buffer_map.iter().any(|((id, _), _)| *id == x)
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
                {
                    if !outside_nodes.contains(tensor) {
                        to_delete.insert(*tensor);
                    } else {
                        graph.to_eval.insert(*tensor);
                        new_leafs.insert(*tensor);
                    }
                } else {
                    for param in self.graph[*tensor].parameters() {
                        if to_delete.contains(&param) {
                            new_leafs.insert(param);
                        }
                    }
                }
            }
        }
        //println!("New leafs: {new_leafs:?}");
        //println!("Realizing {:?}", graph.to_eval);
        // Compile and launch
        if !self.compiled_graph_cache.contains_key(&graph) {
            let compiled_graph = self.compile_graph(graph.clone())?;
            self.compiled_graph_cache
                .insert(graph.clone(), compiled_graph);
        }
        self.launch_graph(&graph)?;
        // Deallocate them from devices
        self.deallocate_tensors(to_delete.clone())?;
        // Remove evaluated part of graph unless needed for backpropagation
        for tensor in new_leafs {
            self.graph.add_shape_dtype(tensor);
            self.graph[tensor] = Node::Leaf;
            to_delete.remove(&tensor);
        }
        //println!("To delete: {to_delete:?}");
        // Delete the node, but do not use release function, just remove it from graph.nodes
        self.graph.delete_tensors(&to_delete);
        return Ok(());
    }

    fn deallocate_tensors(&mut self, to_remove: BTreeSet<TensorId>) -> Result<(), ZyxError> {
        // remove all buffers that are not used by any tensors
        // Check which buffers will possibly need to be dropped
        let mut buffers = BTreeSet::new();
        for ((t, _), b) in self.tensor_buffer_map.iter() {
            if to_remove.contains(t) {
                buffers.insert(*b);
            }
        }
        // Remove unnedded tensors from the map
        self.tensor_buffer_map
            .retain(|(t, _), _| !to_remove.contains(t));
        // Check if buffers are needed elsewhere in the map,
        // otherwise deallocate them
        for buffer in buffers {
            if self.tensor_buffer_map.values().all(|b| *b != buffer) {
                self.memory_pools[buffer.memory_pool_id].deallocate(buffer.buffer_id)?;
            }
        }
        return Ok(());
    }

    pub(super) fn backward(
        &mut self,
        x: TensorId,
        sources: BTreeSet<TensorId>,
    ) -> BTreeMap<TensorId, TensorId> {
        // Does not allocate new tensors, only constant and op nodes
        let topo = self.graph.build_topo(x, &sources);
        //println!("Topo: {topo:?}");

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
        //println!("{:?}", self.nodes.last().unwrap());

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
                    BOp::Cmplt => {
                        panic!("Cmplt is not a differentiable operation.");
                    }
                    BOp::Cmpgt => {
                        panic!("Cmpgt is not a differentiable operation.");
                    }
                    BOp::NotEq => {
                        panic!("NotEq is not a differentiable operation.");
                    }
                    BOp::Max => {
                        todo!("Max backward.");
                    }
                    BOp::Or => {
                        todo!("Or backward.");
                    }
                    BOp::And => {
                        todo!("And backward.");
                    }
                    BOp::BitAnd => {
                        todo!("BitAnd backward.");
                    }
                    BOp::BitOr => {
                        todo!("BitOr backward.");
                    }
                    BOp::BitXor => {
                        todo!("BitXor backward.");
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
                        let temp = self.constant(std::f64::consts::E.log2());
                        let temp1 = self.expand(temp, self.shape(x).into());
                        self.release(temp).unwrap();
                        let temp2 = self.mul(nid, temp1);
                        self.release(temp1).unwrap();
                        let grad = self.mul(nid, temp2);
                        self.release(temp2).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Log2 => {
                        let temp = self.constant(std::f64::consts::E.log2());
                        let temp1 = self.expand(temp, self.shape(x).into());
                        self.release(temp).unwrap();
                        let temp2 = self.mul(x, temp1);
                        self.release(temp1).unwrap();
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

/// Enumeration representing the various errors that can occur within the Zyx library.
#[derive(Debug)]
pub enum ZyxError {
    /// Invalid shapes for operation
    ShapeError(String),
    /// Wrong dtype for given operation
    DTypeError(String),
    /// Backend configuration error
    BackendConfig(&'static str),
    /// Error from file operations
    IOError(std::io::Error),
    /// Error parsing some data
    ParseError(String),
    /// Memory allocation error
    AllocationError,
    /// There are no available backends
    NoBackendAvailable,
    /// Error returned by the CUDA driver
    CUDAError(CUDAError),
    /// Error returned by the HIP runtime
    HIPError(HIPError),
    /// Error returned by the OpenCL runtime
    OpenCLError(OpenCLError),
    /// Error returned by the Vulkan runtime
    VulkanError(VulkanError),
    /// This error is only applicable when the `wgsl` feature is enabled.
    #[cfg(feature = "wgsl")]
    WGSLError(WGSLError),
}

/*impl<Err: std::fmt::Display> From<Err> for ZyxError {
    #[track_caller]
    fn from(err: Err) -> Self {
        panic!("error: {}: {}", std::any::type_name::<Err>(), err);
    }
}*/

impl std::fmt::Display for ZyxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZyxError::ShapeError(e) => f.write_str(&e),
            ZyxError::DTypeError(e) => f.write_fmt(format_args!("Wrong dtype {e:?}")),
            ZyxError::IOError(e) => f.write_fmt(format_args!("IO {e}")),
            ZyxError::ParseError(e) => f.write_fmt(format_args!("IO {e}")),
            ZyxError::BackendConfig(e) => f.write_fmt(format_args!("Backend config {e:?}'")),
            ZyxError::NoBackendAvailable => f.write_fmt(format_args!("No available backend")),
            ZyxError::AllocationError => f.write_fmt(format_args!("Allocation error")),
            ZyxError::CUDAError(e) => f.write_fmt(format_args!("CUDA {e:?}")),
            ZyxError::HIPError(e) => f.write_fmt(format_args!("HIP {e:?}")),
            ZyxError::OpenCLError(e) => f.write_fmt(format_args!("OpenCL {e:?}")),
            ZyxError::VulkanError(e) => f.write_fmt(format_args!("Vulkan {e:?}")),
            #[cfg(feature = "wgsl")]
            ZyxError::WGSLError(_) => todo!(),
        }
    }
}

impl std::error::Error for ZyxError {}

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

impl From<VulkanError> for ZyxError {
    fn from(value: VulkanError) -> Self {
        ZyxError::VulkanError(value)
    }
}

#[cfg(feature = "wgsl")]
impl From<WGSLError> for ZyxError {
    fn from(value: WGSLError) -> Self {
        ZyxError::WGSLError(value)
    }
}

impl From<std::io::Error> for ZyxError {
    fn from(value: std::io::Error) -> Self {
        Self::IOError(value)
    }
}
