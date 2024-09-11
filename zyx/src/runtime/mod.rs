use crate::dtype::{Constant, DType};
use crate::scalar::Scalar;
use crate::shape::Dimension;
use crate::tensor::TensorId;
use backend::{
    BufferId, CUDAConfig, CUDAError, Device, DeviceId, HIPConfig, HIPError, MemoryPool,
    OpenCLConfig, OpenCLError,
};
#[cfg(feature = "wgsl")]
use backend::{WGSLConfig, WGSLError};
use graph::Graph;
use ir::IRKernel;
use node::{BOp, Node, ROp, UOp};
use scheduler::CompiledGraph;
use std::path::PathBuf;
use std::{
    collections::{btree_map::Entry, BTreeMap, BTreeSet},
    vec,
    vec::Vec,
};
use view::View;

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

#[cfg_attr(feature = "py", pyo3::pyclass)]
#[derive(serde::Deserialize, Debug, Default)]
pub struct DeviceConfig {
    pub cuda: CUDAConfig,
    pub hip: HIPConfig,
    pub opencl: OpenCLConfig,
    #[cfg(feature = "wgsl")]
    pub wgsl: WGSLConfig,
}

// This is the whole global state of zyx
pub(super) struct Runtime {
    // Current graph of tensor operations as nodes
    graph: Graph,
    // Random number generator
    #[cfg(feature = "rand")]
    rng: std::cell::OnceCell<SmallRng>,
    // Cache for compiled graphs
    compiled_graph_cache: BTreeMap<Graph, CompiledGraph>,
    memory_pools: Vec<MemoryPool>,
    // Where are tensors stored
    tensor_buffer_map: BTreeMap<(TensorId, View), BufferId>,
    devices: Vec<Device>,
    // Cache which maps IRKernel to device and program id on the device
    ir_kernel_cache: BTreeMap<IRKernel, (DeviceId, usize)>,
    config_dir: Option<PathBuf>, // Why the is hell PathBuf::new not const???????
    // Are we in training mode?
    pub(super) training: bool,
    pub(super) search_iterations: u32,
    pub(super) debug: u32,
}

impl Runtime {
    #[must_use]
    pub(super) const fn new() -> Self {
        Runtime {
            graph: Graph::new(),
            #[cfg(feature = "rand")]
            rng: core::cell::OnceCell::new(),
            compiled_graph_cache: BTreeMap::new(),
            memory_pools: Vec::new(),
            tensor_buffer_map: BTreeMap::new(),
            devices: Vec::new(),
            ir_kernel_cache: BTreeMap::new(),
            config_dir: None,
            training: false,
            search_iterations: 100,
            debug: 0,
        }
    }

    pub(super) fn retain(&mut self, x: TensorId) {
        self.graph.retain(x);
    }

    pub(super) fn release(&mut self, x: TensorId) -> Result<(), ZyxError> {
        let to_remove = self.graph.release(x);
        self.deallocate_tensors(to_remove)
    }

    #[cfg(feature = "rand")]
    pub(super) fn manual_seed(&mut self, seed: u64) {
        use rand::SeedableRng;
        self.rng = std::cell::OnceCell::from(SmallRng::seed_from_u64(seed));
    }

    /// Creates dot plot of graph between given tensors
    #[must_use]
    pub(super) fn plot_dot_graph(&self, tensors: &BTreeSet<TensorId>) -> String {
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

    #[cfg(feature = "rand")]
    #[must_use]
    pub(super) fn uniform<T: Scalar>(
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

    pub(super) fn temp<T: Scalar>(
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
    }

    // Initialization
    #[must_use]
    pub(super) fn full(&mut self, shape: Vec<usize>, value: impl Scalar) -> TensorId {
        let one = self.graph.push(Node::Const {
            value: Constant::new(value),
        });
        let expanded = self.expand(one, shape);
        self.release(one).unwrap();
        return expanded;
    }

    #[must_use]
    pub(super) fn ones(&mut self, shape: Vec<usize>, dtype: DType) -> TensorId {
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
    pub(super) fn zeros(&mut self, shape: Vec<usize>, dtype: DType) -> TensorId {
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

    #[must_use]
    pub(super) fn reciprocal(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Inv });
    }

    #[must_use]
    pub(super) fn neg(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Neg });
    }

    #[must_use]
    pub(super) fn relu(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::ReLU });
    }

    #[must_use]
    pub(super) fn exp2(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Exp2 });
    }

    #[must_use]
    pub(super) fn log2(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Log2 });
    }

    #[must_use]
    pub(super) fn inv(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Inv });
    }

    #[must_use]
    pub(super) fn sin(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Sin });
    }

    #[must_use]
    pub(super) fn cos(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Cos });
    }

    #[must_use]
    pub(super) fn sqrt(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Sqrt });
    }

    #[must_use]
    pub(super) fn nonzero(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary {
            x,
            uop: UOp::Nonzero,
        });
    }

    #[must_use]
    pub(super) fn not(&mut self, x: TensorId) -> TensorId {
        return self.graph.push(Node::Unary { x, uop: UOp::Not });
    }

    #[must_use]
    pub(super) fn reshape(&mut self, x: TensorId, shape: Vec<usize>) -> TensorId {
        if &shape == self.shape(x) {
            self.retain(x);
            return x;
        }
        self.graph.push_wshape(Node::Reshape { x }, shape)
    }

    #[must_use]
    pub(super) fn expand(&mut self, x: TensorId, shape: Vec<usize>) -> TensorId {
        if &shape == self.shape(x) {
            self.retain(x);
            return x;
        }
        self.graph.push_wshape(Node::Expand { x }, shape)
    }

    #[must_use]
    pub(super) fn permute(&mut self, x: TensorId, axes: Vec<usize>) -> TensorId {
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
    pub(super) fn sum_reduce(&mut self, x: TensorId, axes: Vec<usize>) -> TensorId {
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
    pub(super) fn max_reduce(&mut self, x: TensorId, axes: Vec<usize>) -> TensorId {
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

    pub(super) fn load<T: Scalar>(&mut self, x: TensorId) -> Result<Vec<T>, ZyxError> {
        // Check if tensor is evaluated
        if self.tensor_buffer_map.iter().all(|((id, _), _)| *id != x) {
            self.realize(BTreeSet::from([x]))?;
        }
        // If at least part of tensor exists in some device, there must be
        // the rest of the tensor in other devices
        let n: usize = self.shape(x).iter().product();
        let mut data: Vec<T> = Vec::with_capacity(n);
        unsafe { data.set_len(n) };
        for ((tensor_id, view), buffer_id) in &self.tensor_buffer_map {
            if *tensor_id == x {
                if view.numel() == n {
                    self.memory_pools[buffer_id.memory_pool_id]
                        .pool_to_host(buffer_id.buffer_id, &mut data)?;
                    break;
                } else {
                    todo!()
                }
            }
        }
        // for each device where tensor is stored load it
        Ok(data)
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
        let (graph, outside_nodes, order) = self.graph.realize_graph(tensors.clone(), |x| {
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
            let compiled_graph = self.compile_graph(graph.clone(), &tensors)?;
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
        let mut buffers: Vec<BufferId> = Vec::new();
        for tensor in to_remove {
            for (_, buffer_id) in self
                .tensor_buffer_map
                .iter()
                .filter(|((t, _), _)| *t == tensor)
            {
                buffers.push(*buffer_id);
            }
        }
        self.tensor_buffer_map.retain(|_, b| !buffers.contains(b));
        for buffer in buffers {
            self.memory_pools[buffer.memory_pool_id].deallocate(buffer.buffer_id)?;
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

/// Enumeration representing the various errors that can occur within the Zyx library.
#[derive(Debug)]
pub enum ZyxError {
    /// Error indicating an empty tensor.
    EmptyTensor,
    /// Backend configuration error
    BackendConfig(&'static str),
    /// Wrong dtype for given operation
    WrongDType(&'static str),
    /// There are no available backends
    NoBackendAvailable,
    /// Memory allocation error
    AllocationError,
    /// Error returned by the CUDA driver
    CUDAError(CUDAError),
    /// Error returned by the HIP runtime
    HIPError(HIPError),
    /// Error returned by the OpenCL runtime
    OpenCLError(OpenCLError),
    /// This error is only applicable when the `wgsl` feature is enabled.
    #[cfg(feature = "wgsl")]
    WGSLError(WGSLError),
    /// Error from file operations
    IOError(std::io::Error),
    /// Error parsing some data
    ParseError(String),
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

impl std::fmt::Display for ZyxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZyxError::EmptyTensor => f.write_str("Empty tensor"),
            ZyxError::BackendConfig(e) => f.write_fmt(format_args!("Backend config {e:?}'")),
            ZyxError::WrongDType(e) => f.write_fmt(format_args!("Wrong dtype {e:?}")),
            ZyxError::NoBackendAvailable => f.write_fmt(format_args!("No available backend")),
            ZyxError::AllocationError => f.write_fmt(format_args!("Allocation error")),
            ZyxError::CUDAError(e) => f.write_fmt(format_args!("CUDA {e:?}")),
            ZyxError::HIPError(e) => f.write_fmt(format_args!("HIP {e:?}")),
            ZyxError::OpenCLError(e) => f.write_fmt(format_args!("OpenCL {e:?}")),
            ZyxError::IOError(e) => f.write_fmt(format_args!("IO {e}")),
            ZyxError::ParseError(e) => f.write_fmt(format_args!("IO {e}")),
            #[cfg(feature = "wgsl")]
            ZyxError::WGSLError(_) => todo!(),
        }
    }
}

impl std::error::Error for ZyxError {}
