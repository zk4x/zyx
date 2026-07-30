// ----- ASYNC EVENT RULES -----
//
// host_to_pool is async: the host-side source buffer must stay valid
// until the returned event is synced via sync_events.
//
// Kernel launch is async: ALL buffers used by the kernel (both loads
// and stores) must stay valid until the kernel's event is consumed.
//
// Events are tracked in self.events: Map<BTreeSet<BufferId>, Event>.
// The key set must include every buffer the kernel touches, so future
// operations can find and wait on the event before reusing the buffer.
//
// When extracting a pending event for a buffer: iterate events.keys(),
// find the set containing the buffer, remove the event, and add it to
// the wait list passed to the next launch. A removed event is consumed.
// -----------------------------

use std::{
    collections::BTreeSet,
    env,
    hash::BuildHasherDefault,
    path::{Path, PathBuf},
};

use nanoserde::DeJson;

use crate::{
    DType, DebugMask, Map, Scalar, Set, ZyxError,
    backend::{
        AutotuneConfig, BufferId, Config, DTypeCapability, Device, DeviceInfo, DeviceProgramId, Event, MemoryPool, PoolBufferId,
        PoolId, ProgramId,
    },
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    graph::ExecPlan,
    graph::{ClassId, Graph, GraphId, Node},
    kernel::{BOp, DeviceId, Kernel, MemScope, MoveOp, Op, OpId, UOp, autotune::OptSeq},
    rng::Rng,
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    tensor::TensorId,
    view::View,
};

#[derive(Debug, Copy, Clone, Hash, PartialEq, PartialOrd, Eq, Ord)]
pub struct ShapeId(u16);

impl From<usize> for ShapeId {
    fn from(value: usize) -> Self {
        ShapeId(value as u16)
    }
}

impl From<ShapeId> for usize {
    fn from(value: ShapeId) -> Self {
        value.0 as usize
    }
}

impl SlabId for ShapeId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u16::MAX);
    fn inc(&mut self) {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub(crate) struct DeviceInfoId(u32);

impl From<usize> for DeviceInfoId {
    fn from(value: usize) -> Self {
        DeviceInfoId(value as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub(crate) struct KernelId(u16);

impl From<usize> for KernelId {
    fn from(value: usize) -> Self {
        KernelId(value as u16)
    }
}

impl From<KernelId> for usize {
    fn from(value: KernelId) -> Self {
        value.0 as usize
    }
}

impl SlabId for KernelId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u16::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug)]
pub struct TensorData {
    pub shape_id: ShapeId,
    pub dtype: DType,
    pub state: TensorState,
}

#[derive(Debug)]
pub enum TensorState {
    Eager {
        kernel_id: KernelId,
        op_id: OpId,
        // Pending store has only one purpose - for runtime.release
        // This marks that a tensor is in stores of some kernel somewhere, just not realized yet
        // Because release removes from outputs. But the kernel that has this in outputs
        // is not the one that stores it, it's the one that loads it. So pending_stores just marks
        // there is a kernel that has this in the stores despite not being realized yet.
        pending: KernelId,
    },
    Graph {
        class_id: ClassId,
        rc: u32,
        graph_id: GraphId,
    },
}

#[derive(Debug)]
pub(crate) struct KernelData {
    /// Tensor reference count. Each entry is a tensor this kernel must produce.
    /// When a tensor is consumed as input to a new op within the same kernel,
    /// it is removed from outputs (since the kernel produces the new op's result instead).
    pub outputs: Vec<TensorId>,
    pub loads: Vec<TensorId>,
    pub stores: Vec<TensorId>,
    pub kernel: Kernel,
}

pub struct Runtime {
    pub graphs: Slab<GraphId, Graph>,
    shape_map: Map<Vec<Dim>, ShapeId>,
    pub shapes: Slab<ShapeId, Vec<Dim>>,
    pub tensors: Slab<TensorId, TensorData>,
    pub kernels: Slab<KernelId, KernelData>,
    kernel_map: Map<Kernel, KernelId>,
    optimizations: Map<(KernelId, DeviceInfoId), OptSeq>,
    device_infos: Map<DeviceInfo, DeviceInfoId>,
    programs: Map<KernelId, DeviceProgramId>,
    timings: Map<ProgramId, u64>,
    pub devices: Slab<DeviceId, Device>,
    // Pool 0 is always host, pool 1 is disk if disk is present
    pub pools: Slab<PoolId, MemoryPool>,
    config_dir: Option<PathBuf>,
    pub buffer_map: Map<TensorId, BufferId>,
    pub events: Map<BTreeSet<BufferId>, Event>,
    pub rng: Rng,
    autotune_config: AutotuneConfig,
    pub implicit_casts: bool,
    pub training: bool,
    pub debug: DebugMask,
    pub plan_cache: Map<u64, ExecPlan>,
}

impl Runtime {
    pub const fn new() -> Self {
        Runtime {
            graphs: Slab::new(),
            shape_map: Map::with_hasher(BuildHasherDefault::new()),
            shapes: Slab::new(),
            tensors: Slab::new(),
            kernels: Slab::new(),
            kernel_map: Map::with_hasher(BuildHasherDefault::new()),
            device_infos: Map::with_hasher(BuildHasherDefault::new()),
            devices: Slab::new(),
            pools: Slab::new(),
            programs: Map::with_hasher(BuildHasherDefault::new()),
            timings: Map::with_hasher(BuildHasherDefault::new()),
            config_dir: None,
            optimizations: Map::with_hasher(BuildHasherDefault::new()),
            buffer_map: Map::with_hasher(BuildHasherDefault::new()),
            events: Map::with_hasher(BuildHasherDefault::new()),
            rng: Rng::seed_from_u64(42069),
            autotune_config: AutotuneConfig::new(),
            implicit_casts: true,
            training: false,
            debug: DebugMask::new(0),
            plan_cache: Map::with_hasher(BuildHasherDefault::new()),
        }
    }

    pub fn shape(&self, x: TensorId) -> &[Dim] {
        &self.shapes[self.tensors[x].shape_id]
    }

    pub fn dtype(&self, x: TensorId) -> DType {
        self.tensors[x].dtype
    }

    pub fn is_realized(&self, x: TensorId) -> bool {
        self.buffer_map.contains_key(&x)
    }

    /// Returns operation capabilities for a dtype across all devices.
    pub fn supports_dtype(&mut self, dtype: DType) -> DTypeCapability {
        self.initialize_devices().expect("initialize_devices");
        let mut caps = DTypeCapability::none();
        for (_id, dev) in self.devices.iter() {
            caps = caps.include(dev.info().supports_dtype(dtype));
        }
        caps
    }

    pub fn retain(&mut self, x: TensorId) {
        //eprintln!("Retain tensor x={x}");
        match &mut self.tensors[x].state {
            TensorState::Eager { kernel_id, .. } => {
                self.kernels[*kernel_id].outputs.push(x);
            }
            TensorState::Graph { rc, .. } => {
                *rc += 1;
            }
        }
    }

    pub fn release(&mut self, x: TensorId) {
        let (kid, op_id, pending) = match &mut self.tensors[x].state {
            TensorState::Eager { kernel_id, op_id, pending: pending_store, .. } => (*kernel_id, *op_id, *pending_store),
            TensorState::Graph { rc, graph_id, .. } => {
                *rc -= 1;
                if *rc == 0 && !self.graphs.contains_key(*graph_id) {
                    assert!(!self.buffer_map.contains_key(&x));
                    self.tensors.remove(x);
                }
                return;
            }
        };
        let kd = &mut self.kernels[kid];
        kd.outputs.iter().position(|e| *e == x).map(|i| kd.outputs.remove(i));
        if !kd.outputs.contains(&x) && !self.buffer_map.contains_key(&x) && pending.is_null() {
            self.tensors.remove(x);
        }
        if !kd.outputs.contains(&x) {
            let out_ops: Vec<OpId> = kd
                .outputs
                .iter()
                .map(|&tid| match self.tensors[tid].state {
                    TensorState::Eager { op_id, .. } => op_id,
                    _ => unreachable!(),
                })
                .collect();
            let op_removed = kd.kernel.remove_unused_chain(op_id, &out_ops);
            if op_removed {
                kd.loads.retain(|&tid| tid != x);
            }
        }
        if kd.outputs.is_empty() {
            if !kd.kernel.contains_stores() {
                //eprintln!("A: kernels.remove({kid:?})");
                self.kernels.remove(kid);
            } else {
                self.materialize_kernel(kid).unwrap();
            }
        }
    }

    pub fn push_shape(&mut self, shape: Vec<Dim>) -> ShapeId {
        if let Some(&shape_id) = self.shape_map.get(&shape) {
            shape_id
        } else {
            let shape_id = self.shapes.push(shape.clone());
            self.shape_map.insert(shape, shape_id);
            shape_id
        }
    }

    fn new_kernel(&mut self, op: Op, shape: Vec<Dim>, dtype: DType) -> TensorId {
        let shape_id = self.push_shape(shape);
        let mut kernel = Kernel::new(DeviceId::AUTO);
        let op_id = kernel.push_back(op);
        let kernel_id = self.kernels.push(KernelData { outputs: Vec::new(), loads: Vec::new(), stores: Vec::new(), kernel });
        let tid = self.tensors.push(TensorData {
            shape_id,
            dtype,
            state: TensorState::Eager { kernel_id, op_id, pending: KernelId::NULL },
        });
        self.kernels[kernel_id].outputs.push(tid);
        tid
    }

    pub fn new_constant_tensor(&mut self, value: Constant) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::new_constant_tensor(value={value:?})");
        let dtype = value.dtype();
        let op = Op::ConstView(Box::new((value, View::contiguous(&[1]))));
        let result = self.new_kernel(op, [1].into(), dtype);
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={result}, {:?}", self.tensors[result]);
        result
    }

    pub fn new_full(&mut self, shape: Vec<Dim>, value: Constant) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::new_full(shape={shape:?}, value={value:?})");
        let dtype = value.dtype();
        let op = Op::ConstView(Box::new((value, View::contiguous(&[1]))));
        let x = self.new_kernel(op, [1].into(), dtype);
        let expanded = self.expand(x, shape).unwrap();
        self.release(x);
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={expanded}, {:?}", self.tensors[expanded]);
        expanded
    }

    // Creates new tensor in host memory
    pub fn new_host_tensor<T: Scalar>(&mut self, shape: Vec<Dim>, data: Box<[T]>) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::new_host_tensor(shape={shape:?})");
        let dtype = T::dtype();

        self.initialize_devices()?;
        debug_assert_eq!(shape.iter().product::<Dim>(), data.len() as Dim);
        let bytes = (data.len() * dtype.bit_size() as usize + 7) / 8;
        debug_assert_eq!(data.len() * std::mem::size_of::<T>(), bytes as usize);

        // Convert to Box<[u8]>
        let ptr = (Box::into_raw(data) as *mut T) as *mut u8;
        let slice = std::ptr::slice_from_raw_parts_mut(ptr, bytes as usize);
        let data = unsafe { Box::from_raw(slice) };

        // Store to Host memory
        let MemoryPool::Host(ref mut pool) = self.pools[PoolId::HOST] else {
            unreachable!("Host must exist.")
        };
        let buffer_id = BufferId { pool: PoolId::HOST, buffer: pool.insert(data) };

        let shape = self.push_shape(shape);
        let op = Op::LoadView(Box::new((dtype, View::contiguous(&self.shapes[shape]))));
        let tid = self.new_kernel(op, self.shapes[shape].clone(), dtype);
        let TensorState::Eager { kernel_id, .. } = &self.tensors[tid].state else {
            unreachable!()
        };
        self.kernels[*kernel_id].loads.push(tid);

        self.buffer_map.insert(tid, buffer_id);

        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, shape={:?} dtype={}", self.shape(tid), self.dtype(tid));
        Ok(tid)
    }

    // Creates new tensor in disk
    pub fn new_disk_tensor(
        &mut self,
        shape: Vec<Dim>,
        dtype: DType,
        path: &Path,
        offset_bytes: u64,
    ) -> Result<TensorId, ZyxError> {
        self.initialize_devices()?;
        let bytes: Dim = (shape.iter().product::<Dim>() * dtype.bit_size() as Dim + 7) / 8;

        let pool = self.pools[PoolId::DISK]
            .disk_pool()
            .ok_or(BackendError { status: ErrorStatus::Initialization, context: "[disk] not available.".into() })?;
        let buffer_id = BufferId { pool: PoolId::DISK, buffer: pool.buffer_from_path(bytes, path, offset_bytes) };

        let shape_id = self.push_shape(shape);
        let op = Op::LoadView(Box::new((dtype, View::contiguous(&self.shapes[shape_id]))));
        let tid = self.new_kernel(op, self.shapes[shape_id].clone(), dtype);
        let TensorState::Eager { kernel_id, .. } = &self.tensors[tid].state else {
            unreachable!()
        };
        self.kernels[*kernel_id].loads.push(tid);
        self.buffer_map.insert(tid, buffer_id);
        Ok(tid)
    }

    pub(crate) fn promote_to_graph(&mut self, tid: TensorId, graph_id: GraphId) -> Result<ClassId, ZyxError> {
        if let TensorState::Graph { class_id, .. } = self.tensors[tid].state {
            return Ok(class_id);
        }

        let (kernel_id, my_op_id) = match self.tensors[tid].state {
            TensorState::Eager { kernel_id, op_id, .. } => (kernel_id, op_id),
            _ => unreachable!(),
        };
        debug_assert!(self.kernels[kernel_id].outputs.contains(&tid));

        let relevant = {
            let kernel = &self.kernels[kernel_id].kernel;
            let mut relevant: Set<OpId> = Set::default();
            let mut stack = vec![my_op_id];
            while let Some(oid) = stack.pop() {
                if !relevant.insert(oid) {
                    continue;
                }
                match kernel.at(oid) {
                    Op::LoadView(_) | Op::ConstView(_) => {}
                    Op::Unary { x, .. } => stack.push(*x),
                    Op::Binary { x, y, .. } => {
                        stack.push(*x);
                        stack.push(*y);
                    }
                    Op::Cast { x, .. } => stack.push(*x),
                    Op::Reduce { x, .. } => stack.push(*x),
                    Op::Move { x, .. } => stack.push(*x),
                    _ => unreachable!(),
                }
            }
            relevant
        };

        let loads = self.kernels[kernel_id].loads.clone();
        let mut op_to_class: Map<OpId, ClassId> = Map::default();
        let mut load_idx = 0;
        let mut op_id = self.kernels[kernel_id].kernel.head;
        while !op_id.is_null() {
            if relevant.contains(&op_id) {
                let op = self.kernels[kernel_id].kernel.at(op_id).clone();
                let class_id = match &op {
                    Op::LoadView(_) => {
                        let load_tid = loads[load_idx];
                        if !self.buffer_map.contains_key(&load_tid) {
                            let pending = match &self.tensors[load_tid].state {
                                TensorState::Eager { pending, .. } => *pending,
                                TensorState::Graph { .. } => KernelId::NULL,
                            };
                            debug_assert!(!pending.is_null());
                            let outputs: Vec<TensorId> = self.kernels[pending].outputs.clone();
                            for &otid in &outputs {
                                self.add_store(otid)?;
                            }
                        }
                        let shape_id = self.tensors[load_tid].shape_id;
                        let dtype = self.tensors[load_tid].dtype;
                        let (_, class_id) = self.graphs[graph_id].push_leaf(dtype, shape_id);
                        self.graphs[graph_id].leaf_map.insert(class_id, load_tid);
                        class_id
                    }
                    Op::ConstView(x) => {
                        let shape = x.1.shape();
                        let shape_id = self.push_shape(shape);
                        let (_, class_id) = self.graphs[graph_id].push(Node::Const(x.0), shape_id, x.0.dtype());
                        class_id
                    }
                    Op::Unary { x, uop } => {
                        let x_class = op_to_class[x];
                        let shape = self.graphs[graph_id].classes[x_class].shape;
                        let dtype = self.graphs[graph_id].classes[x_class].dtype;
                        let (_, class_id) = self.graphs[graph_id].push(Node::Unary { x: x_class, uop: *uop }, shape, dtype);
                        class_id
                    }
                    Op::Binary { x, y, bop } => {
                        let x_class = op_to_class[x];
                        let shape = self.graphs[graph_id].classes[x_class].shape;
                        let dtype = self.graphs[graph_id].classes[x_class].dtype;
                        let y_class = op_to_class[y];
                        let (_, class_id) =
                            self.graphs[graph_id].push(Node::Binary { x: x_class, y: y_class, bop: *bop }, shape, dtype);
                        class_id
                    }
                    Op::Cast { x, dtype } => {
                        let x_class = op_to_class[x];
                        let shape = self.graphs[graph_id].classes[x_class].shape;
                        let (_, class_id) = self.graphs[graph_id].push(Node::Cast { x: x_class, dtype: *dtype }, shape, *dtype);
                        class_id
                    }
                    Op::Reduce { x, rop, n_axes } => {
                        let x_class = op_to_class[x];
                        let in_shape = self.shapes[self.graphs[graph_id].classes[x_class].shape].clone();
                        let out_shape: Vec<Dim> = in_shape[..in_shape.len() - *n_axes as usize].to_vec();
                        let out_shape_id = self.push_shape(out_shape);
                        let dtype = self.graphs[graph_id].classes[x_class].dtype;
                        let axes: Vec<UAxis> = (in_shape.len() - *n_axes as usize..in_shape.len()).collect();
                        let (_, class_id) = self.graphs[graph_id].push(
                            Node::Reduce { x: x_class, bop: *rop, axes: axes.into() },
                            out_shape_id,
                            dtype,
                        );
                        class_id
                    }
                    Op::Move { x, mop } => match mop.as_ref() {
                        MoveOp::Reshape { shape } => {
                            let x_class = op_to_class[x];
                            let dtype = self.graphs[graph_id].classes[x_class].dtype;
                            let shape_id = self.push_shape(shape.clone());
                            let (_, class_id) =
                                self.graphs[graph_id].push(Node::Reshape { x: x_class, shape: shape_id }, shape_id, dtype);
                            class_id
                        }
                        MoveOp::Expand { shape } => {
                            let x_class = op_to_class[x];
                            let dtype = self.graphs[graph_id].classes[x_class].dtype;
                            let shape_id = self.push_shape(shape.clone());
                            let (_, class_id) =
                                self.graphs[graph_id].push(Node::Expand { x: x_class, shape: shape_id }, shape_id, dtype);
                            class_id
                        }
                        MoveOp::Permute { axes, shape } => {
                            let x_class = op_to_class[x];
                            let dtype = self.graphs[graph_id].classes[x_class].dtype;
                            let shape_id = self.push_shape(shape.clone());
                            let (_, class_id) = self.graphs[graph_id].push(
                                Node::Permute { x: x_class, axes: axes.clone().into() },
                                shape_id,
                                dtype,
                            );
                            class_id
                        }
                        MoveOp::Pad { padding, shape } => {
                            let x_class = op_to_class[x];
                            let dtype = self.graphs[graph_id].classes[x_class].dtype;
                            let shape_id = self.push_shape(shape.clone());
                            let (_, class_id) = self.graphs[graph_id].push(
                                Node::PadZeros { x: x_class, padding: padding.clone().into() },
                                shape_id,
                                dtype,
                            );
                            class_id
                        }
                    },
                    _ => unreachable!(),
                };
                op_to_class.insert(op_id, class_id);
            }

            if matches!(self.kernels[kernel_id].kernel.at(op_id), Op::LoadView(_)) {
                load_idx += 1;
            }
            op_id = self.kernels[kernel_id].kernel.next_op(op_id);
        }

        let class_id = op_to_class[&my_op_id];
        self.tensors[tid].state = TensorState::Graph { class_id, rc: 1, graph_id };
        Ok(class_id)
    }

    pub fn cast(&mut self, x: TensorId, dtype: DType) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::cast(x={x}, dtype={dtype:?})");
        let shape_id = self.tensors[x].shape_id;
        match self.tensors[x].state {
            TensorState::Graph { class_id, graph_id, .. } => {
                let (_, class_id) = self.graphs[graph_id].push(Node::Cast { x: class_id, dtype }, shape_id, dtype);
                let tid =
                    self.tensors.push(TensorData { shape_id, dtype, state: TensorState::Graph { class_id, rc: 1, graph_id } });
                tid
            }
            TensorState::Eager { kernel_id, op_id, .. } => {
                let op_id = self.kernels[kernel_id].kernel.cast(op_id, dtype);
                let tid = self.tensors.push(TensorData {
                    shape_id,
                    dtype,
                    state: TensorState::Eager { kernel_id, op_id, pending: KernelId::NULL },
                });
                self.kernels[kernel_id].outputs.push(tid);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
        }
    }

    pub fn bitcast(&mut self, x: TensorId, dtype: DType) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::bitcast(x={x}, dtype={dtype:?})");
        let shape_id = self.tensors[x].shape_id;
        match self.tensors[x].state {
            TensorState::Graph { class_id, graph_id, .. } => {
                let (_, class_id) = self.graphs[graph_id].push(Node::Cast { x: class_id, dtype }, shape_id, dtype);
                let tid =
                    self.tensors.push(TensorData { shape_id, dtype, state: TensorState::Graph { class_id, rc: 1, graph_id } });
                tid
            }
            TensorState::Eager { kernel_id, op_id, .. } => {
                let op_id = self.kernels[kernel_id].kernel.bitcast(op_id, dtype);
                let tid = self.tensors.push(TensorData {
                    shape_id,
                    dtype,
                    state: TensorState::Eager { kernel_id, op_id, pending: KernelId::NULL },
                });
                self.kernels[kernel_id].outputs.push(tid);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
        }
    }

    pub fn unary(&mut self, x: TensorId, uop: UOp) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::unary(x={x}, uop={uop:?})");
        let shape_id = self.tensors[x].shape_id;
        let dtype = self.tensors[x].dtype;
        match self.tensors[x].state {
            TensorState::Graph { class_id, graph_id, .. } => {
                let (_node_id, class_id) = self.graphs[graph_id].push(Node::Unary { x: class_id, uop }, shape_id, dtype);
                let tid =
                    self.tensors.push(TensorData { shape_id, dtype, state: TensorState::Graph { class_id, rc: 1, graph_id } });
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> tid={tid}, nid={_node_id:?}, cid={class_id:?}");
                tid
            }
            TensorState::Eager { kernel_id, op_id, .. } => {
                let op_id = self.kernels[kernel_id].kernel.unary(op_id, uop);
                let tid = self.tensors.push(TensorData {
                    shape_id,
                    dtype,
                    state: TensorState::Eager { kernel_id, op_id, pending: KernelId::NULL },
                });
                self.kernels[kernel_id].outputs.push(tid);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
        }
    }

    pub fn binary(&mut self, x: TensorId, y: TensorId, bop: BOp) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::binary(x={x}, y={y}, bop={bop:?})");
        let shape_id = self.tensors[x].shape_id;
        let dtype = if bop.returns_bool() {
            DType::Bool
        } else {
            self.tensors[x].dtype
        };
        let x_is_graph = matches!(&self.tensors[x].state, TensorState::Graph { .. });
        let y_is_graph = matches!(&self.tensors[y].state, TensorState::Graph { .. });
        if x_is_graph || y_is_graph {
            let graph_id = if x_is_graph {
                match self.tensors[x].state {
                    TensorState::Graph { graph_id, .. } => graph_id,
                    _ => unreachable!(),
                }
            } else {
                match self.tensors[y].state {
                    TensorState::Graph { graph_id, .. } => graph_id,
                    _ => unreachable!(),
                }
            };
            if !x_is_graph {
                self.promote_to_graph(x, graph_id)?;
            }
            if !y_is_graph {
                self.promote_to_graph(y, graph_id)?;
            }
            let TensorState::Graph { class_id: x, .. } = self.tensors[x].state else {
                unreachable!()
            };
            let TensorState::Graph { class_id: y, .. } = self.tensors[y].state else {
                unreachable!()
            };
            let (_node_id, class_id) = self.graphs[graph_id].push(Node::Binary { x, y, bop }, shape_id, dtype);

            let tid = self.tensors.push(TensorData { shape_id, dtype, state: TensorState::Graph { class_id, rc: 1, graph_id } });
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, nid={_node_id:?}, cid={class_id:?}");
            Ok(tid)
        } else {
            let (kid_x, op_id_x) = match &self.tensors[x].state {
                TensorState::Eager { kernel_id, op_id, .. } => (*kernel_id, *op_id),
                TensorState::Graph { rc, .. } if *rc > 0 => {
                    panic!("tensor was never realized. Did you forget to call tape.realize()?");
                }
                _ => unreachable!("eager binary with graph tensor"),
            };
            let (kid_y, op_id_y) = match &self.tensors[y].state {
                TensorState::Eager { kernel_id, op_id, .. } => (*kernel_id, *op_id),
                TensorState::Graph { rc, .. } if *rc > 0 => {
                    panic!("tensor was never realized. Did you forget to call tape.realize()?");
                }
                _ => unreachable!("eager binary with graph tensor"),
            };
            //println!("Binary input kernels: {kid_x:?} and {kid_y:?}");

            let (kernel_id, op_id) = if kid_x == kid_y {
                let op_id = self.kernels[kid_x].kernel.binary(op_id_x, op_id_y, bop);
                (kid_x, op_id)
            } else {
                let x_stores = !self.kernels[kid_x].stores.is_empty();
                let y_stores = !self.kernels[kid_y].stores.is_empty();
                if x_stores || y_stores {
                    todo!("binary with stores not yet handled (kernelize.rs materializes input via add_store before merge)");
                }

                let swap = self.kernels[kid_y].kernel.is_reduce() && !self.kernels[kid_x].kernel.is_reduce();
                let (keep_kid, merge_kid, keep_op, merge_op) = if swap {
                    (kid_y, kid_x, op_id_y, op_id_x)
                } else {
                    (kid_x, kid_y, op_id_x, op_id_y)
                };

                //println!("Remove kernel {merge_kid:?}");
                let KernelData { outputs: merge_outputs, loads: merge_loads, stores: merge_stores, kernel } = unsafe {
                    //eprintln!("C: kernels.remove_and_return({merge_kid:?})");
                    self.kernels.remove_and_return(merge_kid)
                };
                let Kernel { ops: merge_ops, head: merge_head, .. } = kernel;

                let mut op_map: Map<OpId, OpId> = Map::with_hasher(BuildHasherDefault::new());
                let mut i = merge_head;
                while !i.is_null() {
                    let mut op = merge_ops[i].op.clone();
                    for param in op.parameters_mut() {
                        if let Some(&new_param) = op_map.get(param) {
                            *param = new_param;
                        }
                    }
                    let new_op_id = self.kernels[keep_kid].kernel.push_back(op);
                    op_map.insert(i, new_op_id);
                    i = merge_ops[i].next;
                }

                for (_tid, t_data) in self.tensors.iter_mut() {
                    if let TensorState::Eager { kernel_id, op_id, .. } = &mut t_data.state {
                        if *kernel_id == merge_kid {
                            *kernel_id = keep_kid;
                            if let Some(&new_op_id) = op_map.get(op_id) {
                                *op_id = new_op_id;
                            }
                        }
                    }
                }

                //eprintln!("D: kernel_data.remove({merge_kid:?})");
                let keep_data = &mut self.kernels[keep_kid];
                keep_data.outputs.extend(merge_outputs);
                keep_data.loads.extend(merge_loads);
                keep_data.stores.extend(merge_stores);

                let op_id = if swap {
                    self.kernels[keep_kid].kernel.binary(op_map[&merge_op], keep_op, bop)
                } else {
                    self.kernels[keep_kid].kernel.binary(keep_op, op_map[&merge_op], bop)
                };
                (keep_kid, op_id)
            };

            let tid = self.tensors.push(TensorData {
                shape_id,
                dtype,
                state: TensorState::Eager { kernel_id, op_id, pending: KernelId::NULL },
            });
            self.kernels[kernel_id].outputs.push(tid);

            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
            Ok(tid)
        }
    }

    pub fn to_device(&mut self, x: TensorId, device_id: DeviceId) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::to_device(x={x}, device_id={device_id:?})");
        let TensorState::Graph { class_id, graph_id, .. } = self.tensors[x].state else {
            todo!("eager to_device")
        };
        let graph = &mut self.graphs[graph_id];
        let shape_id = self.tensors[x].shape_id;
        let dtype = self.tensors[x].dtype;
        // TODO measure actual time by running a test copy
        let (_node_id, cid) = graph.push(Node::ToDevice { x: class_id, device: device_id, time: 0 }, shape_id, dtype);
        let tid = self.tensors.push(TensorData { shape_id, dtype, state: TensorState::Graph { class_id: cid, rc: 1, graph_id } });
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, nid={_node_id:?}, cid={cid:?}");
        Ok(tid)
    }

    pub fn reduce(&mut self, x: TensorId, mut axes: Vec<UAxis>, rop: BOp) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::reduce(x={x}, axes={axes:?}, rop={rop:?})");
        let dtype = self.tensors[x].dtype;
        let shape = self.shape(x).to_vec();
        axes.sort_unstable();
        let reduce_shape = crate::shape::reduce(&shape, &axes);
        let shape_id = self.push_shape(reduce_shape);

        match self.tensors[x].state {
            TensorState::Graph { class_id, graph_id, .. } => {
                let (_node_id, class_id) = self.graphs[graph_id].push(
                    Node::Reduce { x: class_id, bop: rop, axes: axes.into_boxed_slice() },
                    shape_id,
                    dtype,
                );
                let tid =
                    self.tensors.push(TensorData { shape_id, dtype, state: TensorState::Graph { class_id, rc: 1, graph_id } });
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> tid={tid}, nid={_node_id:?}, cid={class_id:?}");
                Ok(tid)
            }
            _ => {
                let (kid, mut op_id) = self.duplicate_or_store(x, false)?;

                let n = shape.len();
                let max_axis = *axes.last().unwrap() as usize;
                let mut ai = 0;
                let mut permute_axes = Vec::with_capacity(n);
                for i in 0..=max_axis {
                    if axes[ai] as usize == i {
                        ai += 1;
                    } else {
                        permute_axes.push(i as UAxis);
                    }
                }
                permute_axes.extend((max_axis + 1..n).map(|i| i as UAxis));
                permute_axes.extend_from_slice(&axes);

                if !permute_axes.iter().copied().eq(0..permute_axes.len() as UAxis) {
                    op_id = self.kernels[kid].kernel.permute(op_id, &permute_axes);
                }

                op_id = self.kernels[kid].kernel.push_back(Op::Reduce { x: op_id, rop, n_axes: axes.len() });

                if shape.len() == axes.len() {
                    op_id = self.kernels[kid].kernel.reshape(op_id, &[1]);
                }

                let tid = self.tensors.push(TensorData {
                    shape_id,
                    dtype,
                    state: TensorState::Eager { kernel_id: kid, op_id, pending: KernelId::NULL },
                });

                debug_assert_eq!(self.kernels[kid].outputs.len(), 0, "input into reduce must have empty outputs");
                self.kernels[kid].outputs.push(tid);

                #[cfg(feature = "debug_tensor_op")]
                println!("  -> tid={tid}, kid={kid:?}, op_id={op_id:?}");
                Ok(tid)
            }
        }
    }

    pub(super) fn reshape(&mut self, x: TensorId, shape: Vec<Dim>) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::reshape(x={x}, shape={shape:?})");
        let sh = self.shape(x);
        debug_assert_eq!(
            shape.iter().product::<Dim>(),
            sh.iter().product::<Dim>(),
            "reshape: element count mismatch: {:?} vs {:?}",
            shape,
            sh
        );
        debug_assert!(!shape.is_empty(), "reshape: empty shape");
        if shape == sh {
            self.retain(x);
            return x;
        }

        let shape_id = self.push_shape(shape.clone());
        let dtype = self.tensors[x].dtype;

        match self.tensors[x].state {
            TensorState::Graph { class_id, graph_id, .. } => {
                let (_, class_id) = self.graphs[graph_id].push(Node::Reshape { x: class_id, shape: shape_id }, shape_id, dtype);
                let tid =
                    self.tensors.push(TensorData { shape_id, dtype, state: TensorState::Graph { class_id, rc: 1, graph_id } });
                tid
            }
            TensorState::Eager { .. } => {
                // If x is realized, create a load kernel with the target shape.
                // The result shares x's buffer (set in buffer_map), so add_store
                // won't add a StoreView for it. This avoids copying data for a
                // view-only reshape.
                if let Some(&buf_id) = self.buffer_map.get(&x) {
                    let mut kernel = Kernel::new(DeviceId::AUTO);
                    let op_id = kernel.load_contiguous(dtype, &shape);
                    let kernel_id =
                        self.kernels.push(KernelData { outputs: Vec::new(), loads: vec![x], stores: Vec::new(), kernel });
                    let tid = self.tensors.push(TensorData {
                        shape_id,
                        dtype,
                        state: TensorState::Eager { kernel_id, op_id, pending: KernelId::NULL },
                    });
                    self.kernels[kernel_id].outputs.push(tid);
                    self.buffer_map.insert(tid, buf_id);
                    #[cfg(feature = "debug_tensor_op")]
                    println!("  -> tid={tid} (load kernel, shares buffer with x={x})");
                    return tid;
                }

                let (kernel_id_dup, op_id_dup) = self.duplicate_or_store(x, false).unwrap();
                let op_id = self.kernels[kernel_id_dup].kernel.reshape(op_id_dup, &shape);
                let tid = self.tensors.push(TensorData {
                    shape_id,
                    dtype,
                    state: TensorState::Eager { kernel_id: kernel_id_dup, op_id, pending: KernelId::NULL },
                });

                debug_assert_eq!(self.kernels[kernel_id_dup].outputs.len(), 0, "input into reshape must have empty outputs");
                self.kernels[kernel_id_dup].outputs.push(tid);

                #[cfg(feature = "debug_tensor_op")]
                println!("  -> tid={tid}, kid={kernel_id_dup:?}, op_id={op_id:?}");
                tid
            }
        }
    }

    pub fn expand(&mut self, x: TensorId, shape: Vec<Dim>) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::expand(x={x}, shape={shape:?})");

        let sh = self.shape(x);
        debug_assert!(
            sh.len() <= shape.len(),
            "expand: input rank {} > target rank {}: {:?} -> {:?}",
            sh.len(),
            shape.len(),
            sh,
            shape
        );
        for (old, new) in sh.iter().copied().rev().zip(shape.iter().copied().rev()) {
            debug_assert!(old == new || old == 1, "expand: incompatible dims: {old} vs {new} in {:?} -> {:?}", sh, shape);
        }

        if shape == sh {
            self.retain(x);
            return Ok(x);
        }

        let shape_id = self.push_shape(shape);
        let dtype = self.tensors[x].dtype;

        match self.tensors[x].state {
            TensorState::Graph { class_id, graph_id, .. } => {
                let (_, class_id) = self.graphs[graph_id].push(Node::Expand { x: class_id, shape: shape_id }, shape_id, dtype);
                let tid =
                    self.tensors.push(TensorData { shape_id, dtype, state: TensorState::Graph { class_id, rc: 1, graph_id } });
                Ok(tid)
            }
            TensorState::Eager { kernel_id, op_id, .. } => {
                let force_store = self.kernels[kernel_id].kernel.is_preceded_by_compute(op_id);
                let (kernel_id, op_id) = self.duplicate_or_store(x, force_store)?;

                let op_id = self.kernels[kernel_id].kernel.expand(op_id, &self.shapes[shape_id]);
                let tid = self.tensors.push(TensorData {
                    shape_id,
                    dtype,
                    state: TensorState::Eager { kernel_id, op_id, pending: KernelId::NULL },
                });

                debug_assert_eq!(self.kernels[kernel_id].outputs.len(), 0, "input into expand must have empty outputs");
                self.kernels[kernel_id].outputs.push(tid);

                #[cfg(feature = "debug_tensor_op")]
                println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                Ok(tid)
            }
        }
    }

    pub fn permute(&mut self, x: TensorId, axes: Vec<UAxis>) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::permute(x={x}, axes={axes:?})");
        let sh = self.shape(x);
        debug_assert_eq!(axes.len(), sh.len(), "permute: axes length {} != rank {}", axes.len(), sh.len());
        {
            let mut sorted = axes.clone();
            sorted.sort();
            debug_assert!(
                sorted.iter().copied().eq(0..sh.len() as UAxis),
                "permute: axes not a valid permutation: {axes:?} for rank {}",
                sh.len()
            );
        }
        if axes.iter().copied().eq(0..sh.len() as UAxis) {
            self.retain(x);
            return x;
        }

        let new_shape = crate::shape::permute(self.shape(x), &axes);
        let shape_id = self.push_shape(new_shape.clone());
        let dtype = self.tensors[x].dtype;

        match self.tensors[x].state {
            TensorState::Graph { class_id, graph_id, .. } => {
                let (_, class_id) =
                    self.graphs[graph_id].push(Node::Permute { x: class_id, axes: axes.into_boxed_slice() }, shape_id, dtype);
                let tid =
                    self.tensors.push(TensorData { shape_id, dtype, state: TensorState::Graph { class_id, rc: 1, graph_id } });
                tid
            }
            TensorState::Eager { .. } => {
                let (kernel_id, op_id) = self.duplicate_or_store(x, false).unwrap();
                let op_id = self.kernels[kernel_id]
                    .kernel
                    .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Permute { axes: axes.into(), shape: new_shape }) });
                let tid = self.tensors.push(TensorData {
                    shape_id,
                    dtype,
                    state: TensorState::Eager { kernel_id, op_id, pending: KernelId::NULL },
                });
                debug_assert_eq!(self.kernels[kernel_id].outputs.len(), 0, "input into permute must have empty outputs");
                self.kernels[kernel_id].outputs.push(tid);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
        }
    }

    pub fn pad_zeros(&mut self, x: TensorId, padding: Vec<(i64, i64)>) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::pad_zeros(x={x}, padding={padding:?})");

        let sh = self.shape(x);
        debug_assert_eq!(padding.len(), sh.len(), "pad_zeros: padding length {} != rank {}", padding.len(), sh.len());

        let child_n: Dim = sh.iter().product();
        let mut new_shape = sh.to_vec();
        crate::shape::pad(&mut new_shape, &padding);
        let pad_n: Dim = new_shape.iter().product();
        let shape_id = self.push_shape(new_shape.clone());
        let dtype = self.tensors[x].dtype;

        match self.tensors[x].state {
            TensorState::Graph { class_id, graph_id, .. } => {
                let (_, class_id) = self.graphs[graph_id].push(
                    Node::PadZeros { x: class_id, padding: padding.into_boxed_slice() },
                    shape_id,
                    dtype,
                );
                let tid =
                    self.tensors.push(TensorData { shape_id, dtype, state: TensorState::Graph { class_id, rc: 1, graph_id } });
                tid
            }
            TensorState::Eager { kernel_id, op_id, .. } => {
                let force_store = pad_n > child_n && self.kernels[kernel_id].kernel.is_preceded_by_compute(op_id);
                let (kernel_id, op_id) = self.duplicate_or_store(x, force_store).unwrap();
                let op_id = self.kernels[kernel_id]
                    .kernel
                    .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Pad { padding, shape: new_shape }) });
                let tid = self.tensors.push(TensorData {
                    shape_id,
                    dtype,
                    state: TensorState::Eager { kernel_id, op_id, pending: KernelId::NULL },
                });
                debug_assert_eq!(self.kernels[kernel_id].outputs.len(), 0, "input into pad must have empty outputs");
                self.kernels[kernel_id].outputs.push(tid);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
        }
    }

    // Data can be smaller or equal lenght as number of elements in tensor.
    // If data is smaller, only first elements in tensor will be loaded.
    pub fn load<T: Scalar>(&mut self, x: TensorId, data: &mut [T]) -> Result<(), ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::load(x={x})");
        let dt = self.tensors[x].dtype;
        if dt != T::dtype() {
            return Err(ZyxError::DTypeError(format!("loading dtype {}, but the data has dtype {dt}", T::dtype()).into()));
        }

        let shape_numel: Dim = self.shape(x).iter().product();
        if (data.len() as Dim) > shape_numel {
            return Err(ZyxError::AllocationError(
                format!("load buffer of {} elements is larger than tensor with {shape_numel} elements", data.len()).into(),
            ));
        }

        // Fast path: already realized
        if let Some(&buffer_id) = self.buffer_map.get(&x) {
            let bytes = (data.len() * T::bit_size() as usize + 7) / 8;
            let byte_slice = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), bytes) };
            for buffers in self.events.keys() {
                if buffers.contains(&buffer_id) {
                    let buffers = buffers.clone();
                    let event = self.events.remove(&buffers).unwrap();
                    self.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, vec![event])?;
                    #[cfg(feature = "debug_tensor_op")]
                    println!("  -> x={x}, {:?}", self.tensors[x]);
                    return Ok(());
                }
            }
            self.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, Vec::new())?;
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> x={x}, {:?}", self.tensors[x]);
            return Ok(());
        }

        // Slow path: add store for each output, last one triggers materialize
        self.initialize_devices()?;

        let kid = match &self.tensors[x].state {
            TensorState::Eager { kernel_id, .. } => *kernel_id,
            TensorState::Graph { .. } => {
                return Err(ZyxError::graph_tensor_not_realized(x));
            }
        };

        // Deduplicate: add_store removes ALL occurrences at once and creates a load kernel,
        // so we must process each unique tid only once
        let seen: Set<TensorId> = self.kernels[kid].outputs.iter().copied().collect();
        for tid in seen {
            self.add_store(tid)?;
        }

        // Copy result to host
        let bytes = (data.len() * T::bit_size() as usize + 7) / 8;
        let byte_slice = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), bytes) };
        let buffer_id = self.buffer_map[&x];
        for buffers in self.events.keys() {
            if buffers.contains(&buffer_id) {
                let buffers = buffers.clone();
                let event = self.events.remove(&buffers).unwrap();
                self.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, vec![event])?;
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> x={x}, {:?}", self.tensors[x]);
                return Ok(());
            }
        }
        self.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, Vec::new())?;
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> x={x}, {:?}", self.tensors[x]);
        Ok(())
    }

    // Initializes all available devices, creating a device for each compute
    // device and a memory pool for each physical memory.
    // Does nothing if devices were already initialized.
    // Returns error if all devices failed to initialize
    // DeviceParameters allows to disable some devices if requested
    pub fn initialize_devices(&mut self) -> Result<(), ZyxError> {
        if !self.devices.is_empty() {
            return Ok(());
        }

        // Set env vars
        if let Ok(x) = env::var("ZYX_DEBUG")
            && let Ok(x) = x.parse::<u32>()
        {
            self.debug = DebugMask(x);
        }

        // Search through config directory and find zyx/backend_config.json
        // If not found or failed to parse, use defaults.

        let config_file = env::var_os("XDG_CONFIG_HOME")
            .and_then(|path| {
                let path = PathBuf::from(path);
                if path.is_absolute() { Some(path) } else { None }
            })
            .or_else(|| env::home_dir().map(|home| home.join(".config")))
            .map(|path| path.join("zyx/config.json"))
            .and_then(|mut path| {
                if let Ok(file) = std::fs::read_to_string(&path) {
                    path.pop();
                    self.config_dir = Some(path);
                    Some(file)
                } else {
                    None
                }
            });

        let config = config_file
            .and_then(|file| {
                DeJson::deserialize_json(&file)
                    .map_err(|e| {
                        if self.debug.dev() {
                            println!("Failed to parse config.json, {e}");
                        }
                    })
                    .ok()
            })
            .inspect(|_| {
                if self.debug.dev() {
                    println!("Device config successfully read and parsed.");
                }
            })
            .unwrap_or_else(|| {
                if self.debug.dev() {
                    println!("Failed to get device config, using defaults.");
                }
                Config::default()
            });

        // Load optimizer cache from disk if it exists
        /*if let Some(mut path) = self.config_dir.clone() {
            path.push("cached_kernels");
            if let Ok(mut file) = std::fs::File::open(path) {
                use std::io::Read;
                let mut buf = Vec::new();
                file.read_to_end(&mut buf).unwrap();
                if let Ok(cache) = nanoserde::DeBin::deserialize_bin(&buf) {
                    self.kernel_cache = cache;
                }
            }
        }*/

        crate::backend::initialize_backends(&config, &mut self.pools, &mut self.devices, self.debug.dev())?;

        self.autotune_config = config.autotune;
        //println!("INIT runtime");
        Ok(())
    }

    /// This function deinitializes the whole runtime, deallocates all allocated memory and deallocates all caches
    /// It does not reset the rng and it does not change debug, search, training and `config_dir` fields
    #[allow(unused)]
    pub fn deinitialize(&mut self) {
        #[cfg(feature = "time")]
        {
            let lock = crate::ET.lock();
            let mut timings: Vec<_> = lock.iter().map(|(name, &(total_us, count))| (name.clone(), total_us, count)).collect();
            timings.sort_by_key(|a| std::cmp::Reverse(a.1));
            println!("\n=== Timing Info (sorted by total time, descending) ===");
            for (name, total_us, count) in timings {
                let per_call = total_us.checked_div(count).unwrap_or(0);
                println!("{name}: {total_us}us total, {per_call}us/call ({count} calls)");
            }
        }
        //println!("DEINIT runtime");
        self.shape_map = Map::default();
        self.shapes = Slab::new();
        self.tensors = Slab::new();
        self.kernels = Slab::new();
    }

    pub const fn manual_seed(&mut self, seed: u64) {
        self.rng = Rng::seed_from_u64(seed);
    }

    /// Returns the maximum free bytes available across all memory pools.
    pub fn free_memory(&self) -> Dim {
        self.pools.iter().map(|(_, p)| p.free_bytes()).max().unwrap_or(0)
    }
}

#[allow(clippy::similar_names)]
pub fn get_perf(flop: u64, bytes_read: u64, bytes_written: u64, nanos: u64) -> String {
    const fn value_unit(x: u64) -> (u64, &'static str) {
        match x {
            0..1000 => (x * 100, ""),
            1_000..1_000_000 => (x / 10, "k"),
            1_000_000..1_000_000_000 => (x / 10_000, "M"),
            1_000_000_000..1_000_000_000_000 => (x / 10_000_000, "G"),
            1_000_000_000_000..1_000_000_000_000_000 => (x / 10_000_000_000, "T"),
            1_000_000_000_000_000..1_000_000_000_000_000_000 => (x / 10_000_000_000_000, "P"),
            1_000_000_000_000_000_000.. => (x / 10_000_000_000_000_000, "E"),
        }
    }

    if nanos == u64::MAX {
        return "INF time taken".to_string();
    }

    let (t, t_u) = match nanos {
        0..1_000 => (nanos * 10, "ns"),
        1_000..1_000_000 => (nanos / 100, "μs"),
        1_000_000..1_000_000_000 => (nanos / 100_000, "ms"),
        1_000_000_000..1_000_000_000_000 => (nanos / 100_000_000, "s"),
        1_000_000_000_000.. => (nanos / 6_000_000_000, "min"),
    };

    let (fs, f_us) = value_unit(flop * 1_000_000 / nanos * 1000);
    let (brs, br_us) = value_unit(bytes_read * 1_000_000_000 / nanos);
    let (bws, bw_us) = value_unit(bytes_written * 1_000_000_000 / nanos);

    format!(
        "{}.{} {t_u} ~ {}.{:02} {f_us}FLOP/s, {}.{:02} {br_us}B/s r, {}.{:02} {bw_us}B/s w",
        t / 10,
        t % 10,
        fs / 100,
        fs % 100,
        brs / 100,
        brs % 100,
        bws / 100,
        bws % 100,
    )
}

impl Runtime {
    fn duplicate_or_store(&mut self, x: TensorId, force_store: bool) -> Result<(KernelId, OpId), ZyxError> {
        let (mut kid, mut op_id) = match &self.tensors[x].state {
            TensorState::Eager { kernel_id, op_id, .. } => (*kernel_id, *op_id),
            TensorState::Graph { .. } => unreachable!("duplicate_or_store in graph mode"),
        };

        let contains_stores = self.kernels[kid].kernel.contains_stores();
        let preceded_by_reduce = self.kernels[kid].kernel.is_preceded_by_reduce(op_id);
        if force_store || contains_stores | preceded_by_reduce {
            self.add_store(x)?;
            (kid, op_id) = match self.tensors[x].state {
                TensorState::Eager { kernel_id, op_id, .. } => (kernel_id, op_id),
                _ => unreachable!(),
            };
            // We need to duplicate the new load kernel too, which we do below
        }

        debug_assert!(self.kernels[kid].stores.is_empty(), "duplicated kernel must not have stores");

        let loads = self.kernels[kid].loads.clone();
        let out_op_ids: Vec<OpId> = self.kernels[kid]
            .outputs
            .iter()
            .map(|&tid| match self.tensors[tid].state {
                TensorState::Eager { op_id, .. } => op_id,
                _ => unreachable!(),
            })
            .collect();
        let (kernel, new_op_id, self_loads, loads) = self.kernels[kid].kernel.extract_subkernel(op_id, &out_op_ids, &loads);
        self.kernels[kid].loads = self_loads;

        kid = self.kernels.push(KernelData { outputs: Vec::new(), loads, stores: Vec::new(), kernel });
        op_id = new_op_id;

        Ok((kid, op_id))
    }

    pub fn add_store(&mut self, x: TensorId) -> Result<(), ZyxError> {
        let (kid, op_id, pending) = match &self.tensors[x].state {
            TensorState::Eager { kernel_id, op_id, pending: pending_store, .. } => (*kernel_id, *op_id, *pending_store),
            TensorState::Graph { .. } => unreachable!("add_store in graph mode"),
        };

        // Remove ALL occurrences of x (handles reference counting from retain/clone)
        let prev_len = self.kernels[kid].outputs.len();
        self.kernels[kid].outputs.retain(|&e| e != x);
        let count = prev_len - self.kernels[kid].outputs.len();
        debug_assert!(count > 0, "add_store called for tid not in outputs");

        // Only add StoreView if x isn't already realized or pending
        let add_store = !self.buffer_map.contains_key(&x) && pending.is_null();
        let pending = if add_store {
            // Invariant: a kernel must never both load and store the same tensor
            debug_assert!(!self.kernels[kid].loads.contains(&x), "kernel {kid:?} both loads and stores tid {x}");

            let dtype = self.tensors[x].dtype;
            self.kernels[kid].kernel.store_contiguous(op_id, dtype);
            self.kernels[kid].stores.push(x);
            kid
        } else {
            pending
        };

        let outputs_empty = self.kernels[kid].outputs.is_empty();

        // Create load kernel so the tensor remains usable (visited must point to a live kernel)
        let dtype = self.tensors[x].dtype;
        let mut kernel = Kernel::new(DeviceId::AUTO);
        let shape = self.shape(x);
        let load_op_id = kernel.load_contiguous(dtype, &shape);
        let load_kid = self.kernels.push(KernelData { outputs: vec![x; count], loads: vec![x], stores: Vec::new(), kernel });
        self.tensors[x].state = TensorState::Eager { kernel_id: load_kid, op_id: load_op_id, pending };

        if outputs_empty {
            self.materialize_kernel(kid)?;
        }
        Ok(())
    }

    pub fn get_or_autotune(
        &mut self,
        mut kernel: Kernel,
        pool_id: PoolId,
        flop: u64,
        read: u64,
        write: u64,
        init_buffers: Option<&[PoolBufferId]>,
    ) -> Result<(DeviceProgramId, u64), ZyxError> {
        let kernel_id = if let Some(&cached_kid) = self.kernel_map.get(&kernel) {
            if let Some(&program_id) = self.programs.get(&cached_kid) {
                let pid = ProgramId { device: kernel.device_id, program: program_id };
                let timing = self.timings.get(&pid).copied().unwrap_or(10_000_000_000);
                return Ok((program_id, timing));
            }

            let dev_info = self.devices[kernel.device_id].info().clone();
            let dev_info_id = self.get_or_add_dev_info(&dev_info);

            if let Some(opt_seq) = self.optimizations.get(&(cached_kid, dev_info_id)) {
                opt_seq.apply(&mut kernel, &dev_info);
                let program_id = {
                    let device = &mut self.devices[kernel.device_id];
                    device.compile(&kernel, self.debug.asm())?
                };
                self.programs.insert(cached_kid, program_id);
                return Ok((program_id, 0));
            }
            cached_kid
        } else {
            let kernel_id =
                KernelId::from(self.kernel_map.values().copied().max().map_or(0, |id| usize::from(id).checked_add(1).unwrap()));
            let newly_inserted = self.kernel_map.insert(kernel.clone(), kernel_id).is_none();
            assert!(newly_inserted);
            kernel_id
        };

        let dev_info = self.devices[kernel.device_id].info().clone();
        let dev_info_id = self.get_or_add_dev_info(&dev_info);

        kernel.sort_global_defines();

        if self.debug.sched() {
            kernel.debug();
        }

        kernel.unfold_movement_ops();

        {
            let device = &mut self.devices[kernel.device_id];
            let global_indices = kernel.get_group_indices();
            let max_global_dims = device.info().max_global_work_dims.len();
            if global_indices.len() > max_global_dims {
                let n = global_indices.len() + 1 - max_global_dims;
                let indices: Vec<OpId> = global_indices.values().copied().take(n).collect();
                kernel.merge_indices(&indices);
            }
            kernel.renumber_indices();
            kernel.verify();
        }

        #[cfg(debug_assertions)]
        if let Some(buffers) = init_buffers {
            let n_ro_global =
                kernel.ops.values().filter(|op| matches!(&op.op, Op::Define { scope: MemScope::Global, ro: true, .. })).count();
            assert_eq!(
                buffers.len(),
                n_ro_global,
                "init_buffers len ({}) must match number of global read-only defines ({}) in kernel",
                buffers.len(),
                n_ro_global,
            );
        }

        let (program_id, opts, timing) = kernel.autotune_(
            &mut self.devices[kernel.device_id],
            &mut self.pools[pool_id],
            &self.autotune_config,
            flop,
            read,
            write,
            self.debug,
            init_buffers,
        )?;

        self.programs.insert(kernel_id, program_id);
        self.optimizations.insert((kernel_id, dev_info_id), opts);
        self.timings.insert(ProgramId { device: kernel.device_id, program: program_id }, timing);

        Ok((program_id, timing))
    }

    /// Materializes a kernel by adding store ops for all its outputs, compiling,
    /// launching, then creating load kernels for each output so the tensors remain
    /// usable in further graph construction. The kernel is consumed (removed from
    /// the slab) and cached in `kernel_map`/`programs` for reuse.
    ///
    /// # Invariant
    /// A kernel must never both load and store the same tensor (prevents aliasing).
    /// The debug_assert in the recursive materialization loop enforces this.
    fn materialize_kernel(&mut self, kid: KernelId) -> Result<(), ZyxError> {
        let KernelData { outputs, loads, stores, mut kernel } = unsafe { self.kernels.remove_and_return(kid) };

        debug_assert!(outputs.is_empty(), "all outputs must be stored before materialize");

        if stores.is_empty() {
            return Ok(());
        }

        debug_assert!(
            loads.iter().all(|&tid| {
                self.buffer_map.contains_key(&tid)
                    || outputs.contains(&tid)
                    || self.kernels.values().any(|kd| kd.outputs.contains(&tid) || kd.stores.contains(&tid))
            }),
            "load tid must be realized or in kernel's own outputs, or in some other kernel's outputs/stores"
        );

        // Debug: ensure each store tid is in exactly one kernel's outputs
        // (count may be 0 if add_store removed it and triggered this materialization)
        #[cfg(debug_assertions)]
        {
            for &tid in &stores {
                let count = self.kernels.values().filter(|kd| kd.outputs.contains(&tid)).count();
                debug_assert!(count <= 1, "store tid={tid} is in {count} kernels' outputs");
            }
        }

        // Recursive materialization: find producer kernels (those that have stores for our loads)
        // and materialize them so our loads become available.
        for &load in &loads {
            let pending = match &self.tensors[load].state {
                TensorState::Eager { pending, .. } => *pending,
                TensorState::Graph { .. } => KernelId::NULL,
            };
            if pending.is_null() {
                continue;
            }
            let outputs: Set<TensorId> = self.kernels[pending].outputs.iter().copied().collect();
            for output in outputs {
                self.add_store(output)?;
            }
        }

        debug_assert!(
            loads.iter().all(|&tid| self.buffer_map.contains_key(&tid)),
            "all loads must be realized after recursive materialization"
        );

        // Pick device and pool
        self.initialize_devices()?;
        let mut dev_ids: Vec<DeviceId> = self.devices.ids().collect();
        dev_ids.sort_unstable_by_key(|&dev_id| self.devices[dev_id].free_compute());
        dev_ids.reverse();
        let dev_id = *dev_ids.first().ok_or_else(|| ZyxError::AllocationError("no available device".into()))?;
        let pool_id = self.devices[dev_id].memory_pool_id();
        kernel.device_id = dev_id;

        // Ensure loads are in target pool
        let mut event_wait_list = Vec::new();
        for &tid in &loads {
            let buf_id = self.buffer_map[&tid];
            if buf_id.pool != pool_id {
                let src = buf_id.buffer;
                let bytes = (self.shape(tid).iter().product::<Dim>() as usize * self.dtype(tid).bit_size() as usize + 7) / 8;
                let mut byte_slice = vec![0u8; bytes];

                let mut ev = Vec::new();
                for buffers in self.events.keys() {
                    if buffers.contains(&buf_id) {
                        let buffers = buffers.clone();
                        let event = self.events.remove(&buffers).unwrap();
                        ev.push(event);
                        break;
                    }
                }
                self.pools[buf_id.pool].pool_to_host(src, &mut byte_slice, ev)?;
                self.buffer_map.remove(&tid);
                // Deallocate old buffer if no other mapping uses it
                if !self.buffer_map.values().any(|b| b.buffer == src) {
                    self.pools[buf_id.pool].deallocate(src, vec![]);
                }

                let (dst, event) = self.pools[pool_id].allocate(bytes as Dim)?;
                let dst_global = BufferId { pool: pool_id, buffer: dst };
                let event = self.pools[pool_id].host_to_pool(&byte_slice, dst, vec![event])?;
                self.pools[pool_id].sync_events(vec![event])?;
                self.buffer_map.insert(tid, dst_global);
            } else {
                for buffers in self.events.keys() {
                    if buffers.contains(&buf_id) {
                        let buffers = buffers.clone();
                        let event = self.events.remove(&buffers).unwrap();
                        event_wait_list.push(event);
                        break;
                    }
                }
            }
        }

        // Allocate store buffers (one per unique tid)
        let mut kernel_buffers = BTreeSet::new();
        // All kernel buffers (loads + stores) must be tracked in kernel_buffers
        // so future operations can find and wait on the kernel's event before
        // reusing any of these buffers.
        for &tid in &loads {
            kernel_buffers.insert(self.buffer_map[&tid]);
        }
        for &tid in &stores {
            let bytes = (self.shape(tid).iter().product::<Dim>() as usize * self.dtype(tid).bit_size() as usize + 7) / 8;
            // Add one trash element
            let alloc_bytes = bytes as Dim + Dim::from(self.dtype(tid).bit_size() / 8);
            let (buf, event) = self.pools[pool_id].allocate(alloc_bytes)?;
            let global_id = BufferId { pool: pool_id, buffer: buf };
            self.buffer_map.insert(tid, global_id);
            self.tensors[tid].state = match self.tensors[tid].state {
                TensorState::Eager { kernel_id, op_id, .. } => TensorState::Eager { kernel_id, op_id, pending: KernelId::NULL },
                TensorState::Graph { .. } => unreachable!("materialize_kernel with graph tensor"),
            };
            kernel_buffers.insert(global_id);
            event_wait_list.push(event);
        }

        // Build args: load buffers first, then store buffers
        let mut args = Vec::new();
        for &tid in &loads {
            args.push(self.buffer_map[&tid].buffer);
        }

        // Compile and launch (caches in kernel_map / programs)
        let (flop, read, write) = kernel.flop_mem_rw();
        let (dev_prog, _timing) = self.get_or_autotune(kernel, pool_id, flop, read, write, Some(&args))?;

        for &tid in &stores {
            args.push(self.buffer_map[&tid].buffer);
        }

        let event = self.devices[dev_id].launch(dev_prog, &mut self.pools[pool_id], &args, event_wait_list)?;
        self.events.insert(kernel_buffers, event);

        Ok(())
    }

    fn get_or_add_dev_info(&mut self, device_info: &DeviceInfo) -> DeviceInfoId {
        if let Some(&dev_info_id) = self.device_infos.get(device_info) {
            dev_info_id
        } else {
            let dev_info_id =
                DeviceInfoId(self.device_infos.values().copied().max().map_or(0, |id| id.0.checked_add(1).unwrap()));
            let newly_inserted = self.device_infos.insert(device_info.clone(), dev_info_id).is_none();
            assert!(newly_inserted);
            dev_info_id
        }
    }
}
