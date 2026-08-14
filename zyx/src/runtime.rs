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
    graph::{ClassId, EClass, ExecPlan, Graph, GraphId, Node, NodeData, NodeId, plan::drain_events_for_buf},
    kernel::{BOp, DeviceId, IDX_T, Kernel, MemLayout, MemScope, MoveOp, Op, OpId, UOp, autotune::OptSeq},
    rng::Rng,
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    tensor::TensorId,
};

/// Loads present in `old` but not in `new`, counting multiplicities.
fn loads_dropped_by_prune(old: &[TensorId], new: &[TensorId]) -> Vec<TensorId> {
    let mut dropped = Vec::new();
    let mut seen: Set<TensorId> = Set::default();
    for &tid in old {
        if !seen.insert(tid) {
            continue;
        }
        let old_c = old.iter().filter(|&&t| t == tid).count();
        let new_c = new.iter().filter(|&&t| t == tid).count();
        dropped.extend(std::iter::repeat_n(tid, old_c - new_c));
    }
    dropped
}

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
        self.0 += 1;
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
    pub kernel_id: KernelId,
    pub op_id: OpId,
    pub depends_on: KernelId,
    pub class_id: ClassId,
    pub graph_id: GraphId,
    pub rc: u16,
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

    // True if x is currently a graph tensor (class_id set and its graph alive).
    // A promoted non-realized tensor whose graph has died is treated as eager
    // (its kernel_id is still valid), so is_graph returns false in that case.
    pub(crate) fn is_graph(&self, x: TensorId) -> bool {
        let td = &self.tensors[x];
        !td.class_id.is_null() && !self.graphs[td.graph_id].dead
    }

    fn graph_ids(&self, x: TensorId) -> (ClassId, GraphId) {
        let td = &self.tensors[x];
        debug_assert!(!td.class_id.is_null());
        self.assert_graph_alive(td.graph_id);
        (td.class_id, td.graph_id)
    }

    pub fn eager_ids(&self, x: TensorId) -> (KernelId, OpId) {
        let td = &self.tensors[x];
        if td.kernel_id.is_null() {
            panic!(
                "tape scope has ended (tensor belongs to a dead tape scope; Tape dropped or realized without this tensor being an output)"
            );
        }
        (td.kernel_id, td.op_id)
    }

    /// Returns operation capabilities for a dtype across all devices.
    pub fn supports_dtype(&mut self, dtype: DType) -> DTypeCapability {
        self.initialize_backends();
        let mut caps = DTypeCapability::none();
        for (_id, dev) in self.devices.iter() {
            caps = caps.include(dev.info().supports_dtype(dtype));
        }
        caps
    }

    pub fn retain(&mut self, x: TensorId) {
        //eprintln!("Retain tensor x={x}");
        self.tensors[x].rc += 1;
        let kernel_id = self.tensors[x].kernel_id;
        if !kernel_id.is_null() {
            self.kernels[kernel_id].outputs.push(x);
        }
    }

    pub fn release(&mut self, x: TensorId) {
        let rc = self.tensors[x].rc - 1;
        self.tensors[x].rc = rc;
        let (kernel_id, op_id, pending, class_id) =
            (self.tensors[x].kernel_id, self.tensors[x].op_id, self.tensors[x].depends_on, self.tensors[x].class_id);

        // Keep the eager kernel's outputs in sync with rc.
        if !kernel_id.is_null() {
            let kd = &mut self.kernels[kernel_id];
            kd.outputs.iter().position(|e| *e == x).map(|i| kd.outputs.remove(i));
        }

        if !class_id.is_null() {
            // Graph-affiliated tensor (pure graph or "both" while graph alive).
            if rc == 0 {
                self.on_rc_zero(x);
            }
            return;
        }

        // Eager tensor path.
        // With custom kernels, op_id is null, so we have to skip the chain pruning.
        let pruned = if !op_id.is_null() {
            let kd = &mut self.kernels[kernel_id];
            if kd.outputs.contains(&x) {
                Vec::new()
            } else {
                let out_ops: Vec<OpId> = kd.outputs.iter().map(|&tid| self.tensors[tid].op_id).collect();
                let old_loads = std::mem::take(&mut kd.loads);
                let new_loads = kd.kernel.remove_unused_chain(op_id, &out_ops, &old_loads);
                kd.loads = new_loads.clone();
                loads_dropped_by_prune(&old_loads, &new_loads)
            }
        } else {
            Vec::new()
        };
        for tid in pruned {
            self.release_load(tid);
        }

        // rc == 0 means no handles and no kernel loads reference x anymore.
        // x is dead: remove it and free its buffer (unless a pending kernel is
        // still producing it). Loads dropped by the prune above are released
        // before this so that rc is accurate.
        if rc == 0 && pending.is_null() {
            self.on_rc_zero(x);
        }

        // In case it was used in assign, all of graph_id and kernel_id and class_id and op_id are null
        if kernel_id.is_null() {
            return;
        }

        if self.kernels[kernel_id].outputs.is_empty() {
            if !self.kernels[kernel_id].kernel.contains_stores() {
                //eprintln!("A: kernels.remove({kid:?})");
                self.remove_dead_eager_kernel(kernel_id);
            } else {
                self.materialize_kernel(kernel_id).unwrap();
            }
        }
    }

    /// A kernel-load reference on `x` was added (loads.push). Kernel loads are
    /// counted in `rc` so that load tensors and their buffers are freed once
    /// the last kernel referencing them dies.
    pub(crate) fn retain_load(&mut self, x: TensorId) {
        self.tensors[x].rc += 1;
    }

    /// A kernel-load reference on `x` was dropped (kernel removal or load
    /// pruning). If this was the last reference, `x` may be removed.
    fn release_load(&mut self, x: TensorId) {
        let rc = self.tensors[x].rc - 1;
        self.tensors[x].rc = rc;
        if rc == 0 {
            self.on_rc_zero(x);
        }
    }

    /// A tensor's reference count reached zero: no handles and no kernel loads
    /// reference it. Remove it, freeing its buffer if no other tensor maps to
    /// the same buffer. Graph-affiliated tensors may be kept by their graph.
    fn on_rc_zero(&mut self, x: TensorId) {
        let (pending, class_id, graph_id) = (self.tensors[x].depends_on, self.tensors[x].class_id, self.tensors[x].graph_id);

        if !class_id.is_null() {
            // Graph-affiliated tensor (pure graph or "both" while graph alive).
            if self.graphs.contains_key(graph_id) {
                if !self.graphs[graph_id].is_leaf(class_id) {
                    debug_assert!(!self.buffer_map.contains_key(&x), "dead non-leaf graph tensor holds a buffer");
                    self.tensors.remove(x);
                }
                self.graphs[graph_id].ref_count -= 1;
                if self.graphs[graph_id].dead && self.graphs[graph_id].ref_count == 0 {
                    self.remove_dead_graph(graph_id);
                }
            } else if !self.buffer_map.contains_key(&x) {
                self.tensors.remove(x);
            }
            return;
        }

        // Eager tensor: no references remain. If a pending kernel is still
        // producing x, keep it until that kernel materializes.
        if !pending.is_null() {
            return;
        }
        if let Some(buf_id) = self.buffer_map.remove(&x) {
            let still_used = self.buffer_map.values().any(|b| b.pool == buf_id.pool && b.buffer == buf_id.buffer);
            if !still_used {
                let wait_list = drain_events_for_buf(&mut self.events, buf_id);
                self.pools[buf_id.pool].deallocate(buf_id.buffer, wait_list);
            }
        }
        self.tensors.remove(x);
    }

    /// Remove a kernel that has no outputs and no stores, releasing its load
    /// references.
    fn remove_dead_eager_kernel(&mut self, kid: KernelId) {
        let loads = std::mem::take(&mut self.kernels[kid].loads);
        self.kernels.remove(kid);
        for tid in loads {
            self.release_load(tid);
        }
    }

    pub(crate) fn remove_dead_graph(&mut self, graph_id: GraphId) {
        let leaf_tids: Vec<TensorId> = self.graphs[graph_id].leaf_map.values().copied().collect();
        for tid in leaf_tids {
            // Dead leaves may already have been removed by Tape::drop.
            if !self.tensors.contains_key(tid) {
                continue;
            }
            if self.tensors[tid].graph_id == graph_id {
                if let Some(buf_id) = self.buffer_map.remove(&tid) {
                    let wait_list = drain_events_for_buf(&mut self.events, buf_id);
                    self.pools[buf_id.pool].deallocate(buf_id.buffer, wait_list);
                }
                self.tensors.remove(tid);
            }
        }
        self.graphs.remove(graph_id);
    }

    pub fn eagerify(&mut self, tid: TensorId) {
        if self.tensors[tid].class_id.is_null() {
            return;
        }
        let rc = self.tensors[tid].rc;
        let graph_id = self.tensors[tid].graph_id;
        let old_kernel_id = self.tensors[tid].kernel_id;

        // Release tid from its old eager kernel (if any): remove it from outputs,
        // prune the unused chain (releasing pruned loads), and drop the kernel if
        // nothing else uses it (releasing its remaining loads).
        let mut handles = rc as usize;
        let mut pruned: Vec<TensorId> = Vec::new();
        if !old_kernel_id.is_null() {
            let old_op_id = self.tensors[tid].op_id;
            let kernel_died = {
                let kd = &mut self.kernels[old_kernel_id];
                handles = kd.outputs.iter().filter(|&&e| e == tid).count();
                kd.outputs.retain(|&e| e != tid);
                if !old_op_id.is_null() {
                    let out_ops: Vec<OpId> = kd.outputs.iter().map(|&t| self.tensors[t].op_id).collect();
                    let old_loads = std::mem::take(&mut kd.loads);
                    let new_loads = kd.kernel.remove_unused_chain(old_op_id, &out_ops, &old_loads);
                    kd.loads = new_loads.clone();
                    pruned = loads_dropped_by_prune(&old_loads, &new_loads);
                }
                kd.outputs.is_empty()
            };
            for t in pruned {
                self.release_load(t);
            }
            if kernel_died {
                if !self.kernels[old_kernel_id].kernel.contains_stores() {
                    self.remove_dead_eager_kernel(old_kernel_id);
                } else {
                    self.materialize_kernel(old_kernel_id).unwrap();
                }
            }
        }

        self.tensors[tid].class_id = ClassId::NULL;
        self.tensors[tid].graph_id = GraphId::NULL;
        let shape: Box<[Dim]> = self.shape(tid).into();
        let dtype = self.dtype(tid);
        let scope = if self.is_variable_tensor(tid) {
            MemScope::Variable
        } else {
            MemScope::Global
        };
        let kernel_id = self.kernels.push(KernelData {
            outputs: vec![tid; handles],
            loads: Vec::new(),
            stores: Vec::new(),
            kernel: Kernel::new(DeviceId::AUTO),
        });
        let op_id = self.kernels[kernel_id].kernel.push_back(Op::Define { dtype, scope, ro: true, shape });
        self.kernels[kernel_id].loads.push(tid);
        self.tensors[tid].kernel_id = kernel_id;
        self.tensors[tid].op_id = op_id;
        self.tensors[tid].depends_on = KernelId::NULL;
        self.retain_load(tid);
        self.graphs[graph_id].ref_count -= 1;
    }

    fn assert_graph_alive(&self, graph_id: GraphId) {
        assert!(
            !self.graphs[graph_id].dead,
            "tape scope has ended (tensor belongs to a dead tape scope; Tape dropped or realized without this tensor being an output)"
        );
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

    pub fn push_leaf_node(&mut self, graph_id: GraphId, dtype: DType, shape: ShapeId) -> (NodeId, ClassId) {
        let g = &mut self.graphs[graph_id];
        let leaf_id = g.max_leaf_id;
        g.max_leaf_id += 1;
        let node = Node::Leaf { dtype, leaf_id };
        if let Some(&nid) = g.hashcons.get(&node) {
            return (nid, g.nodes[nid].class_of);
        }
        let nid = g.nodes.push(NodeData { node: node.clone(), class_of: ClassId::NULL });
        let cid = g.classes.push(EClass { nodes: vec![nid], shape, dtype });
        g.nodes[nid].class_of = cid;
        g.hashcons.insert(node, nid);
        (nid, cid)
    }

    pub(crate) fn new_graph_tensor(&mut self, graph_id: GraphId, class_id: ClassId, shape_id: ShapeId, dtype: DType) -> TensorId {
        self.graphs[graph_id].ref_count += 1;
        self.tensors.push(TensorData {
            shape_id,
            dtype,
            kernel_id: KernelId::NULL,
            op_id: OpId::NULL,
            depends_on: KernelId::NULL,
            class_id,
            graph_id,
            rc: 1,
        })
    }

    pub fn push_node(&mut self, graph_id: GraphId, node: Node, shape: ShapeId, dtype: DType) -> (NodeId, ClassId) {
        //println!("push node to graph_id={graph_id:?}");
        match node {
            Node::Permute { x, ref axes } => {
                let in_shape = &self.shapes[self.graphs[graph_id].classes[x].shape];
                assert_eq!(
                    axes.len(),
                    in_shape.len(),
                    "Permute: axes length {} != input rank {} (shape {:?})",
                    axes.len(),
                    in_shape.len(),
                    in_shape
                );
            }
            Node::Reshape { x, shape: out_shape_id } => {
                let in_shape = &self.shapes[self.graphs[graph_id].classes[x].shape];
                let out_shape = &self.shapes[out_shape_id];
                assert_eq!(
                    in_shape.iter().product::<Dim>(),
                    out_shape.iter().product::<Dim>(),
                    "Reshape: element count mismatch {:?} -> {:?}",
                    in_shape,
                    out_shape
                );
            }
            Node::Expand { x, shape: out_shape_id } => {
                let in_shape = &self.shapes[self.graphs[graph_id].classes[x].shape];
                let out_shape = &self.shapes[out_shape_id];
                assert!(
                    in_shape.len() <= out_shape.len(),
                    "Expand: input rank {} > output rank {}: {:?} -> {:?}",
                    in_shape.len(),
                    out_shape.len(),
                    in_shape,
                    out_shape
                );
                for (old, new) in in_shape.iter().copied().rev().zip(out_shape.iter().copied().rev()) {
                    assert!(
                        old == new || old == 1,
                        "Expand: incompatible dims: {old} vs {new} in {:?} -> {:?}",
                        in_shape,
                        out_shape
                    );
                }
            }
            Node::Reduce { x, ref axes, .. } => {
                let in_shape = &self.shapes[self.graphs[graph_id].classes[x].shape];
                for &a in axes.iter() {
                    assert!(
                        a < in_shape.len(),
                        "Reduce: axis {} out of range for input rank {} (shape {:?})",
                        a,
                        in_shape.len(),
                        in_shape
                    );
                }
            }
            Node::PadZeros { x, ref padding } => {
                let in_shape = &self.shapes[self.graphs[graph_id].classes[x].shape];
                assert_eq!(
                    padding.len(),
                    in_shape.len(),
                    "PadZeros: padding length {} != input rank {} (shape {:?})",
                    padding.len(),
                    in_shape.len(),
                    in_shape
                );
            }
            _ => {}
        }
        let g = &mut self.graphs[graph_id];
        if let Some(&nid) = g.hashcons.get(&node) {
            return (nid, g.nodes[nid].class_of);
        }
        let nid = g.nodes.push(NodeData { node: node.clone(), class_of: ClassId::NULL });
        let cid = g.classes.push(EClass { nodes: vec![nid], shape, dtype });
        g.nodes[nid].class_of = cid;
        g.hashcons.insert(node, nid);
        (nid, cid)
    }

    pub fn push_binary_node(&mut self, graph_id: GraphId, x: ClassId, y: ClassId, bop: BOp) -> ClassId {
        debug_assert_eq!(
            self.shapes[self.graphs[graph_id].classes[x].shape],
            self.shapes[self.graphs[graph_id].classes[y].shape]
        );
        self.push_node(
            graph_id,
            Node::Binary { x, y, bop },
            self.graphs[graph_id].classes[x].shape,
            self.graphs[graph_id].classes[x].dtype,
        )
        .1
    }

    /// Returns whether `x` is a variable: its buffer (if any) is stored as a
    /// scalar constant via `store_variable` rather than a real buffer.
    fn is_variable_tensor(&mut self, x: TensorId) -> bool {
        match self.buffer_map.get(&x) {
            Some(&buf_id) => self.pools[buf_id.pool].get_variable(buf_id.buffer).is_some(),
            None => false,
        }
    }

    pub fn new_eager_tensor(&mut self, shape: Vec<Dim>, dtype: DType, scope: MemScope) -> TensorId {
        let shape_id = self.push_shape(shape.clone());
        let mut kernel = Kernel::new(DeviceId::AUTO);
        let op_id = kernel.push_back(Op::Define { dtype, scope, ro: true, shape: shape.into() });
        let kernel_id = self.kernels.push(KernelData { outputs: Vec::new(), loads: Vec::new(), stores: Vec::new(), kernel });
        let tid = self.tensors.push(TensorData {
            shape_id,
            dtype,
            kernel_id,
            op_id,
            depends_on: KernelId::NULL,
            class_id: ClassId::NULL,
            graph_id: GraphId::NULL,
            rc: 1,
        });
        self.kernels[kernel_id].loads.push(tid);
        self.kernels[kernel_id].outputs.push(tid);
        tid
    }

    pub fn new_constant_tensor(&mut self, value: Constant) -> TensorId {
        let shape_id = self.push_shape(vec![1]);
        let dtype = value.dtype();
        let mut kernel = Kernel::new(DeviceId::AUTO);
        let op_id = kernel.push_back(Op::Const(value));
        let kernel_id = self.kernels.push(KernelData { outputs: Vec::new(), loads: Vec::new(), stores: Vec::new(), kernel });
        let tid = self.tensors.push(TensorData {
            shape_id,
            dtype,
            kernel_id,
            op_id,
            depends_on: KernelId::NULL,
            class_id: ClassId::NULL,
            graph_id: GraphId::NULL,
            rc: 1,
        });
        self.kernels[kernel_id].outputs.push(tid);
        tid
    }

    pub fn new_full(&mut self, shape: Vec<Dim>, value: Constant) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::new_full(shape={shape:?}, value={value:?})");
        let x = self.new_constant_tensor(value);
        let expanded = self.expand(x, shape).unwrap();
        self.release(x);
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={expanded}, {:?}", self.tensors[expanded]);
        expanded
    }

    pub fn new_variable_tensor<T: Scalar>(&mut self, x: T) -> TensorId {
        let dtype = T::dtype();
        self.initialize_backends();
        let tid = self.new_eager_tensor(vec![1], dtype, MemScope::Variable);
        self.retain_load(tid);

        let MemoryPool::Host(ref mut pool) = self.pools[PoolId::HOST] else {
            unreachable!("Host must exist.")
        };
        let buffer_id = BufferId { pool: PoolId::HOST, buffer: pool.store_variable(Constant::new(x)) };
        self.buffer_map.insert(tid, buffer_id);

        return tid;
    }

    // Creates new tensor in host memory
    pub fn new_host_tensor<T: Scalar>(&mut self, shape: Vec<Dim>, data: Box<[T]>) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::new_host_tensor(shape={shape:?})");

        if data.len() == 1 && shape.len() == 1 {
            let tid = self.new_constant_tensor(Constant::new(data[0]));
            return Ok(tid);
        }

        let dtype = T::dtype();
        self.initialize_backends();

        debug_assert_eq!(shape.iter().product::<Dim>(), data.len() as Dim);
        let bytes = (data.len() * dtype.bit_size() as usize).div_ceil(8);
        debug_assert_eq!(data.len() * std::mem::size_of::<T>(), bytes);

        // Allocate one element extra so masked store writes to the trash
        // element stay within bounds (eager tensors can become store
        // targets, e.g. in-place assign).
        let alloc_bytes = bytes + dtype.bit_size() as usize / 8;
        // Store to Host memory
        let MemoryPool::Host(ref mut pool) = self.pools[PoolId::HOST] else {
            unreachable!("Host must exist.")
        };
        let free_bytes = pool.free_bytes();
        if alloc_bytes as Dim > free_bytes {
            return Err(ZyxError::AllocationError(
                format!("Attempted to allocate {alloc_bytes} B on host, but it only has {free_bytes} B free").into(),
            ));
        }

        let mut buf = vec![0u8; alloc_bytes].into_boxed_slice();
        let src = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), bytes) };
        buf[..bytes].copy_from_slice(src);

        let buffer_id = BufferId { pool: PoolId::HOST, buffer: pool.insert(buf) };

        let tid = self.new_eager_tensor(shape, dtype, MemScope::Global);
        self.retain_load(tid);

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
        self.initialize_backends();
        let bytes: Dim = (shape.iter().product::<Dim>() * dtype.bit_size() as Dim).div_ceil(8);

        let pool = self.pools[PoolId::DISK]
            .disk_pool()
            .ok_or(BackendError { status: ErrorStatus::Initialization, context: "[disk] not available.".into() })?;
        let buffer_id = BufferId { pool: PoolId::DISK, buffer: pool.buffer_from_path(bytes, path, offset_bytes) };

        let tid = self.new_eager_tensor(shape, dtype, MemScope::Global);
        self.retain_load(tid);
        self.buffer_map.insert(tid, buffer_id);
        Ok(tid)
    }

    pub fn cast(&mut self, x: TensorId, dtype: DType) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::cast(x={x}, dtype={dtype:?})");
        let td = &self.tensors[x];
        let shape_id = self.tensors[x].shape_id;
        if td.class_id.is_null() {
            let kernel_id = td.kernel_id;
            let op_id = self.kernels[kernel_id].kernel.cast(td.op_id, dtype);
            let tid = self.tensors.push(TensorData {
                shape_id,
                dtype,
                kernel_id,
                op_id,
                depends_on: KernelId::NULL,
                class_id: ClassId::NULL,
                graph_id: GraphId::NULL,
                rc: 1,
            });
            self.kernels[kernel_id].outputs.push(tid);
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
            tid
        } else {
            let graph_id = td.graph_id;
            self.assert_graph_alive(graph_id);
            let (_, class_id) = self.push_node(graph_id, Node::Cast { x: td.class_id, dtype }, shape_id, dtype);
            self.new_graph_tensor(graph_id, class_id, shape_id, dtype)
        }
    }

    pub fn bitcast(&mut self, x: TensorId, dtype: DType) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::bitcast(x={x}, dtype={dtype:?})");
        let shape_id = self.tensors[x].shape_id;
        if self.is_graph(x) {
            let (class_id, graph_id) = self.graph_ids(x);
            let (_, class_id) = self.push_node(graph_id, Node::Cast { x: class_id, dtype }, shape_id, dtype);
            self.new_graph_tensor(graph_id, class_id, shape_id, dtype)
        } else {
            let (kernel_id, op_id) = self.eager_ids(x);
            let op_id = self.kernels[kernel_id].kernel.bitcast(op_id, dtype);
            let tid = self.tensors.push(TensorData {
                shape_id,
                dtype,
                kernel_id,
                op_id,
                depends_on: KernelId::NULL,
                class_id: ClassId::NULL,
                graph_id: GraphId::NULL,
                rc: 1,
            });
            self.kernels[kernel_id].outputs.push(tid);
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
            tid
        }
    }

    pub fn unary(&mut self, x: TensorId, uop: UOp) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::unary(x={x}, uop={uop:?})");
        let shape_id = self.tensors[x].shape_id;
        let dtype = self.tensors[x].dtype;
        if self.is_graph(x) {
            let (class_id, graph_id) = self.graph_ids(x);
            let (_node_id, class_id) = self.push_node(graph_id, Node::Unary { x: class_id, uop }, shape_id, dtype);
            let tid = self.new_graph_tensor(graph_id, class_id, shape_id, dtype);
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, nid={_node_id:?}, cid={class_id:?}");
            tid
        } else {
            let (kernel_id, op_id) = self.eager_ids(x);
            let op_id = self.kernels[kernel_id].kernel.unary(op_id, uop);
            let tid = self.tensors.push(TensorData {
                shape_id,
                dtype,
                kernel_id,
                op_id,
                depends_on: KernelId::NULL,
                class_id: ClassId::NULL,
                graph_id: GraphId::NULL,
                rc: 1,
            });
            self.kernels[kernel_id].outputs.push(tid);
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
            tid
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
        let x_is_graph = self.is_graph(x);
        let y_is_graph = self.is_graph(y);
        if x_is_graph || y_is_graph {
            let graph_id = if x_is_graph {
                self.graph_ids(x).1
            } else {
                self.graph_ids(y).1
            };
            self.assert_graph_alive(graph_id);
            if !x_is_graph {
                self.promote_to_graph(x, graph_id)?;
            }
            if !y_is_graph {
                self.promote_to_graph(y, graph_id)?;
            }
            let x = self.graph_ids(x).0;
            let y = self.graph_ids(y).0;
            let class_id = self.push_binary_node(graph_id, x, y, bop);

            Ok(self.new_graph_tensor(graph_id, class_id, shape_id, dtype))
        } else {
            let (mut kid_x, mut op_id_x) = self.eager_ids(x);
            let (mut kid_y, mut op_id_y) = self.eager_ids(y);
            eprintln!("BIN bop={bop:?} kid_x={kid_x:?} op_x={op_id_x:?} kid_y={kid_y:?} op_y={op_id_y:?} stores_x={} stores_y={}", !self.kernels[kid_x].stores.is_empty(), !self.kernels[kid_y].stores.is_empty());

            let (kernel_id, op_id) = if kid_x == kid_y {
                let op_id = self.kernels[kid_x].kernel.binary(op_id_x, op_id_y, bop);
                (kid_x, op_id)
            } else {
                let x_stores = !self.kernels[kid_x].stores.is_empty();
                let y_stores = !self.kernels[kid_y].stores.is_empty();
                match (x_stores, y_stores) {
                    (true, true) => {
                        self.add_store(x)?;
                        self.add_store(y)?;
                    }
                    (true, false) => self.add_store(x)?,
                    (false, true) => self.add_store(y)?,
                    (false, false) => {}
                }
                (kid_x, op_id_x) = self.eager_ids(x);
                (kid_y, op_id_y) = self.eager_ids(y);

                let swap = self.kernels[kid_y].kernel.is_reduce() && !self.kernels[kid_x].kernel.is_reduce();
                let (keep_kid, merge_kid, keep_op, merge_op) = if swap {
                    (kid_y, kid_x, op_id_y, op_id_x)
                } else {
                    (kid_x, kid_y, op_id_x, op_id_y)
                };

                eprintln!("  -> merge: keep={keep_kid:?} merge={merge_kid:?}");

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
                    if t_data.kernel_id == merge_kid {
                        t_data.kernel_id = keep_kid;
                        if let Some(&new_op_id) = op_map.get(&t_data.op_id) {
                            t_data.op_id = new_op_id;
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
                kernel_id,
                op_id,
                depends_on: KernelId::NULL,
                class_id: ClassId::NULL,
                graph_id: GraphId::NULL,
                rc: 1,
            });
            self.kernels[kernel_id].outputs.push(tid);

            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
            Ok(tid)
        }
    }

    #[allow(clippy::wrong_self_convention)] // naming convention from GPU API, not a conversion method
    pub fn to_device(&mut self, x: TensorId, device_id: DeviceId) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::to_device(x={x}, device_id={device_id:?})");
        let (class_id, graph_id) = self.graph_ids(x);
        self.assert_graph_alive(graph_id);
        let shape_id = self.tensors[x].shape_id;
        let dtype = self.tensors[x].dtype;
        // TODO measure actual time by running a test copy
        let (_node_id, cid) =
            self.push_node(graph_id, Node::ToDevice { x: class_id, device: device_id, time: 0 }, shape_id, dtype);
        let tid = self.new_graph_tensor(graph_id, cid, shape_id, dtype);
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, nid={_node_id:?}, cid={cid:?}");
        Ok(tid)
    }

    pub fn contiguous(&mut self, x: TensorId) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::contiguous(x={x})");

        if self.is_graph(x) {
            let (class_id, graph_id) = self.graph_ids(x);
            let shape_id = self.tensors[x].shape_id;
            let dtype = self.tensors[x].dtype;
            let (_node_id, cid) = self.push_node(graph_id, Node::Contiguous { x: class_id }, shape_id, dtype);
            let tid = self.new_graph_tensor(graph_id, cid, shape_id, dtype);
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, nid={_node_id:?}, cid={cid:?}");
            Ok(tid)
        } else if self.buffer_map.contains_key(&x) {
            // Already realized: the tensor is a load from its own contiguous
            // buffer, so this is a no-op. Mirror reshape's already-resized path.
            self.retain(x);
            Ok(x)
        } else {
            self.add_store(x)?;
            self.retain(x);
            Ok(x)
        }
    }

    pub fn reduce(&mut self, x: TensorId, mut axes: Vec<UAxis>, rop: BOp) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::reduce(x={x}, axes={axes:?}, rop={rop:?})");
        let dtype = self.tensors[x].dtype;
        let shape = self.shape(x).to_vec();
        axes.sort_unstable();
        let reduce_shape = crate::shape::reduce(&shape, &axes);
        let shape_id = self.push_shape(reduce_shape);

        if self.is_graph(x) {
            let (class_id, graph_id) = self.graph_ids(x);
            let (_node_id, class_id) =
                self.push_node(graph_id, Node::Reduce { x: class_id, rop, axes: axes.into_boxed_slice() }, shape_id, dtype);
            let tid = self.new_graph_tensor(graph_id, class_id, shape_id, dtype);
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, nid={_node_id:?}, cid={class_id:?}");
            Ok(tid)
        } else {
            let (kid, mut op_id) = self.duplicate_or_store(x, false)?;

            let n = shape.len();
            let max_axis = *axes.last().unwrap();
            let mut ai = 0;
            let mut permute_axes = Vec::with_capacity(n);
            for i in 0..=max_axis {
                if axes[ai] == i {
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
                kernel_id: kid,
                op_id,
                depends_on: KernelId::NULL,
                class_id: ClassId::NULL,
                graph_id: GraphId::NULL,
                rc: 1,
            });

            debug_assert_eq!(self.kernels[kid].outputs.len(), 0, "input into reduce must have empty outputs");
            self.kernels[kid].outputs.push(tid);

            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, kid={kid:?}, op_id={op_id:?}");
            Ok(tid)
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

        if self.is_graph(x) {
            let (class_id, graph_id) = self.graph_ids(x);
            let (_, class_id) = self.push_node(graph_id, Node::Reshape { x: class_id, shape: shape_id }, shape_id, dtype);
            self.new_graph_tensor(graph_id, class_id, shape_id, dtype)
        } else {
            // If x is realized, create a load kernel with the target shape.
            // The result shares x's buffer (set in buffer_map), so add_store
            // won't add a StoreView for it. This avoids copying data for a
            // view-only reshape.
            if let Some(&buf_id) = self.buffer_map.get(&x) {
                let scope = if self.is_variable_tensor(x) {
                    MemScope::Variable
                } else {
                    MemScope::Global
                };
                let mut kernel = Kernel::new(DeviceId::AUTO);
                let op_id = kernel.define(dtype, scope, true, &shape);
                let kernel_id =
                    self.kernels.push(KernelData { outputs: Vec::new(), loads: Vec::new(), stores: Vec::new(), kernel });
                let tid = self.tensors.push(TensorData {
                    shape_id,
                    dtype,
                    kernel_id,
                    op_id,
                    depends_on: KernelId::NULL,
                    class_id: ClassId::NULL,
                    graph_id: GraphId::NULL,
                    rc: 1,
                });
                self.kernels[kernel_id].outputs.push(tid);
                self.kernels[kernel_id].loads.push(tid);
                self.retain_load(tid);
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
                kernel_id: kernel_id_dup,
                op_id,
                depends_on: KernelId::NULL,
                class_id: ClassId::NULL,
                graph_id: GraphId::NULL,
                rc: 1,
            });

            debug_assert_eq!(self.kernels[kernel_id_dup].outputs.len(), 0, "input into reshape must have empty outputs");
            self.kernels[kernel_id_dup].outputs.push(tid);

            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, kid={kernel_id_dup:?}, op_id={op_id:?}");
            tid
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

        if self.is_graph(x) {
            let (class_id, graph_id) = self.graph_ids(x);
            let (_, class_id) = self.push_node(graph_id, Node::Expand { x: class_id, shape: shape_id }, shape_id, dtype);
            Ok(self.new_graph_tensor(graph_id, class_id, shape_id, dtype))
        } else {
            let (kernel_id, op_id) = self.eager_ids(x);
            let force_store = self.kernels[kernel_id].kernel.is_preceded_by_compute(op_id);
            let (kernel_id, op_id) = self.duplicate_or_store(x, force_store)?;

            let op_id = self.kernels[kernel_id].kernel.expand(op_id, &self.shapes[shape_id]);
            let tid = self.tensors.push(TensorData {
                shape_id,
                dtype,
                kernel_id,
                op_id,
                depends_on: KernelId::NULL,
                class_id: ClassId::NULL,
                graph_id: GraphId::NULL,
                rc: 1,
            });

            debug_assert_eq!(self.kernels[kernel_id].outputs.len(), 0, "input into expand must have empty outputs");
            self.kernels[kernel_id].outputs.push(tid);

            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
            Ok(tid)
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

        if self.is_graph(x) {
            let (class_id, graph_id) = self.graph_ids(x);
            let (_, class_id) =
                self.push_node(graph_id, Node::Permute { x: class_id, axes: axes.into_boxed_slice() }, shape_id, dtype);
            self.new_graph_tensor(graph_id, class_id, shape_id, dtype)
        } else {
            let (kernel_id, op_id) = self.duplicate_or_store(x, false).unwrap();
            let op_id = self.kernels[kernel_id]
                .kernel
                .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Permute { axes, shape: new_shape }) });
            let tid = self.tensors.push(TensorData {
                shape_id,
                dtype,
                kernel_id,
                op_id,
                depends_on: KernelId::NULL,
                class_id: ClassId::NULL,
                graph_id: GraphId::NULL,
                rc: 1,
            });
            debug_assert_eq!(self.kernels[kernel_id].outputs.len(), 0, "input into permute must have empty outputs");
            self.kernels[kernel_id].outputs.push(tid);
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
            tid
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

        if self.is_graph(x) {
            let (class_id, graph_id) = self.graph_ids(x);
            let (_, class_id) =
                self.push_node(graph_id, Node::PadZeros { x: class_id, padding: padding.into_boxed_slice() }, shape_id, dtype);
            self.new_graph_tensor(graph_id, class_id, shape_id, dtype)
        } else {
            let (kernel_id, op_id) = self.eager_ids(x);
            let force_store = pad_n > child_n && self.kernels[kernel_id].kernel.is_preceded_by_compute(op_id);
            let (kernel_id, op_id) = self.duplicate_or_store(x, force_store).unwrap();
            let op_id = self.kernels[kernel_id]
                .kernel
                .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Pad { padding, shape: new_shape }) });
            let tid = self.tensors.push(TensorData {
                shape_id,
                dtype,
                kernel_id,
                op_id,
                depends_on: KernelId::NULL,
                class_id: ClassId::NULL,
                graph_id: GraphId::NULL,
                rc: 1,
            });
            debug_assert_eq!(self.kernels[kernel_id].outputs.len(), 0, "input into pad must have empty outputs");
            self.kernels[kernel_id].outputs.push(tid);
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
            tid
        }
    }

    /// Narrow
    pub fn narrow(&mut self, x: TensorId, axis: UAxis, start: Dim, len: Dim) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::slice(x={x}, axis={axis}, start={start}, len={len})");

        let sh = self.shape(x);
        debug_assert!(axis < sh.len() as UAxis, "slice: axis {axis} out of range for rank {}", sh.len());

        let mut new_shape = sh.to_vec();
        new_shape[axis] = len;
        let shape_id = self.push_shape(new_shape.clone());
        let dtype = self.tensors[x].dtype;

        if self.is_graph(x) {
            let (class_id, graph_id) = self.graph_ids(x);
            let start_const = Constant::idx(start);
            let scalar_shape = self.push_shape(vec![1]);
            let (_, start_cid) = self.push_node(graph_id, Node::Const(start_const), scalar_shape, start_const.dtype());
            let (_, class_id) =
                self.push_node(graph_id, Node::Narrow { x: class_id, axis, start: OpId(start_cid.0), len }, shape_id, dtype);
            self.new_graph_tensor(graph_id, class_id, shape_id, dtype)
        } else {
            // Create the tensor and store the start variable
            let (kernel_id, x) = self.duplicate_or_store(x, false).unwrap();
            debug_assert_eq!(self.kernels[kernel_id].outputs.len(), 0, "input into slice must have empty outputs");

            let MemoryPool::Host(ref mut pool) = self.pools[PoolId::HOST] else {
                unreachable!("Host must exist.")
            };
            let buffer_id = BufferId { pool: PoolId::HOST, buffer: pool.store_variable(Constant::idx(start)) };

            let start_op_id = self.kernels[kernel_id].kernel.define(IDX_T, MemScope::Variable, true, &[1]);

            // Tensor for start variable
            let tid = self.tensors.push(TensorData {
                shape_id,
                dtype: IDX_T,
                kernel_id,
                op_id: start_op_id,
                depends_on: KernelId::NULL,
                class_id: ClassId::NULL,
                graph_id: GraphId::NULL,
                rc: 1,
            });
            self.buffer_map.insert(tid, buffer_id);
            self.kernels[kernel_id].loads.push(tid);

            // Move op
            let op_id = self.kernels[kernel_id]
                .kernel
                .push_back(Op::Move { x, mop: Box::new(MoveOp::Narrow { axis, start: start_op_id, len }) });
            let tid = self.tensors.push(TensorData {
                shape_id,
                dtype,
                kernel_id,
                op_id,
                depends_on: KernelId::NULL,
                class_id: ClassId::NULL,
                graph_id: GraphId::NULL,
                rc: 1,
            });
            self.kernels[kernel_id].outputs.push(tid);
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
            tid
        }
    }

    /// Flip tensor along axes.
    ///
    /// # Errors
    /// Returns shape error if the axes list is empty.
    pub fn flip(&mut self, x: TensorId, mut axes: Vec<UAxis>) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::flip(x={x}, axes={axes:?})");

        let sh = self.shape(x);
        if axes.is_empty() {
            return Err(ZyxError::shape_error(format!("flip: axes must not be empty for tensor of shape {sh:?}").into()));
        }
        for &axis in &axes {
            if axis >= sh.len() {
                return Err(ZyxError::shape_error(format!("Axis {axis} is out of range of rank {}", sh.len()).into()));
            }
        }
        axes.sort_unstable();
        axes.dedup();

        let shape_id = self.push_shape(sh.to_vec());
        let dtype = self.tensors[x].dtype;

        if self.is_graph(x) {
            let (class_id, graph_id) = self.graph_ids(x);
            let (_, class_id) =
                self.push_node(graph_id, Node::Flip { x: class_id, axes: axes.into_boxed_slice() }, shape_id, dtype);
            let tid = self.new_graph_tensor(graph_id, class_id, shape_id, dtype);
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, cid={class_id:?}");
            Ok(tid)
        } else {
            let (kernel_id, op_id) = self.duplicate_or_store(x, false).unwrap();
            let op_id = self.kernels[kernel_id].kernel.flip(op_id, &axes);
            let tid = self.tensors.push(TensorData {
                shape_id,
                dtype,
                kernel_id,
                op_id,
                depends_on: KernelId::NULL,
                class_id: ClassId::NULL,
                graph_id: GraphId::NULL,
                rc: 1,
            });
            debug_assert_eq!(self.kernels[kernel_id].outputs.len(), 0, "input into flip must have empty outputs");
            self.kernels[kernel_id].outputs.push(tid);
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
            Ok(tid)
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
        let Some(mut buffer_id) = self.buffer_map.get(&x).copied() else {
            let this = &mut *self;
            this.initialize_backends();
            let kid = if this.is_graph(x) {
                return Err(ZyxError::graph_tensor_not_realized(x));
            } else {
                this.eager_ids(x).0
            };
            let seen: Set<TensorId> = this.kernels[kid].outputs.iter().copied().collect();
            for tid in seen {
                this.add_store(tid)?;
            }
            let bytes = (data.len() * T::bit_size() as usize).div_ceil(8);
            let byte_slice = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), bytes) };
            let buffer_id = this.buffer_map[&x];
            for buffers in this.events.keys() {
                if buffers.contains(&buffer_id) {
                    let buffers = buffers.clone();
                    let event = this.events.remove(&buffers).unwrap();
                    this.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, vec![event])?;
                    #[cfg(feature = "debug_tensor_op")]
                    println!("  -> x={x}, {:?}", self.tensors[x]);
                    return Ok(());
                }
            }
            this.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, Vec::new())?;
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> x={x}, {:?}", self.tensors[x]);
            return Ok(());
        };

        // A store may still be pending on this tensor (assign wrote into
        // this buffer in place). Run the pending producer kernel first so
        // the buffer is up to date, then re-fetch the buffer id (the store
        // may have moved it to a device pool).
        if !self.tensors[x].depends_on.is_null() {
            let kid = self.tensors[x].depends_on;
            let seen: Set<TensorId> = self.kernels[kid].outputs.iter().copied().collect();
            for tid in seen {
                self.add_store(tid)?;
            }
            buffer_id = self.buffer_map.get(&x).copied().ok_or_else(|| {
                ZyxError::AllocationError(format!("load: tensor {x} lost its buffer during pending store").into())
            })?;
        }
        let bytes = (data.len() * T::bit_size() as usize).div_ceil(8);
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
        Ok(())
    }

    /// Assigns the value of `src` to `dst` in-place using StoreView in the kernel IR.
    ///
    /// A StoreView is added to `src`'s kernel that writes into `dst`'s
    /// existing buffer. Materialization happens naturally when `src`'s
    /// kernel is released.
    ///
    /// # Errors
    ///
    /// Returns [`ZyxError::DTypeError`] if the dtypes do not match.
    ///
    /// Returns [`ZyxError::ShapeError`] if the shapes do not match.
    ///
    /// Returns [`ZyxError::GraphTensorNotRealized`] if `dst` is a
    /// graph tensor that has not been realized yet.
    pub fn assign(&mut self, dst: TensorId, src: TensorId) -> Result<(), ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::assign(dst={dst}, src={src})");

        let dst_dtype = self.tensors[dst].dtype;
        let src_dtype = self.tensors[src].dtype;
        if dst_dtype != src_dtype {
            return Err(ZyxError::DTypeError(format!("assign dtype mismatch: dst={dst_dtype}, src={src_dtype}").into()));
        }
        let dst_shape = self.shape(dst);
        let src_shape = self.shape(src);
        if dst_shape != src_shape {
            return Err(ZyxError::shape_error(format!("assign shape mismatch: dst={dst_shape:?}, src={src_shape:?}").into()));
        }
        if self.is_graph(dst) {
            // Graph-mode in-place assign: record a Node::Assign inside the tape
            // graph. The plan writes src's value into dst's buffer in-place; dst
            // is either a realized (promoted) leaf tensor or a movement view over
            // one (e.g. a slice), whose movement chain the kernelizer replays
            // into src's kernel so the store lands at the view's position.
            if dst == src {
                return Err(ZyxError::ShapeError("assign: dst equals src (self-assign)".into()));
            }
            let (dst_cid, graph_id) = self.graph_ids(dst);
            self.assert_graph_alive(graph_id);
            if self.is_graph(src) {
                let (_, src_graph_id) = self.graph_ids(src);
                if src_graph_id != graph_id {
                    panic!("tensor belongs to a different tape scope");
                }
            } else {
                self.promote_to_graph(src, graph_id)?;
            }
            let mut dst_define_cid = dst_cid;
            // Walk graph to find the source of the lvalue
            let graph = &self.graphs[graph_id];
            loop {
                match graph.nodes[graph.classes[dst_define_cid].nodes[0]].node {
                    Node::PadZeros { x, .. }
                    | Node::Flip { x, .. }
                    | Node::Expand { x, .. }
                    | Node::Reshape { x, .. }
                    | Node::Permute { x, .. } => dst_define_cid = x,
                    Node::After { .. } | Node::Leaf { .. } => break,
                    _ => unreachable!(),
                }
            }
            // Resolve the base leaf through any After chain (a previous assign on
            // the same buffer) to find the base tensor. The After for this assign
            // threads onto the previous After, not the original buffer.
            let mut leaf_cid = dst_define_cid;
            while let Node::After { x, .. } = &graph.nodes[graph.classes[leaf_cid].nodes[0]].node {
                leaf_cid = *x;
            }
            let dst_define = graph.leaf_map[&leaf_cid];

            // The Assign node keeps the ORIGINAL dst-chain and src classes; the
            // output class cid is what any later use of dst or src resolves to,
            // so both tensors are re-pointed at it.
            let src_cid = self.graph_ids(src).0;
            let shape_id = self.tensors[dst].shape_id;
            let dtype = self.tensors[dst].dtype;
            let define_shape_id = self.tensors[dst_define].shape_id;
            let define_dtype = self.tensors[dst_define].dtype;
            let (_node_id, assign_cid) = self.push_node(graph_id, Node::Assign { dst: dst_cid, src: src_cid }, shape_id, dtype);
            self.tensors[dst_define].class_id =
                self.push_node(graph_id, Node::After { x: dst_define_cid, dep: assign_cid }, define_shape_id, define_dtype).1;
            self.tensors[dst].class_id = self.push_node(graph_id, Node::After { x: dst_cid, dep: assign_cid }, shape_id, dtype).1;
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> cid={cid:?}");
            return Ok(());
        }
        // Merge dst's (movement-only) kernel into src's kernel, then store src's
        // value into dst's base buffer in-place.
        let (src_kid, src_op) = self.eager_ids(src);
        let (dst_kid, dst_op) = self.eager_ids(dst);
        // The destination must be a movement-only view kernel with no outputs
        // other than dst itself (dst may appear multiple times, once per
        // cloned handle).
        if self.kernels[dst_kid].outputs.iter().any(|&e| e != dst) {
            return Err(ZyxError::ShapeError(
                format!("assign: dst kernel {dst_kid:?} has other outputs {:?}, only dst allowed", self.kernels[dst_kid].outputs)
                    .into(),
            ));
        }
        for op in self.kernels[dst_kid].kernel.ops.values() {
            if !matches!(op.op, Op::Define { .. } | Op::Move { .. } | Op::Const(_)) {
                return Err(ZyxError::ShapeError(
                    format!("assign: dst kernel {dst_kid:?} has unsupported op {:?}, only movement ops allowed", op.op).into(),
                ));
            }
        }
        if src_kid == dst_kid {
            return Err(ZyxError::ShapeError(
                format!("assign: src and dst share kernel {dst_kid:?}; dst must be a separate movement-only kernel").into(),
            ));
        }
        if !self.kernels[dst_kid].stores.is_empty() {
            return Err(ZyxError::ShapeError(
                format!("assign: dst kernel {dst_kid:?} has stores {}; expected none", self.kernels[dst_kid].stores.len()).into(),
            ));
        }
        if self.kernels[src_kid].loads.contains(&dst) {
            return Err(ZyxError::ShapeError(
                format!("assign: src kernel {dst_kid:?} loads dst tensor, not allowed to avoid data races").into(),
            ));
        }
        if self.kernels[dst_kid].loads.len() > 1 {
            return Err(ZyxError::ShapeError(format!("assign: dst kernel {dst_kid:?} must have exactly 1 load").into()));
        }

        // Remove the dst (movement-only) kernel; its base buffer is dst_org.
        // The removed kernel held a kernel-load reference on dst_org.
        let KernelData { kernel, loads, .. } = unsafe { self.kernels.remove_and_return(dst_kid) };
        let dst_org = loads[0];
        self.release_load(dst_org);

        let mut dst_define = dst_op;
        for _ in 0..100 {
            match kernel.ops[dst_define].op {
                Op::Move { x, .. } => {
                    dst_define = x;
                }
                Op::Define { .. } => {
                    break;
                }
                _ => {}
            }
        }

        // Replay dst's movement chain into src's kernel. The replayed base
        // define becomes the (mutable) store target; the last replayed
        // movement op yields dst's final value within src's kernel.
        let mut op_map = Map::default();
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            match kernel.ops[op_id].op {
                Op::Const(value) => {
                    let id = self.kernels[src_kid].kernel.push_back(Op::Const(value));
                    op_map.insert(op_id, id);
                }
                Op::Define { dtype, scope, mut ro, ref shape } => {
                    if op_id == dst_define {
                        ro = false;
                    }
                    let id = self.kernels[src_kid].kernel.push_back(Op::Define { dtype, scope, ro, shape: shape.clone() });
                    op_map.insert(op_id, id);
                }
                Op::Move { x, ref mop } => {
                    let x = if let Some(x) = op_map.get(&x) {
                        *x
                    } else {
                        // this is the move on the load
                        op_map[&dst_define]
                    };
                    let id = self.kernels[src_kid].kernel.push_back(Op::Move { x, mop: mop.clone() });
                    op_map.insert(op_id, id);
                }
                _ => unreachable!("should've already returned error"),
            }
            op_id = kernel.next_op(op_id);
        }

        let dst_op = op_map.get(&dst_op).copied().unwrap_or(op_map[&dst_define]);
        // Store src's value into dst's base buffer through the replayed chain.
        self.kernels[src_kid].kernel.store(dst_op, src_op, OpId::NULL, MemLayout::Scalar);
        self.kernels[src_kid].stores.push(dst_org);

        // The store writes IN PLACE into dst_org's existing buffer, which
        // stays resident in buffer_map — no new buffer is allocated for the
        // destination. If dst owns that buffer (dst == dst_org), dst stays
        // valid and is marked pending on src_kid so a read of dst runs the
        // store first. If dst is a movement view (dst != dst_org) it has no
        // buffer and is invalid after the in-place write, so run the store
        // right away and invalidate dst.
        if dst == dst_org {
            // dst owns the target buffer. Keep it valid but pending on the
            // store kernel so a read runs the in-place write first. Re-point
            // dst onto its own load kernel (as add_store does) so clone
            // drops / releases target a live kernel instead of the store
            // kernel that gets consumed on materialization.
            self.tensors[dst].depends_on = src_kid;
            self.kernels[src_kid].outputs.push(dst);
            self.tensors[dst].kernel_id = src_kid;
            self.tensors[dst].op_id = dst_op;
            self.add_store(dst)?;
        } else {
            self.tensors[dst].kernel_id = KernelId::NULL;
            self.tensors[dst].op_id = OpId::NULL;
            self.tensors[dst].depends_on = KernelId::NULL;
            let seen: Set<TensorId> = self.kernels[src_kid].outputs.iter().copied().collect();
            for tid in seen {
                self.add_store(tid)?;
            }
        }

        Ok(())
    }

    /// Initializes all available devices, creating a device for each compute
    /// device and a memory pool for each physical memory.
    /// Does nothing if devices were already initialized.
    pub fn initialize_backends(&mut self) {
        if !self.pools.is_empty() {
            return;
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

        crate::backend::initialize_backends(&config, &mut self.pools, &mut self.devices, self.debug.dev());

        self.autotune_config = config.autotune;
        //println!("INIT runtime");
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
    pub fn free_memory(&mut self) -> Dim {
        if self.pools.is_empty() {
            self.initialize_backends();
        }
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
        let (mut kid, mut op_id) = self.eager_ids(x);

        let contains_stores = self.kernels[kid].kernel.contains_stores();
        let preceded_by_reduce = self.kernels[kid].kernel.is_preceded_by_reduce(op_id);
        if force_store || contains_stores | preceded_by_reduce {
            self.add_store(x)?;
            (kid, op_id) = self.eager_ids(x);
            // We need to duplicate the new load kernel too, which we do below
        }

        debug_assert!(self.kernels[kid].stores.is_empty(), "duplicated kernel must not have stores");

        let old_loads = self.kernels[kid].loads.clone();
        let out_op_ids: Vec<OpId> = self.kernels[kid].outputs.iter().map(|&tid| self.tensors[tid].op_id).collect();
        let (kernel, new_op_id, self_loads, new_loads) =
            self.kernels[kid].kernel.extract_subkernel(op_id, &out_op_ids, &old_loads);
        self.kernels[kid].loads = self_loads.clone();

        // Each kernel-load occurrence carries its own rc reference. The split
        // may duplicate a load into both kernels (an extra ref) or drop it
        // (release the ref).
        let mut seen: Set<TensorId> = Set::default();
        for &tid in old_loads.iter().chain(self_loads.iter()).chain(new_loads.iter()) {
            if !seen.insert(tid) {
                continue;
            }
            let old_c = old_loads.iter().filter(|&&t| t == tid).count();
            let self_c = self_loads.iter().filter(|&&t| t == tid).count();
            let new_c = self_c + new_loads.iter().filter(|&&t| t == tid).count();
            let delta = (new_c as i64) - (old_c as i64);
            for _ in 0..delta {
                self.retain_load(tid);
            }
            for _ in 0..(-delta) {
                self.release_load(tid);
            }
        }

        kid = self.kernels.push(KernelData { outputs: Vec::new(), loads: new_loads, stores: Vec::new(), kernel });
        op_id = new_op_id;

        Ok((kid, op_id))
    }

    pub fn add_store(&mut self, x: TensorId) -> Result<(), ZyxError> {
        let (kid, op_id, pending) = {
            let (kid, op_id) = self.eager_ids(x);
            (kid, op_id, self.tensors[x].depends_on)
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
            let shape = &self.shapes[self.tensors[x].shape_id];
            let dst_id = self.kernels[kid].kernel.define(dtype, MemScope::Global, false, shape);
            self.kernels[kid].kernel.store(dst_id, op_id, OpId::NULL, MemLayout::Scalar);
            self.kernels[kid].stores.push(x);
            kid
        } else {
            pending
        };

        let outputs_empty = self.kernels[kid].outputs.is_empty();

        // Create load kernel so the tensor remains usable (visited must point to a live kernel)
        let dtype = self.tensors[x].dtype;
        let scope = if self.is_variable_tensor(x) {
            MemScope::Variable
        } else {
            MemScope::Global
        };
        let shape = self.shape(x);
        let mut kernel = Kernel::new(DeviceId::AUTO);
        let load_op_id = kernel.define(dtype, scope, true, shape);
        let load_kid = self.kernels.push(KernelData { outputs: vec![x; count], loads: vec![x], stores: Vec::new(), kernel });
        self.tensors[x].kernel_id = load_kid;
        self.tensors[x].op_id = load_op_id;
        self.tensors[x].depends_on = pending;
        self.retain_load(x);

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
        buffers: &[PoolBufferId],
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

        if self.debug.sched() {
            kernel.debug();
        }

        kernel.linearize();
        kernel.common_subexpression_elimination();
        kernel.dead_code_elimination();
        kernel.instruction_schedule();

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
        {
            let n_global_defines = kernel
                .ops
                .values()
                .filter(|op| matches!(&op.op, Op::Define { scope: MemScope::Global | MemScope::Variable, .. }))
                .count();
            let n_buffers = buffers.iter().filter(|&&b| b != PoolBufferId::NULL).count();
            assert!(
                n_buffers <= n_global_defines,
                "buffers len ({}) must not exceed number of global/scalar defines ({}) in kernel",
                n_buffers,
                n_global_defines,
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
            buffers,
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

        for &tid in &loads {
            assert!(
                self.buffer_map.contains_key(&tid)
                    || outputs.contains(&tid)
                    || self.kernels.values().any(|kd| kd.outputs.contains(&tid) || kd.stores.contains(&tid)),
                "load tid {tid} not realized, not in outputs, not in any kernel; kernels loading it: {:?}",
                self.kernels.iter().filter(|(_, kd)| kd.loads.contains(&tid)).map(|(k, _)| k).collect::<Vec<_>>(),
            );
        }

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
            let pending = if self.tensors[load].class_id.is_null() {
                self.tensors[load].depends_on
            } else {
                KernelId::NULL
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
        self.initialize_backends();

        // If stores already have buffers (e.g. assign writes in-place), a
        // kernel can only touch memory of one pool, so those buffers dictate
        // the pool — and hence the device. Stores spanning multiple pools is
        // an error. Without existing store buffers (or if no device shares
        // their pool), fall back to the freest device and move the buffers.
        let mut store_pools: BTreeSet<PoolId> = BTreeSet::new();
        for &tid in &stores {
            if let Some(buf_id) = self.buffer_map.get(&tid) {
                store_pools.insert(buf_id.pool);
            }
        }
        let (dev_id, pool_id) = if store_pools.len() == 1 {
            let pool_id = *store_pools.iter().next().unwrap();
            let dev_id = self.devices.ids().find(|&dev_id| self.devices[dev_id].memory_pool_id() == pool_id);
            match dev_id {
                Some(dev_id) => (dev_id, pool_id),
                None => {
                    let mut dev_ids: Vec<DeviceId> = self.devices.ids().collect();
                    dev_ids.sort_unstable_by_key(|&dev_id| self.devices[dev_id].free_compute());
                    dev_ids.reverse();
                    let dev_id = *dev_ids.first().ok_or_else(|| ZyxError::AllocationError("no available device".into()))?;
                    (dev_id, self.devices[dev_id].memory_pool_id())
                }
            }
        } else if store_pools.is_empty() {
            let mut dev_ids: Vec<DeviceId> = self.devices.ids().collect();
            dev_ids.sort_unstable_by_key(|&dev_id| self.devices[dev_id].free_compute());
            dev_ids.reverse();
            let dev_id = *dev_ids.first().ok_or_else(|| ZyxError::AllocationError("no available device".into()))?;
            (dev_id, self.devices[dev_id].memory_pool_id())
        } else {
            return Err(ZyxError::AllocationError(
                format!("stores span multiple pools {store_pools:?}; a kernel can only touch memory of a single pool").into(),
            ));
        };
        kernel.device_id = dev_id;

        // Ensure loads are in target pool
        let mut event_wait_list = Vec::new();
        for &tid in &loads {
            let buf_id = self.buffer_map[&tid];
            if buf_id.pool != pool_id {
                let src = buf_id.buffer;
                if let Some(constant) = self.pools[buf_id.pool].get_variable(src) {
                    let dst = self.pools[pool_id].store_variable(constant);
                    self.buffer_map.remove(&tid);
                    self.buffer_map.insert(tid, BufferId { pool: pool_id, buffer: dst });
                    continue;
                }
                let bytes = (self.shape(tid).iter().product::<Dim>() as usize * self.dtype(tid).bit_size() as usize).div_ceil(8);
                let alloc_bytes = bytes + self.dtype(tid).bit_size() as usize / 8;

                // Gather the events that the source buffer depends on (prior
                // writers), so the copy waits for them.
                let mut wait_list = Vec::new();
                for buffers in self.events.keys() {
                    if buffers.contains(&buf_id) {
                        let buffers = buffers.clone();
                        let event = self.events.remove(&buffers).unwrap();
                        wait_list.push(event);
                        break;
                    }
                }

                let (dst, alloc_ev) = self.pools[pool_id].allocate(alloc_bytes as Dim)?;
                let dst_global = BufferId { pool: pool_id, buffer: dst };
                debug_assert_ne!(buf_id.pool, pool_id, "pool_to_pool across the same pool is disallowed");
                let src_pool_ptr: *mut MemoryPool = &mut self.pools[buf_id.pool];
                let copy_ev = self.pools[pool_id].pool_to_pool(unsafe { &mut *src_pool_ptr }, src, dst, {
                    wait_list.push(alloc_ev);
                    wait_list
                })?;
                self.pools[pool_id].sync_events(vec![copy_ev])?;

                // Remove and deallocate the old buffer only AFTER pool_to_pool
                // has finished reading it.
                self.buffer_map.remove(&tid);
                if !self.buffer_map.values().any(|b| b.buffer == src) {
                    self.pools[buf_id.pool].deallocate(src, vec![]);
                }
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

        // Ensure stores are in target pool (assign writes in-place into an
        // existing buffer, which may live in a different pool).
        for &tid in &stores {
            let Some(buf_id) = self.buffer_map.get(&tid).copied() else {
                continue;
            };
            if buf_id.pool != pool_id {
                let src = buf_id.buffer;
                let bytes = (self.shape(tid).iter().product::<Dim>() as usize * self.dtype(tid).bit_size() as usize).div_ceil(8);
                let alloc_bytes = bytes as Dim + Dim::from(self.dtype(tid).bit_size() / 8);
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
                if !self.buffer_map.values().any(|b| b.buffer == src) {
                    self.pools[buf_id.pool].deallocate(src, vec![]);
                }

                let (dst, event) = self.pools[pool_id].allocate(alloc_bytes)?;
                let dst_global = BufferId { pool: pool_id, buffer: dst };
                let event = self.pools[pool_id].host_to_pool(&byte_slice, dst, vec![event])?;
                self.pools[pool_id].sync_events(vec![event])?;
                self.buffer_map.insert(tid, dst_global);
            }
        }

        // Collect existing store buffers for already-realized store tensors,
        // allocate new buffers for the rest.
        let mut kernel_buffers = BTreeSet::new();
        for &tid in &loads {
            kernel_buffers.insert(self.buffer_map[&tid]);
        }
        for &tid in &stores {
            if let Some(&buf_id) = self.buffer_map.get(&tid) {
                kernel_buffers.insert(buf_id);
                self.tensors[tid].depends_on = KernelId::NULL;
            } else {
                let bytes = (self.shape(tid).iter().product::<Dim>() as usize * self.dtype(tid).bit_size() as usize).div_ceil(8);
                let alloc_bytes = bytes as Dim + Dim::from(self.dtype(tid).bit_size() / 8);
                let (buf, event) = self.pools[pool_id].allocate(alloc_bytes)?;
                let global_id = BufferId { pool: pool_id, buffer: buf };
                self.buffer_map.insert(tid, global_id);
                self.tensors[tid].depends_on = KernelId::NULL;
                kernel_buffers.insert(global_id);
                event_wait_list.push(event);
            }
        }

        // Build buffers: load buffers first, then store buffers
        let mut buffers: Vec<PoolBufferId> = Vec::new();
        for &tid in &loads {
            buffers.push(self.buffer_map[&tid].buffer);
        }
        for &tid in &stores {
            buffers.push(self.buffer_map[&tid].buffer);
        }

        // Compile and launch (caches in kernel_map / programs)
        let (flop, read, write) = kernel.flop_mem_rw();
        let (dev_prog, _timing) = self.get_or_autotune(kernel, pool_id, flop, read, write, &buffers)?;

        let event = self.devices[dev_id].launch(dev_prog, &mut self.pools[pool_id], &buffers, event_wait_list)?;
        self.events.insert(kernel_buffers, event);

        // The kernel has consumed its loads. Release the load references so
        // dead load tensors and their buffers are reclaimed. Buffers still in
        // use keep rc > 0 via other kernels' load references or handles.
        for &tid in &loads {
            self.release_load(tid);
        }

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
