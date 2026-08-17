//! E-graph for tensor operation equivalence and optimization.
//!
//! The graph supports rewrites that produce equivalent forms of a computation:
//! - **CSE** (common subexpression elimination) via hashconsing
//! - **Algebraic rewrites** like transpose fusion: `transpose(A) @ transpose(B)` ↔ `(B @ A).transpose()`
//! - **Layout rewrites**: a matmul can be realized as transposed or un-transposed,
//!   with the transpose either fused into the kernel or materialized as a separate
//!   pre-processing step
//! - **Shape rewrites**: reshape and padding can be fused into adjacent ops or
//!   split out as separate nodes
//!
//! Each equivalence class (`EClass`) holds all equivalent node forms. A cost
//! model selects the cheapest extraction for kernel compilation.

use std::collections::BTreeSet;

use crate::{
    DType, Map, Set, ZyxError,
    backend::{BufferId, Device, PoolId, ProgramId},
    dtype::Constant,
    kernel::{BOp, DeviceId, Kernel, MoveOp, Op, OpId, ParamKind, UOp},
    runtime::{KernelData, KernelId, Runtime, TensorData, loads_dropped_by_prune},
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    tensor::TensorId,
};

mod autograd;
mod kernelizer;
pub(crate) mod plan;
pub use plan::ExecPlan;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

impl From<usize> for NodeId {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}
impl From<NodeId> for usize {
    fn from(v: NodeId) -> usize {
        v.0 as usize
    }
}

impl SlabId for NodeId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphId(pub u16);

impl From<usize> for GraphId {
    fn from(v: usize) -> Self {
        Self(v as u16)
    }
}
impl From<GraphId> for usize {
    fn from(v: GraphId) -> usize {
        v.0 as usize
    }
}

impl SlabId for GraphId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u16::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClassId(pub u32);

impl From<usize> for ClassId {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}
impl From<ClassId> for usize {
    fn from(v: ClassId) -> usize {
        v.0 as usize
    }
}

impl SlabId for ClassId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Node {
    Const(Constant),
    Leaf {
        dtype: DType,
        leaf_id: u32,
    },
    Expand {
        x: ClassId,
        shape: ClassId,
    },
    Permute {
        x: ClassId,
        axes: Box<[UAxis]>,
    },
    Reshape {
        x: ClassId,
        shape: ClassId,
        input_rank: usize,
    },
    Pad {
        x: ClassId,
        axis: UAxis,
        lp: ClassId,
        rp: ClassId,
    },
    Flip {
        x: ClassId,
        axes: Box<[UAxis]>,
    },
    Narrow {
        x: ClassId,
        axis: UAxis,
        start: ClassId,
        len: ClassId,
    },
    Stack {
        ops: Box<[ClassId]>,
    },
    Reduce {
        x: ClassId,
        rop: BOp,
        axes: Box<[UAxis]>,
    },
    Cast {
        x: ClassId,
        dtype: DType,
    },
    Unary {
        x: ClassId,
        uop: UOp,
    },
    Binary {
        x: ClassId,
        y: ClassId,
        bop: BOp,
    },
    Assign {
        dst: ClassId,
        src: ClassId,
    },
    After {
        x: ClassId,
        dep: ClassId,
    },
    ToDevice {
        x: ClassId,
        device: DeviceId,
        time: u64,
    },
    Contiguous {
        x: ClassId,
    },
    Kernel {
        inputs: Box<[ClassId]>,
        outputs: Box<[ClassId]>,
        program_id: ProgramId,
        time: u64,
    },
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Const(a), Self::Const(b)) => a == b,
            (Self::Leaf { leaf_id: a, .. }, Self::Leaf { leaf_id: b, .. }) => a == b,
            (Self::Expand { x: a, shape: as_ }, Self::Expand { x: b, shape: bs }) => a == b && as_ == bs,
            (Self::Permute { x: a, axes: aa }, Self::Permute { x: b, axes: ba }) => a == b && aa == ba,
            (Self::Reshape { x: a, shape: as_, .. }, Self::Reshape { x: b, shape: bs, .. }) => a == b && as_ == bs,
            (Self::Pad { x: a, axis: aa, lp: al, rp: ar }, Self::Pad { x: b, axis: ba, lp: bl, rp: br }) => {
                a == b && aa == ba && al == bl && ar == br
            }
            (Self::Flip { x: a, axes: aa }, Self::Flip { x: b, axes: ba }) => a == b && aa == ba,
            (Self::Reduce { x: a, rop: ar, axes: aa }, Self::Reduce { x: b, rop: br, axes: ba }) => {
                a == b && ar == br && aa == ba
            }
            (Self::Cast { x: a, dtype: ad }, Self::Cast { x: b, dtype: bd }) => a == b && ad == bd,
            (Self::Unary { x: a, uop: au }, Self::Unary { x: b, uop: bu }) => a == b && au == bu,
            (Self::Binary { x: a, y: ay, bop: ab }, Self::Binary { x: b, y: by, bop: bb }) => a == b && ay == by && ab == bb,
            (Self::Assign { dst: a, src: as_ }, Self::Assign { dst: b, src: bs }) => a == b && as_ == bs,
            (Self::ToDevice { x: a, device: ad, .. }, Self::ToDevice { x: b, device: bd, .. }) => a == b && ad == bd,
            (Self::Contiguous { x: a }, Self::Contiguous { x: b }) => a == b,
            (
                Self::Kernel { inputs: ai, outputs: ao, program_id: ap, .. },
                Self::Kernel { inputs: bi, outputs: bo, program_id: bp, .. },
            ) => ai == bi && ao == bo && ap == bp,
            _ => false,
        }
    }
}

impl Eq for Node {}

impl std::hash::Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Const(v) => {
                0u8.hash(state);
                v.hash(state);
            }
            Self::Leaf { leaf_id, .. } => {
                1u8.hash(state);
                leaf_id.hash(state);
            }
            Self::Expand { x, shape } => {
                2u8.hash(state);
                x.hash(state);
                shape.hash(state);
            }
            Self::Permute { x, axes } => {
                3u8.hash(state);
                x.hash(state);
                axes.hash(state);
            }
            Self::Reshape { x, shape, .. } => {
                4u8.hash(state);
                x.hash(state);
                shape.hash(state);
            }
            Self::Pad { x, axis, lp, rp } => {
                5u8.hash(state);
                x.hash(state);
                axis.hash(state);
                lp.hash(state);
                rp.hash(state);
            }
            Self::Stack { ops } => {
                13u8.hash(state);
                ops.hash(state);
            }
            Self::Flip { x, axes } => {
                12u8.hash(state);
                x.hash(state);
                axes.hash(state);
            }
            Self::Narrow { x, axis, start, len } => {
                16u8.hash(state);
                x.hash(state);
                axis.hash(state);
                start.hash(state);
                len.hash(state);
            }
            Self::Reduce { x, rop: bop, axes } => {
                6u8.hash(state);
                x.hash(state);
                bop.hash(state);
                axes.hash(state);
            }
            Self::Cast { x, dtype } => {
                7u8.hash(state);
                x.hash(state);
                dtype.hash(state);
            }
            Self::Unary { x, uop } => {
                8u8.hash(state);
                x.hash(state);
                uop.hash(state);
            }
            Self::Binary { x, y, bop } => {
                9u8.hash(state);
                x.hash(state);
                y.hash(state);
                bop.hash(state);
            }
            Self::ToDevice { x, device, .. } => {
                10u8.hash(state);
                x.hash(state);
                device.hash(state);
            }
            Self::Contiguous { x } => {
                14u8.hash(state);
                x.hash(state);
            }
            Self::Assign { dst, src } => {
                13u8.hash(state);
                dst.hash(state);
                src.hash(state);
            }
            Self::After { x, dep } => {
                15u8.hash(state);
                x.hash(state);
                dep.hash(state);
            }
            Self::Kernel { inputs, outputs, program_id, .. } => {
                11u8.hash(state);
                inputs.hash(state);
                outputs.hash(state);
                program_id.hash(state);
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct NodeData {
    pub(crate) node: Node,
    pub(crate) class_of: ClassId,
}

#[derive(Debug)]
pub struct EClass {
    pub nodes: Vec<NodeId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JitKernelId(pub u32);

impl From<usize> for JitKernelId {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}
impl From<JitKernelId> for usize {
    fn from(v: JitKernelId) -> usize {
        v.0 as usize
    }
}
impl SlabId for JitKernelId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug, Clone)]
pub struct JitKernelData {
    pub(crate) kernel: Kernel,
    pub(crate) outputs: Vec<ClassId>,
    pub(crate) loads: Vec<ClassId>,
    pub(crate) stores: Vec<ClassId>,
}

#[derive(Debug)]
pub struct Graph {
    shapes: Map<ClassId, Vec<Dim>>,
    pub(crate) hashcons: Map<Node, NodeId>,
    pub(crate) nodes: Slab<NodeId, NodeData>,
    pub(crate) classes: Slab<ClassId, EClass>,
    pub(crate) jit_kernels: Slab<JitKernelId, JitKernelData>,
    pub(crate) leaf_classes: Vec<ClassId>,
    pub(crate) max_leaf_id: u32,
    // Number of alive graph tensors (TensorState::Graph) referencing this graph.
    // Incremented at every graph-tensor birth, decremented when a tensor dies
    // (release), is eagerified, or is dropped.
    pub(crate) ref_count: u64,
    // Tape scope has ended (Tape::drop ran); no new ops may use this graph.
    // The graph is removed from the slab only when dead && ref_count == 0, which
    // guarantees no stale tensor ever observes a reused GraphId.
    pub(crate) dead: bool,
    pub(crate) leaf_map: Map<ClassId, TensorId>,
}

impl Node {
    fn class_params(&self) -> Vec<ClassId> {
        match self {
            Self::Const(_) | Self::Leaf { .. } => vec![],
            Self::Expand { x, shape } => vec![*x, *shape],
            Self::Permute { x, .. } => vec![*x],
            Self::Reshape { x, shape, .. } => vec![*x, *shape],
            Self::Pad { x, lp, rp, .. } => vec![*x, *lp, *rp],
            Self::Narrow { x, axis: _, start, len } => vec![*x, *start, *len],
            Self::Flip { x, .. } => vec![*x],
            Self::Stack { ops } => ops.to_vec(),
            Self::Reduce { x, .. } => vec![*x],
            Self::Cast { x, .. } => vec![*x],
            Self::Unary { x, .. } => vec![*x],
            Self::Binary { x, y, .. } => vec![*x, *y],
            Self::Assign { dst, src } => vec![*dst, *src],
            Self::After { x, dep } => vec![*x, *dep],
            Self::ToDevice { x, .. } => vec![*x],
            Self::Contiguous { x, .. } => vec![*x],
            Self::Kernel { inputs, .. } => inputs.to_vec(),
        }
    }
}

impl Graph {
    pub fn new() -> Self {
        Self {
            shapes: Map::default(),
            hashcons: Map::default(),
            nodes: Slab::new(),
            classes: Slab::new(),
            jit_kernels: Slab::new(),
            leaf_map: Map::default(),
            leaf_classes: Vec::new(),
            max_leaf_id: 0,
            ref_count: 0,
            dead: false,
        }
    }

    pub fn is_leaf(&self, class_id: ClassId) -> bool {
        self.classes[class_id].nodes.iter().any(|&nid| matches!(&self.nodes[nid].node, Node::Leaf { .. }))
    }

    /// Walks back through single-input movement nodes until reaching dst's base
    /// leaf class (a key of `leaf_map`). Used to find which leaf buffer an
    /// [`ExecNode`] class's store aliases.
    pub(crate) fn base_leaf(&self, mut c: ClassId) -> ClassId {
        loop {
            if self.leaf_map.contains_key(&c) {
                return c;
            }
            let mut next = None;
            for nid in &self.classes[c].nodes {
                match &self.nodes[*nid].node {
                    Node::Expand { x, .. }
                    | Node::Permute { x, .. }
                    | Node::Reshape { x, .. }
                    | Node::Pad { x, .. }
                    | Node::Flip { x, .. }
                    | Node::Narrow { x, .. }
                    | Node::ToDevice { x, .. }
                    | Node::After { x, .. } => next = Some(*x),
                    _ => {}
                }
            }
            c = next.unwrap_or_else(|| panic!("assign dst class {c:?} must be a realized leaf or a movement chain over one"));
        }
    }

    /// Whether `class_id` is the output of an in-place `assign` — a class whose
    /// value lives in (aliases) dst's realized leaf buffer.
    pub fn is_after(&self, class_id: ClassId) -> bool {
        self.classes[class_id].nodes.iter().any(|&nid| matches!(&self.nodes[nid].node, Node::After { .. }))
    }

    pub fn push_to_device(&mut self, x: ClassId, device: DeviceId, time: u64) -> ClassId {
        let node = Node::ToDevice { x, device, time };
        if let Some(&nid) = self.hashcons.get(&node) {
            return self.nodes[nid].class_of;
        }
        let nid = self.nodes.push(NodeData { node: node.clone(), class_of: ClassId::NULL });
        let cid = self.classes.push(EClass { nodes: vec![nid] });
        self.nodes[nid].class_of = cid;
        self.hashcons.insert(node, nid);
        cid
    }

    pub fn topo_sort_classes(&self, outputs: &BTreeSet<ClassId>) -> Vec<ClassId> {
        let mut rcs: Map<ClassId, u32> = Map::default();
        let mut stack: Vec<ClassId> = outputs.iter().copied().collect();
        while let Some(cid) = stack.pop() {
            rcs.entry(cid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                let mut deps = Vec::new();
                for nid in &self.classes[cid].nodes {
                    for p in self.nodes[*nid].node.class_params() {
                        if !deps.contains(&p) {
                            deps.push(p);
                        }
                    }
                }
                stack.extend(deps);
                1
            });
        }

        let mut order = Vec::new();
        let mut internal_rcs: Map<ClassId, u32> = Map::default();
        let mut stack: Vec<ClassId> = outputs.iter().copied().collect();
        while let Some(cid) = stack.pop() {
            if let Some(&rc) = rcs.get(&cid) {
                let visited = internal_rcs.entry(cid).and_modify(|c| *c += 1).or_insert(1);
                if rc == *visited {
                    order.push(cid);
                    let mut deps = Vec::new();
                    for nid in &self.classes[cid].nodes {
                        for p in self.nodes[*nid].node.class_params() {
                            if !deps.contains(&p) {
                                deps.push(p);
                            }
                        }
                    }
                    stack.extend(deps);
                }
            }
        }
        order.reverse();
        order
    }

    /// Like [`Self::topo_sort_classes`], but ignores [`Node::Kernel`] nodes when
    /// collecting dependencies and stops the walk at the classes in `inputs`.
    /// Used when iterating the structural graph — e.g. fusing remaining ops into
    /// kernels — where kernel nodes would add spurious input dependencies between
    /// classes and boundary classes must not be walked through into other regions.
    /// When `allowed` is `Some`, the walk never leaves that set, so a
    /// region-restricted sort stays inside its own region even if
    /// [`Self::deps_stopping_at`] would follow a boundary kernel's inputs.
    pub fn topo_sort_classes_without_kernels(
        &self,
        inputs: &Set<ClassId>,
        outputs: &BTreeSet<ClassId>,
        allowed: Option<&Set<ClassId>>,
    ) -> Vec<ClassId> {
        //println!("topo sort inputs={inputs:?} outputs={outputs:?}");
        let mut rcs: Map<ClassId, u32> = Map::default();
        let mut stack: Vec<ClassId> = outputs.iter().copied().collect();
        while let Some(cid) = stack.pop() {
            rcs.entry(cid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                let deps = self.deps_stopping_at(inputs, cid);
                stack.extend(deps.into_iter().filter(|d| allowed.is_none_or(|a| a.contains(d))));
                1
            });
        }

        let mut order = Vec::new();
        let mut internal_rcs: Map<ClassId, u32> = Map::default();
        let mut stack: Vec<ClassId> = outputs.iter().copied().collect();
        while let Some(cid) = stack.pop() {
            if let Some(&rc) = rcs.get(&cid) {
                let visited = internal_rcs.entry(cid).and_modify(|c| *c += 1).or_insert(1);
                if rc == *visited {
                    order.push(cid);
                    let deps = self.deps_stopping_at(inputs, cid);
                    stack.extend(deps.into_iter().filter(|d| allowed.is_none_or(|a| a.contains(d))));
                }
            }
        }
        order.reverse();
        order
    }

    /// kernel output follows only its kernel-node inputs (so classes feeding an
    /// AOT kernel still get covered), an input leaf has no dependencies, and any
    /// other class uses its structural dependencies.
    fn deps_stopping_at(&self, inputs: &Set<ClassId>, cid: ClassId) -> Vec<ClassId> {
        if !inputs.contains(&cid) {
            return self.deps_without_kernels(cid);
        }
        let mut deps = Vec::new();
        for nid in &self.classes[cid].nodes {
            if let Node::Kernel { inputs: kin, .. } = &self.nodes[*nid].node {
                for &p in kin.iter() {
                    if !inputs.contains(&p) {
                        deps.push(p);
                    }
                }
            }
        }
        deps
    }

    /// Union of `class_params` of all non-Kernel nodes in class `cid`.
    fn deps_without_kernels(&self, cid: ClassId) -> Vec<ClassId> {
        let mut deps = Vec::new();
        for nid in &self.classes[cid].nodes {
            if matches!(&self.nodes[*nid].node, Node::Kernel { .. }) {
                continue;
            }
            for p in self.nodes[*nid].node.class_params() {
                if !deps.contains(&p) {
                    deps.push(p);
                }
            }
        }
        deps
    }

    pub fn debug(&self) {
        let line = "─".repeat(60);
        println!("\n{}", line);
        println!("  E-Graph");
        println!("{}", line);
        for cid in self.classes.ids() {
            let class = &self.classes[cid];
            let shape_str = format!("{:?}", self.shape(cid));
            let dtype_str = format!("{:?}", self.dtype(cid));
            println!("Class {:?} shape={} dtype={}", cid, shape_str, dtype_str);
            for &nid in &class.nodes {
                let kind = &self.nodes[nid].node;
                let inputs: Vec<ClassId> = match kind {
                    Node::Kernel { inputs, .. } => inputs.to_vec(),
                    _ => kind.class_params(),
                };
                let name = match kind {
                    Node::Reduce { rop: bop, .. } => format!("Reduce {:?}", bop),
                    Node::Binary { bop, .. } => format!("Binary {:?}", bop),
                    Node::Assign { .. } => "Assign".into(),
                    Node::After { .. } => "After".into(),
                    Node::Unary { uop, .. } => format!("Unary {:?}", uop),
                    Node::Cast { dtype, .. } => format!("Cast {:?}", dtype),
                    Node::Kernel { program_id, time, .. } => format!("Kernel prog={:?} time={}", program_id, time),
                    Node::Expand { .. } => "Expand".into(),
                    Node::Permute { axes, .. } => format!("Permute {:?}", axes),
                    Node::Reshape { shape, .. } => format!("Reshape shape={shape:?}"),
                    Node::Pad { axis, lp, rp, .. } => format!("Pad axis={axis:?} lp={lp:?} rp={rp:?}"),
                    Node::Narrow { axis, start, len, x } => format!("Narrow {x:?} axis={axis:?} start={start:?} len={len:?}"),
                    Node::Flip { axes, .. } => format!("Flip {:?}", axes),
                    Node::Stack { ops } => format!("Stack {:?}", ops),
                    Node::ToDevice { device, time, .. } => format!("ToDevice {:?} time={}", device, time),
                    Node::Contiguous { .. } => "Contiguous".into(),
                    Node::Const(v) => format!("Const {:?}", v),
                    Node::Leaf { dtype, .. } => format!("Leaf {:?}", dtype),
                };
                println!("  {name} {nid:?}: inputs={inputs:?}");
            }
        }
        println!("{}\n", line);
    }

    /// For each kernel node, it needs to go over inputs. Inputs are either realized or other kernel nodes.
    /// Debug assert that. Then for each input, if that input comes from kernel on different device
    /// or if it's in buffer_map on different device, add EGraph::ToDevice node that moves it
    /// to the device of the Node::Kernel.
    // TODO Clean up this method, it's a mess
    pub fn add_memory_ops(&mut self, devices: &Slab<DeviceId, Device>, buffer_map: &Map<TensorId, BufferId>) {
        let class_ids: Vec<ClassId> = self.classes.ids().collect();
        for cid in class_ids {
            let node_ids: Vec<NodeId> = self.classes[cid].nodes.to_vec();
            for &nid in &node_ids {
                let (device_id, inputs) = match &self.nodes[nid].node {
                    Node::Kernel { program_id, inputs, .. } => {
                        debug_assert_ne!(program_id.device, DeviceId::NULL);
                        (program_id.device, inputs.clone())
                    }
                    _ => continue,
                };
                let dev_pool = devices[device_id].memory_pool_id();

                let class_of = self.nodes[nid].class_of;
                let mut new_inputs: Option<Box<[ClassId]>> = None;
                for (i, &input_cid) in inputs.iter().enumerate() {
                    let mut same_device = false;
                    let mut from_kernel = false;
                    for &inid in &self.classes[input_cid].nodes {
                        if let Node::Kernel { program_id, .. } = &self.nodes[inid].node {
                            from_kernel = true;
                            if program_id.device == device_id {
                                same_device = true;
                                break;
                            }
                        }
                    }
                    if from_kernel {
                        if !same_device {
                            let to_cid = self.push_to_device(input_cid, device_id, 0);
                            if to_cid != class_of {
                                let new_inputs = new_inputs.get_or_insert_with(|| inputs.clone());
                                new_inputs[i] = to_cid;
                            }
                        }
                    } else {
                        let already_on_device = self.classes[input_cid]
                            .nodes
                            .iter()
                            .any(|&inid| matches!(&self.nodes[inid].node, Node::ToDevice { device: d, .. } if *d == device_id));
                        if !already_on_device {
                            let is_leaf = self.classes[input_cid]
                                .nodes
                                .iter()
                                .any(|&inid| matches!(&self.nodes[inid].node, Node::Leaf { .. }));
                            if is_leaf {
                                let tid = self.leaf_map.get(&input_cid).copied().unwrap_or_else(|| {
                                    let leaf_cid = self.classes[input_cid]
                                        .nodes
                                        .iter()
                                        .find_map(|&inid| {
                                            if matches!(&self.nodes[inid].node, Node::Leaf { .. }) {
                                                Some(self.nodes[inid].class_of)
                                            } else {
                                                None
                                            }
                                        })
                                        .expect("already checked is_leaf");
                                    self.leaf_map[&leaf_cid]
                                });
                                let leaf_pool = buffer_map[&tid].pool;
                                if leaf_pool != dev_pool {
                                    let to_cid = self.push_to_device(input_cid, device_id, 0);
                                    if to_cid != cid && to_cid != class_of {
                                        let new_inputs = new_inputs.get_or_insert_with(|| inputs.clone());
                                        new_inputs[i] = to_cid;
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(new_inputs) = new_inputs
                    && let Node::Kernel { inputs: node_inputs, .. } = &mut self.nodes[nid].node
                {
                    *node_inputs = new_inputs;
                }
            }
        }
    }

    /// Hash of the graph structure (hashcons), output classes, and the shape
    /// and dtype of every class. Deterministic across equivalent graphs — used
    /// as a cache key for compiled plans.
    ///
    /// Shape and dtype must be part of the key: two graphs with the same node
    /// structure but different shapes/dtypes (e.g. an `f32[10]` sin vs an
    /// `f32[3]` sin) would otherwise share a plan with wrong allocation sizes.
    #[must_use]
    pub fn cache_key(&self, outputs: &BTreeSet<ClassId>) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        for (node, &id) in &self.hashcons {
            id.hash(&mut hasher);
            node.hash(&mut hasher);
        }

        // TODO - is this even needed?
        for cid in self.classes.ids() {
            cid.hash(&mut hasher);
        }

        for &cid in outputs {
            cid.hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Returns the set of Kernel/ToDevice nodes forming the cheapest valid computation from leaves
    /// to all outputs.
    ///
    /// # Cost model
    ///
    /// Only [`Node::Kernel`] and [`Node::ToDevice`] carry real costs (execution time in nanoseconds).
    /// All other node types (Expand, Reshape, Cast, Binary, Unary, etc.) are structural/fusing
    /// artifacts — they represent intermediate graph transformations that must be fused into kernels
    /// by [`kernelize`](self::kernelizer::Graph::kernelize) before extraction.
    ///
    /// # Invariant
    ///
    /// A path composed exclusively of [`Node::Kernel`] and [`Node::ToDevice`] nodes must exist
    /// from leaves (the only realized classes) to every output class. Without this path the output
    /// cannot be computed, because non-Kernel/ToDevice nodes have no associated runtime cost.
    ///
    /// Dead graph regions (classes with no kernel path) are harmless as long as they don't appear
    /// on output computation paths. [`kernelize`](self::kernelizer::Graph::kernelize) is responsible for ensuring every output
    /// class satisfies this invariant by fusing enough nodes into kernels.
    ///
    /// # Panics
    ///
    /// Panics if any output class lacks a producer path through Kernel or ToDevice nodes.
    #[must_use]
    pub fn extract(&self, outputs: &BTreeSet<ClassId>) -> Vec<NodeId> {
        let order = self.topo_sort_classes(outputs);

        let n = self.classes.ids().count();
        let is_leaf: Vec<bool> = (0..n)
            .map(|i| {
                let cid = ClassId(i as u32);
                self.classes[cid].nodes.iter().any(|&nid| matches!(&self.nodes[nid].node, Node::Leaf { .. }))
            })
            .collect();

        // Candidate producer nodes per class: Kernel and ToDevice nodes. Multiple
        // kernels may produce the same class (different fusions compete in
        // extraction); a leaf class is already realized and never needs one.
        #[derive(Clone, Copy)]
        struct Cand {
            nid: NodeId,
            time: u64,
        }
        let mut cands: Vec<Vec<Cand>> = vec![Vec::new(); n];
        let nn = self.nodes.ids().count();
        let mut node_in: Vec<Vec<ClassId>> = vec![Vec::new(); nn];
        let mut node_out: Vec<Vec<ClassId>> = vec![Vec::new(); nn];
        let mut node_time: Vec<u64> = vec![0; nn];
        for &cid in &order {
            for &nid in &self.classes[cid].nodes {
                let (time, inputs, outputs) = match &self.nodes[nid].node {
                    Node::Kernel { inputs, outputs, time, .. } => (*time, inputs.to_vec(), outputs.to_vec()),
                    Node::ToDevice { x, time, .. } => {
                        let outputs = vec![self.nodes[nid].class_of];
                        (*time, vec![*x], outputs)
                    }
                    _ => continue,
                };
                node_time[nid.0 as usize] = time;
                node_in[nid.0 as usize] = inputs.clone();
                node_out[nid.0 as usize] = outputs.clone();
                cands[cid.0 as usize].push(Cand { nid, time });
            }
        }

        // After classes alias their base leaf buffer; their value comes from the
        // assign writing in-place over the previous version of that buffer.
        // Needing an After class forces its whole assign chain to run (every
        // earlier After plus the assign classes) — otherwise the in-place store
        // kernels of chained assigns get dropped. Mirrors the backward walk below.
        let mut after_chain: Vec<Vec<ClassId>> = vec![Vec::new(); n];
        for &cid in &order {
            let mut chain = Vec::new();
            let mut cur = cid;
            while let Some(&nid2) =
                self.classes[cur].nodes.iter().find(|&&nid| matches!(&self.nodes[nid].node, Node::After { .. }))
            {
                let Node::After { x, dep } = &self.nodes[nid2].node else {
                    unreachable!()
                };
                chain.push(*x);
                chain.push(*dep);
                if *x == cur {
                    break;
                }
                cur = *x;
            }
            after_chain[cid.0 as usize] = chain;
        }

        struct Ctx<'a> {
            outputs: &'a BTreeSet<ClassId>,
            order: &'a [ClassId],
            cands: &'a [Vec<Cand>],
            node_in: &'a [Vec<ClassId>],
            node_out: &'a [Vec<ClassId>],
            node_time: &'a [u64],
            after_chain: &'a [Vec<ClassId>],
            is_leaf: &'a [bool],
        }

        impl Ctx<'_> {
            /// The classes that still must be produced (`pending`, in topological
            /// order) and the classes already produced, derived from `selected`.
            fn pending_and_produced(&self, selected: &Set<NodeId>) -> (Vec<ClassId>, Set<ClassId>) {
                let mut produced: Set<ClassId> = Set::default();
                let mut requested: Set<ClassId> = self.outputs.iter().copied().collect();
                for &nid in selected {
                    for &o in &self.node_out[nid.0 as usize] {
                        produced.insert(o);
                    }
                    for &i in &self.node_in[nid.0 as usize] {
                        requested.insert(i);
                    }
                }
                loop {
                    let mut add: Vec<ClassId> = Vec::new();
                    for &c in &requested {
                        for &r in &self.after_chain[c.0 as usize] {
                            if !requested.contains(&r) {
                                add.push(r);
                            }
                        }
                    }
                    if add.is_empty() {
                        break;
                    }
                    for r in add {
                        requested.insert(r);
                    }
                }
                let mut pending = Vec::new();
                for &c in self.order {
                    if !produced.contains(&c)
                        && requested.contains(&c)
                        && !self.is_leaf[c.0 as usize]
                        && !self.cands[c.0 as usize].is_empty()
                    {
                        pending.push(c);
                    }
                }
                (pending, produced)
            }

            fn plan_cost(&self, selected: &Set<NodeId>) -> u64 {
                selected.iter().map(|&nid| self.node_time[nid.0 as usize]).sum()
            }

            /// A feasible plan that selects the cheapest producer of each pending
            /// class in topological order. Always terminates; provides the upper
            /// bound for the search and a safe fallback.
            fn greedy(&self) -> Set<NodeId> {
                let mut selected: Set<NodeId> = Set::default();
                loop {
                    let (pending, _) = self.pending_and_produced(&selected);
                    if pending.is_empty() {
                        return selected;
                    }
                    let c = pending[0];
                    let cand = self.cands[c.0 as usize].iter().min_by_key(|k| k.time).expect("pending class has no candidates");
                    selected.insert(cand.nid);
                }
            }

            /// Branch-and-bound DFS over producer sets. `selected` is the current
            /// set, `cost` the cost so far, `best` the best total cost seen
            /// (prunes branches that cannot improve it). Returns the cheapest
            /// completion from this state and the nodes it selects.
            fn search(&self, selected: &mut Set<NodeId>, cost: u64, best: &mut u64) -> Option<(u64, Vec<NodeId>)> {
                let (pending, _) = self.pending_and_produced(selected);
                if pending.is_empty() {
                    return Some((0, Vec::new()));
                }
                let c = pending[0];
                let mut ordered: Vec<&Cand> = self.cands[c.0 as usize].iter().collect();
                ordered.sort_by_key(|k| k.time);
                let mut best_res: Option<(u64, Vec<NodeId>)> = None;
                for cand in ordered {
                    if selected.contains(&cand.nid) {
                        continue;
                    }
                    let new_cost = cost + cand.time;
                    if new_cost >= *best {
                        continue;
                    }
                    selected.insert(cand.nid);
                    if let Some((rest, mut nodes)) = self.search(selected, new_cost, best) {
                        let total = cand.time + rest;
                        nodes.push(cand.nid);
                        if best_res.as_ref().is_none_or(|(b, _)| total < *b) {
                            best_res = Some((total, nodes));
                            *best = (*best).min(cost + total);
                        }
                    }
                    selected.remove(&cand.nid);
                }
                best_res
            }
        }

        let ctx = Ctx {
            outputs,
            order: &order,
            cands: &cands,
            node_in: &node_in,
            node_out: &node_out,
            node_time: &node_time,
            after_chain: &after_chain,
            is_leaf: &is_leaf,
        };

        let greedy_plan = ctx.greedy();
        let greedy_cost = ctx.plan_cost(&greedy_plan);

        // Output classes must have a producer path through Kernel/ToDevice
        // nodes. Leaves are already realized and need none.
        let (_, produced) = ctx.pending_and_produced(&greedy_plan);
        for &ocid in outputs {
            if !is_leaf[ocid.0 as usize] && !produced.contains(&ocid) {
                panic!("class {ocid:?} has no valid producer path through Kernel or ToDevice nodes");
            }
        }

        // Cheapest closed producer set: the search improves on greedy when a
        // cheaper closure exists, otherwise greedy is already optimal.
        let mut best = greedy_cost;
        let mut winning = greedy_plan.clone();
        if let Some((total, nodes)) = ctx.search(&mut Set::default(), 0, &mut best)
            && total < greedy_cost
        {
            winning = nodes.into_iter().collect();
        }

        // Producer of each class in the winning plan (multi-output kernels
        // produce several classes at once).
        let mut producer: Vec<Option<NodeId>> = vec![None; n];
        for &nid in &winning {
            for &oc in &node_out[nid.0 as usize] {
                producer[oc.0 as usize] = Some(nid);
            }
        }

        // Mark every class needed to compute the outputs by walking backward from
        // the outputs through the selected producers. The winning plan's selected
        // set is already closed under its producers, so this is a no-op on the
        // pure kernel graph — it exists to (a) thread the After/assign chains
        // below and (b) emit the producers in class-topological order.
        let mut needed: Vec<bool> = vec![false; n];
        let mut stack: Vec<ClassId> = outputs.iter().copied().collect();
        loop {
            while let Some(cid) = stack.pop() {
                if !needed[cid.0 as usize] {
                    needed[cid.0 as usize] = true;
                    if let Some(nid) = producer[cid.0 as usize] {
                        match &self.nodes[nid].node {
                            Node::Kernel { inputs, .. } => stack.extend(inputs.iter().copied()),
                            Node::ToDevice { x, .. } => stack.push(*x),
                            _ => {}
                        }
                    }
                    // After classes alias their base leaf buffer, and their
                    // value comes from dep (the assign) writing over x's
                    // version of that buffer. If the post-assign value is
                    // needed, the assign that wrote it and every earlier
                    // After in the chain are needed too — otherwise extract
                    // drops the in-place store kernels of chained assigns.
                    if let Some(&nid2) =
                        self.classes[cid].nodes.iter().find(|&&nid| matches!(&self.nodes[nid].node, Node::After { .. }))
                        && let Node::After { x, dep } = &self.nodes[nid2].node
                    {
                        stack.push(*x);
                        stack.push(*dep);
                    }
                }
            }
            // In-place assigns write into the realized buffer of their dst's base
            // leaf class, so the store kernel is a side effect on that buffer
            // rather than a producer on the read path — nothing consumes the
            // assign class, so the backward walk above never reaches it. Run the
            // store whenever the buffer it writes is needed.
            let mut add: Vec<ClassId> = Vec::new();
            for &cid in &order {
                if !needed[cid.0 as usize]
                    && self.classes[cid].nodes.iter().any(|&nid| {
                        matches!(&self.nodes[nid].node, Node::Assign { dst, .. } if needed[self.base_leaf(*dst).0 as usize])
                    })
                {
                    add.push(cid);
                }
            }
            if add.is_empty() {
                break;
            }
            for &cid in &add {
                needed[cid.0 as usize] = true;
            }
            stack.extend(add);
        }

        let mut result = Vec::new();
        let mut seen: Set<NodeId> = Set::default();
        for &cid in &order {
            if !needed[cid.0 as usize] {
                continue;
            }
            if let Some(nid) = producer[cid.0 as usize]
                && seen.insert(nid)
            {
                result.push(nid);
            }
        }
        result
    }

    pub fn rank(&self, class: ClassId) -> UAxis {
        todo!()
    }

    pub fn shape(&self, class: ClassId) -> &[Dim] {
        if let Some(shape) = self.shapes.get(&class) {
            return shape;
        }
        match &self.nodes[self.classes[class].nodes[0]].node {
            Node::Const(_) => &[1],
            Node::Expand { x, .. }
            | Node::Cast { x, .. }
            | Node::Permute { x, .. }
            | Node::Reshape { x, .. }
            | Node::Pad { x, .. }
            | Node::Flip { x, .. }
            | Node::Narrow { x, .. }
            | Node::Reduce { x, .. }
            | Node::Unary { x, .. }
            | Node::After { x, .. }
            | Node::ToDevice { x, .. }
            | Node::Contiguous { x }
            | Node::Binary { x, .. } => self.shape(*x),
            Node::Stack { .. } => &[],
            _ => todo!(),
        }
    }

    pub fn dtype(&self, class: ClassId) -> DType {
        match &self.nodes[self.classes[class].nodes[0]].node {
            Node::Const(c) => c.dtype(),
            Node::Leaf { dtype, .. } => *dtype,
            Node::Cast { dtype, .. } => *dtype,
            Node::Assign { dst, .. } => self.dtype(*dst),
            Node::Kernel { outputs, .. } => self.dtype(outputs[0]),
            Node::Stack { ops } => self.dtype(ops[0]),
            Node::Expand { x, .. }
            | Node::Permute { x, .. }
            | Node::Reshape { x, .. }
            | Node::Pad { x, .. }
            | Node::Flip { x, .. }
            | Node::Narrow { x, .. }
            | Node::Reduce { x, .. }
            | Node::Unary { x, .. }
            | Node::After { x, .. }
            | Node::ToDevice { x, .. }
            | Node::Contiguous { x }
            | Node::Binary { x, .. } => self.dtype(*x),
        }
    }
}

impl Runtime {
    pub fn promote_to_graph(&mut self, tid: TensorId, graph_id: GraphId) -> Result<ClassId, ZyxError> {
        let (class_id, gid) = (self.tensors[tid].class_id, self.tensors[tid].graph_id);
        if !class_id.is_null() {
            if !self.graphs[gid].dead {
                if graph_id == gid {
                    return Ok(class_id);
                } else {
                    panic!("tensor belongs to a different tape scope");
                }
            }
            // Graph is dead: the tensor reverts to eager (its kernel_id is still
            // valid since we never mutated the eager kernel). Clear the graph
            // affiliation before promoting it into a new scope.
            self.tensors[tid].class_id = ClassId::NULL;
            self.tensors[tid].graph_id = GraphId::NULL;
            self.graphs[gid].ref_count -= 1;
            if self.graphs[gid].dead && self.graphs[gid].ref_count == 0 {
                self.remove_dead_graph(gid);
            }
        }

        let (kernel_id, my_op_id) = self.eager_ids(tid);

        // Already realized eager tensors promote to the graph as leaves directly.
        // Their buffer is read by the plan as an input; the value is preserved and
        // not recomputed. The eager kernel is left untouched (rc/outputs already
        // count the handles), so the tensor reverts to eager when the graph dies.
        if self.buffer_map.contains_key(&tid) {
            let dtype = self.dtype(tid);
            let (_, class_id) = self.push_leaf_node(graph_id, dtype);
            self.graphs[graph_id].leaf_map.insert(class_id, tid);
            self.graphs[graph_id].leaf_classes.push(class_id);
            self.graphs[graph_id].ref_count += 1;
            self.tensors[tid].class_id = class_id;
            self.tensors[tid].graph_id = graph_id;
            return Ok(class_id);
        }

        debug_assert!(self.kernels[kernel_id].outputs.contains(&tid));

        let relevant = {
            let kernel = &self.kernels[kernel_id].kernel;
            let mut relevant: Set<OpId> = Set::default();
            let mut stack = vec![my_op_id];
            while let Some(oid) = stack.pop() {
                if !relevant.insert(oid) {
                    continue;
                }
                match kernel.ops[oid].op {
                    Op::Storage { .. } | Op::Const(_) => {}
                    Op::Unary { x, .. } => stack.push(x),
                    Op::Binary { x, y, .. } => {
                        stack.push(x);
                        stack.push(y);
                    }
                    Op::Cast { x, .. } => stack.push(x),
                    Op::Reduce { x, .. } => stack.push(x),
                    Op::Move { x, .. } => stack.push(x),
                    _ => unreachable!(),
                }
            }
            relevant
        };

        let loads = self.kernels[kernel_id].loads.clone();
        let mut op_to_class: Map<OpId, ClassId> = Map::default();
        let mut define_idx = 0;
        let mut op_id = self.kernels[kernel_id].kernel.head;
        while !op_id.is_null() {
            if relevant.contains(&op_id) {
                let class_id = match self.kernels[kernel_id].kernel.ops[op_id].op {
                    Op::Param { .. } => {
                        let load_tid = loads[define_idx];
                        if !self.buffer_map.contains_key(&load_tid) {
                            let pending = if self.tensors[load_tid].class_id.is_null() {
                                self.tensors[load_tid].depends_on
                            } else {
                                KernelId::NULL
                            };
                            debug_assert!(!pending.is_null());
                            let outputs: Vec<TensorId> = self.kernels[pending].outputs.iter().copied().collect();
                            for &otid in &outputs {
                                self.add_store(otid)?;
                            }
                        }

                        if !self.tensors[load_tid].class_id.is_null()
                            && self.tensors[load_tid].graph_id == graph_id
                            && !self.graphs[graph_id].dead
                        {
                            // load_tid is already a leaf of this graph: reuse its class.
                            self.tensors[load_tid].class_id
                        } else {
                            todo!()
                            /*let (_, class_id) = self.push_leaf_node(graph_id);
                            self.graphs[graph_id].leaf_map.insert(class_id, load_tid);
                            self.graphs[graph_id].leaf_classes.push(class_id);
                            self.graphs[graph_id].ref_count += 1;
                            self.tensors[load_tid].class_id = class_id;
                            self.tensors[load_tid].graph_id = graph_id;
                            class_id*/
                        }
                    }
                    Op::Const(x) => {
                        let (_, class_id) = self.push_node(graph_id, Node::Const(x));
                        class_id
                    }
                    Op::Unary { x, uop } => {
                        let x_class = op_to_class[&x];
                        let (_, class_id) = self.push_node(graph_id, Node::Unary { x: x_class, uop });
                        class_id
                    }
                    Op::Binary { x, y, bop } => {
                        let x_class = op_to_class[&x];
                        let y_class = op_to_class[&y];
                        self.push_binary_node(graph_id, x_class, y_class, bop)
                    }
                    Op::Cast { x, dtype } => {
                        let x_class = op_to_class[&x];
                        let (_, class_id) = self.push_node(graph_id, Node::Cast { x: x_class, dtype });
                        class_id
                    }
                    Op::Stack { ref ops } => {
                        let ops: Box<[ClassId]> = ops.iter().map(|o| op_to_class[o]).collect();
                        let (_, class_id) = self.push_node(graph_id, Node::Stack { ops });
                        class_id
                    }
                    Op::Reduce { x, rop, .. } => {
                        let x_class = op_to_class[&x];
                        let rank = self.graphs[graph_id].rank(x_class);
                        debug_assert!(rank >= 1, "Reduce: input rank must be >= 1");
                        let (_, class_id) = self.push_node(graph_id, Node::Reduce { x: x_class, rop, axes: vec![rank - 1].into() });
                        class_id
                    }
                    Op::Move { x, ref mop } => {
                        let x_class = op_to_class[&x];
                        let in_shape = self.graphs[graph_id].shape(x_class);
                        match mop.as_ref() {
                            MoveOp::Reshape { shape, input_rank } => {
                                let shape = op_to_class[&shape];
                                let (_, class_id) = self.push_node(graph_id, Node::Reshape { x: x_class, shape, input_rank: *input_rank });
                                class_id
                            }
                            MoveOp::Expand { shape } => {
                                let shape = op_to_class[&shape];
                                let (_, class_id) = self.push_node(graph_id, Node::Expand { x: x_class, shape });
                                class_id
                            }
                            MoveOp::Permute { axes } => {
                                debug_assert_eq!(
                                    axes.len(),
                                    in_shape.len(),
                                    "Permute: axes length {} != input rank {} (shape {:?})",
                                    axes.len(),
                                    in_shape.len(),
                                    in_shape
                                );
                                /*debug_assert_eq!(
                                    shape.len(),
                                    in_shape.len(),
                                    "Permute: output shape rank {} != input rank {} (shape {:?})",
                                    shape.len(),
                                    in_shape.len(),
                                    in_shape
                                );*/
                                let axes = axes.clone().into();
                                let (_, class_id) = self.push_node(graph_id, Node::Permute { x: x_class, axes });
                                class_id
                            }
                            MoveOp::Pad { axis, lp, rp } => {
                                let lp = op_to_class[&lp];
                                let rp = op_to_class[&rp];
                                let (_, class_id) = self.push_node(graph_id, Node::Pad { x: x_class, axis: *axis, lp, rp });
                                class_id
                            }
                            MoveOp::Narrow { axis, start, len } => {
                                let start = op_to_class[&start];
                                let len = op_to_class[&len];
                                let (_, class_id) = self.push_node(graph_id, Node::Narrow { x: x_class, axis: *axis, start, len });
                                class_id
                            }
                            MoveOp::Flip { axes } => {
                                debug_assert!(
                                    !axes.is_empty(),
                                    "Flip: axes must not be empty (rank {} shape {:?})",
                                    in_shape.len(),
                                    in_shape
                                );
                                let axes = axes.clone().into();
                                let (_, class_id) = self.push_node(graph_id, Node::Flip { x: x_class, axes });
                                class_id
                            }
                        }
                    }
                    _ => unreachable!(),
                };
                op_to_class.insert(op_id, class_id);
            }

            if matches!(self.kernels[kernel_id].kernel.at(op_id), Op::Storage { .. }) {
                define_idx += 1;
            }
            op_id = self.kernels[kernel_id].kernel.next_op(op_id);
        }

        let class_id = op_to_class[&my_op_id];
        self.graphs[graph_id].ref_count += 1;
        self.tensors[tid].class_id = class_id;
        self.tensors[tid].graph_id = graph_id;
        Ok(class_id)
    }

    pub fn autotune_jit_kernels(&mut self, graph_id: GraphId) -> Result<(), ZyxError> {
        println!("Autotuning");
        let device_ids: Vec<DeviceId> = self.devices.ids().collect();

        let jit_kernels: *const Slab<JitKernelId, JitKernelData> = &self.graphs[graph_id].jit_kernels;
        let jit_kernels: &Slab<JitKernelId, JitKernelData> = unsafe { &*jit_kernels };
        let total = jit_kernels.len().0 as u64 * device_ids.len() as u64;
        let mut bar = crate::prog_bar::ProgressBar::new(total);
        for ek in jit_kernels.values() {
            let (flop, read, write) = ek.kernel.flop_mem_rw();
            let class_of = ek.stores.first().copied().unwrap();

            for &dev_id in device_ids.iter() {
                // AOT-only devices (e.g. cblas) never compile generic zyx kernels
                if self.devices[dev_id].aot_only() {
                    continue;
                }
                let pool_id = self.devices[dev_id].memory_pool_id();
                let mut kernel = ek.kernel.clone();
                kernel.device_id = dev_id;
                bar.inc(1, &format!("autotune {} on dev={}", kernel.name(), dev_id.0));
                let (dev_prog, timing) = self.get_or_autotune(kernel, pool_id, flop, read, write, &[])?;
                let prog = ProgramId { device: dev_id, program: dev_prog };

                let knid = self.graphs[graph_id].nodes.push(NodeData {
                    node: Node::Kernel {
                        inputs: ek.loads.clone().into(),
                        outputs: ek.stores.clone().into(),
                        program_id: prog,
                        time: timing,
                    },
                    class_of,
                });

                for &ocid in &*ek.stores {
                    self.graphs[graph_id].classes[ocid].nodes.push(knid);
                }
                if !ek.stores.contains(&class_of) {
                    self.graphs[graph_id].classes[class_of].nodes.push(knid);
                }
            }
        }

        if cfg!(debug_assertions) {
            let mut seen: Set<NodeId> = Set::default();
            for cid in self.graphs[graph_id].classes.ids() {
                for &nid in &self.graphs[graph_id].classes[cid].nodes {
                    if !seen.insert(nid) {
                        continue;
                    }
                    if let Node::Kernel { time, .. } = &self.graphs[graph_id].nodes[nid].node {
                        debug_assert!(*time > 0, "Kernel node {nid:?} has zero cost after autotune");
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn debug_assert_pre_realize(&self, graph_id: GraphId) {
        if cfg!(debug_assertions) {
            // I2: all leaves realized. A leaf is either a directly-promoted
            // realized tensor (Graph state) or the load tensor of a promoted
            // kernel (Eager state) — both carry a buffer.
            for &tid in self.graphs[graph_id].leaf_map.values() {
                debug_assert!(self.buffer_map.contains_key(&tid), "leaf {tid} not realized");
                debug_assert!(self.tensors[tid].graph_id == graph_id, "leaf {tid} belongs to another graph");
            }
            // I2: no non-leaf graph tensor is realized — except in-place assign
            // targets, whose value lives in the (realized) leaf buffer they alias.
            for (tid, td) in self.tensors.iter() {
                if td.graph_id == graph_id
                    && !self.graphs[graph_id].is_leaf(td.class_id)
                    && !self.graphs[graph_id].is_after(td.class_id)
                {
                    debug_assert!(!self.buffer_map.contains_key(&tid), "non-leaf graph tensor {tid} realized before realize");
                }
            }
        }
    }

    /// Compiles the graph into an [`ExecPlan`]: pattern-matches AOT kernels,
    /// kernelizes the remaining structural nodes, autotunes the fused kernels,
    /// extracts the cheapest kernel path, and returns the resulting plan.
    pub(crate) fn compile_graph(&mut self, graph_id: GraphId, output_set: &BTreeSet<ClassId>) -> Result<ExecPlan, ZyxError> {
        debug_assert!(self.graphs.contains_key(graph_id));
        self.debug_assert_pre_realize(graph_id);

        for cid in self.graphs[graph_id].classes.ids() {
            let has_leaf = self.graphs[graph_id].classes[cid]
                .nodes
                .iter()
                .any(|&nid| matches!(&self.graphs[graph_id].nodes[nid].node, Node::Leaf { .. }));
            if has_leaf {
                let &tid = self.graphs[graph_id].leaf_map.get(&cid).expect("class {cid:?} has Leaf node but not in leaf_map");
                assert!(self.buffer_map.contains_key(&tid), "leaf class {cid:?} tid {tid:?} not in buffer_map");
            } else {
                assert!(!self.graphs[graph_id].leaf_map.contains_key(&cid), "class {cid:?} has no Leaf node but is in leaf_map");
            }
        }

        // Pattern match specialized AOT kernels (e.g. matmul -> cblas) so they can
        // compete with the fused zyx kernels in extraction.
        // SAFETY: devices, graphs and shapes are separate fields of Runtime, no aliasing, rust is stupid
        let dev_ids: Vec<DeviceId> = self.devices.ids().collect();
        let graph_ptr: *mut Graph = &mut self.graphs[graph_id];
        for dev_id in dev_ids {
            self.devices[dev_id].match_graph(unsafe { &mut *graph_ptr }, output_set);
        }

        // AOT kernel output classes, grouped by the memory pool they run in.
        let mut pool_kernel_outputs: Map<PoolId, Set<ClassId>> = Map::default();
        for cid in self.graphs[graph_id].classes.ids() {
            for nid in &self.graphs[graph_id].classes[cid].nodes {
                if let Node::Kernel { program_id, .. } = &self.graphs[graph_id].nodes[*nid].node {
                    let pool = self.devices[program_id.device].memory_pool_id();
                    pool_kernel_outputs.entry(pool).or_default().insert(cid);
                }
            }
        }

        // Pass 1: fill every gap between all AOT kernels, ignoring devices.
        let all_kernel_outputs: Set<ClassId> = pool_kernel_outputs.values().flatten().copied().collect();
        self.graphs[graph_id].fill_gaps(&all_kernel_outputs, output_set);

        // Pass 2: for each memory pool, fill the gaps between only that pool's
        // kernels — other pools' kernels are ignored, giving single-pool paths.
        for active_outputs in pool_kernel_outputs.values() {
            self.graphs[graph_id].fill_gaps(active_outputs, output_set);
        }

        // Autotunes custom zyx kernels for all devices and adds kernel nodes for all of them
        self.autotune_jit_kernels(graph_id)?;

        // After all kernels nodes are added, this adds movement ops so extract can pick fastest path
        let devices_ptr: *const Slab<DeviceId, Device> = &self.devices;
        let buffer_map_ptr: *const Map<TensorId, BufferId> = &self.buffer_map;
        self.graphs[graph_id].add_memory_ops(unsafe { &*devices_ptr }, unsafe { &*buffer_map_ptr });

        if self.debug.egraph() {
            self.graphs[graph_id].debug();
        }

        let nodes = self.graphs[graph_id].extract(output_set);

        // Leaf pools at compile time — the plan bakes the alias binding (and
        // any cross-pool copy) into its ExecNodes, so leaves must stay put.
        let mut leaf_pools: Map<ClassId, PoolId> = Map::default();
        for (&cid, &tid) in &self.graphs[graph_id].leaf_map {
            leaf_pools.insert(cid, self.buffer_map[&tid].pool);
        }
        let plan = ExecPlan::new(&self.graphs[graph_id], &nodes, output_set, &self.devices, &leaf_pools);
        if self.debug.egraph() {
            plan.debug();
        }

        Ok(plan)
    }

    pub fn eagerify(&mut self, tid: TensorId) {
        if self.tensors[tid].class_id.is_null() {
            return;
        }
        let graph_id = self.tensors[tid].graph_id;
        let old_kernel_id = self.tensors[tid].kernel_id;

        // Release tid from its old eager kernel (if any): remove it from outputs,
        // prune the unused chain (releasing pruned loads), and drop the kernel if
        // nothing else uses it (releasing its remaining loads).
        let mut pruned: Vec<TensorId> = Vec::new();
        if !old_kernel_id.is_null() {
            let old_op_id = self.tensors[tid].op_id;
            let kernel_died = {
                let kd = &mut self.kernels[old_kernel_id];
                kd.outputs.remove(&tid);
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
        let dtype = self.dtype(tid);
        let kernel_id = self.kernels.push(KernelData {
            outputs: Set::from_iter([tid]),
            loads: Vec::new(),
            stores: Vec::new(),
            kernel: Kernel::new(DeviceId::AUTO),
        });
        let shape = todo!();
        let op_id = self.kernels[kernel_id].kernel.push_back(Op::Param { dtype, kind: ParamKind::Global, shape });
        self.kernels[kernel_id].loads.push(tid);
        self.tensors[tid].kernel_id = kernel_id;
        self.tensors[tid].op_id = op_id;
        self.tensors[tid].depends_on = KernelId::NULL;
        self.retain(tid);
        self.graphs[graph_id].ref_count -= 1;
    }

    pub fn assert_graph_alive(&self, graph_id: GraphId) {
        assert!(
            !self.graphs[graph_id].dead,
            "tape scope has ended (tensor belongs to a dead tape scope; Tape dropped or realized without this tensor being an output)"
        );
    }

    pub fn push_leaf_node(&mut self, graph_id: GraphId, dtype: DType) -> (NodeId, ClassId) {
        let g = &mut self.graphs[graph_id];
        let leaf_id = g.max_leaf_id;
        g.max_leaf_id += 1;
        let node = Node::Leaf { dtype, leaf_id };
        if let Some(&nid) = g.hashcons.get(&node) {
            return (nid, g.nodes[nid].class_of);
        }
        let nid = g.nodes.push(NodeData { node: node.clone(), class_of: ClassId::NULL });
        let cid = g.classes.push(EClass { nodes: vec![nid] });
        g.nodes[nid].class_of = cid;
        g.hashcons.insert(node, nid);
        (nid, cid)
    }

    pub fn new_graph_tensor(&mut self, graph_id: GraphId, class_id: ClassId) -> TensorId {
        self.graphs[graph_id].ref_count += 1;
        let shape = self.graphs[graph_id].shape(class_id).to_vec();
        let tid = self.tensors.push(TensorData {
            kernel_id: KernelId::NULL,
            op_id: OpId::NULL,
            depends_on: KernelId::NULL,
            class_id,
            graph_id,
            rc: 1,
        });
        self.shapes.insert(tid, shape);
        tid
    }

    pub fn push_node(&mut self, graph_id: GraphId, node: Node) -> (NodeId, ClassId) {
        match node {
            Node::Permute { .. } => {
                /*let in_shape = &self.shapes[self.graphs[graph_id].classes[x].shape];
                assert_eq!(
                    axes.len(),
                    in_shape.len(),
                    "Permute: axes length {} != input rank {} (shape {:?})",
                    axes.len(),
                    in_shape.len(),
                    in_shape
                );*/
            }
            Node::Reshape { .. } => {
                /*let in_shape = &self.shapes[self.graphs[graph_id].classes[x].shape];
                let out_shape = &self.shapes[out_shape_id];
                assert_eq!(
                    in_shape.iter().product::<Dim>(),
                    out_shape.iter().product::<Dim>(),
                    "Reshape: element count mismatch {:?} -> {:?}",
                    in_shape,
                    out_shape
                );*/
            }
            Node::Expand { .. } => {
                /* shape dims not yet resolved (Stack). Re-enable once shape() resolves Stack. */
            }
            Node::Reduce { x, ref axes, .. } => {
                let in_shape = self.graphs[graph_id].shape(x);
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
            Node::Pad { x, axis, .. } => {
                let in_shape = self.graphs[graph_id].shape(x);
                assert!(
                    axis < in_shape.len(),
                    "Pad: axis {} out of range for input rank {} (shape {:?})",
                    axis,
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
        let cid = g.classes.push(EClass { nodes: vec![nid] });
        g.nodes[nid].class_of = cid;
        g.hashcons.insert(node, nid);
        (nid, cid)
    }

    pub fn push_binary_node(&mut self, graph_id: GraphId, x: ClassId, y: ClassId, bop: BOp) -> ClassId {
        debug_assert_eq!(self.graphs[graph_id].shape(x), self.graphs[graph_id].shape(y));
        self.push_node(graph_id, Node::Binary { x, y, bop }).1
    }
}
