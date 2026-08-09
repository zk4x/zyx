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
    kernel::{BOp, DeviceId, Kernel, UOp},
    runtime::{Runtime, ShapeId},
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
        shape: ShapeId,
    },
    Permute {
        x: ClassId,
        axes: Box<[UAxis]>,
    },
    Reshape {
        x: ClassId,
        shape: ShapeId,
    },
    PadZeros {
        x: ClassId,
        padding: Box<[(i64, i64)]>,
    },
    Flip {
        x: ClassId,
        axes: Box<[UAxis]>,
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
    ToDevice {
        x: ClassId,
        device: DeviceId,
        time: u64,
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
            (Self::Reshape { x: a, shape: as_ }, Self::Reshape { x: b, shape: bs }) => a == b && as_ == bs,
            (Self::PadZeros { x: a, padding: ap }, Self::PadZeros { x: b, padding: bp }) => a == b && ap == bp,
            (Self::Flip { x: a, axes: aa }, Self::Flip { x: b, axes: ba }) => a == b && aa == ba,
            (Self::Reduce { x: a, rop: ar, axes: aa }, Self::Reduce { x: b, rop: br, axes: ba }) => {
                a == b && ar == br && aa == ba
            }
            (Self::Cast { x: a, dtype: ad }, Self::Cast { x: b, dtype: bd }) => a == b && ad == bd,
            (Self::Unary { x: a, uop: au }, Self::Unary { x: b, uop: bu }) => a == b && au == bu,
            (Self::Binary { x: a, y: ay, bop: ab }, Self::Binary { x: b, y: by, bop: bb }) => a == b && ay == by && ab == bb,
            (Self::Assign { dst: a, src: as_ }, Self::Assign { dst: b, src: bs }) => a == b && as_ == bs,
            (Self::ToDevice { x: a, device: ad, .. }, Self::ToDevice { x: b, device: bd, .. }) => a == b && ad == bd,
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
            Self::Reshape { x, shape } => {
                4u8.hash(state);
                x.hash(state);
                shape.hash(state);
            }
            Self::PadZeros { x, padding } => {
                5u8.hash(state);
                x.hash(state);
                padding.hash(state);
            }
            Self::Flip { x, axes } => {
                12u8.hash(state);
                x.hash(state);
                axes.hash(state);
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
            Self::Assign { dst, src } => {
                13u8.hash(state);
                dst.hash(state);
                src.hash(state);
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
    pub shape: ShapeId,
    pub dtype: DType,
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
    pub(crate) hashcons: Map<Node, NodeId>,
    pub(crate) nodes: Slab<NodeId, NodeData>,
    pub(crate) classes: Slab<ClassId, EClass>,
    pub(crate) jit_kernels: Slab<JitKernelId, JitKernelData>,
    pub(crate) leaf_map: Map<ClassId, TensorId>,
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
}

impl Node {
    fn class_params(&self) -> Vec<ClassId> {
        match self {
            Self::Const(_) | Self::Leaf { .. } => vec![],
            Self::Expand { x, .. } => vec![*x],
            Self::Permute { x, .. } => vec![*x],
            Self::Reshape { x, .. } => vec![*x],
            Self::PadZeros { x, .. } => vec![*x],
            Self::Flip { x, .. } => vec![*x],
            Self::Reduce { x, .. } => vec![*x],
            Self::Cast { x, .. } => vec![*x],
            Self::Unary { x, .. } => vec![*x],
            Self::Binary { x, y, .. } => vec![*x, *y],
            Self::Assign { dst, src } => vec![*dst, *src],
            Self::ToDevice { x, .. } => vec![*x],
            Self::Kernel { inputs, .. } => inputs.to_vec(),
        }
    }
}

impl Graph {
    pub fn new() -> Self {
        Self {
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

    /// Whether `class_id` is the output of an in-place `assign` — a class whose
    /// value lives in (aliases) dst's realized leaf buffer.
    pub fn is_assign_alias(&self, class_id: ClassId) -> bool {
        self.classes[class_id].nodes.iter().any(|&nid| matches!(&self.nodes[nid].node, Node::Assign { .. }))
    }

    pub fn push_to_device(&mut self, x: ClassId, device: DeviceId, time: u64) -> ClassId {
        let node = Node::ToDevice { x, device, time };
        if let Some(&nid) = self.hashcons.get(&node) {
            return self.nodes[nid].class_of;
        }
        let shape = self.classes[x].shape;
        let dtype = self.classes[x].dtype;
        let nid = self.nodes.push(NodeData { node: node.clone(), class_of: ClassId::NULL });
        let cid = self.classes.push(EClass { nodes: vec![nid], shape, dtype });
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

    /// Dependencies of a class, stopping at `inputs`: an input class that is a
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

    pub fn debug_print(&self, shapes: &Slab<ShapeId, Vec<Dim>>) {
        let line = "─".repeat(60);
        println!("\n{}", line);
        println!("  E-Graph");
        println!("{}", line);
        for cid in self.classes.ids() {
            let class = &self.classes[cid];
            let shape_str = format!("{:?}", shapes[class.shape]);
            let dtype_str = format!("{:?}", class.dtype);
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
                    Node::Unary { uop, .. } => format!("Unary {:?}", uop),
                    Node::Cast { dtype, .. } => format!("Cast {:?}", dtype),
                    Node::Kernel { program_id, time, .. } => format!("Kernel prog={:?} time={}", program_id, time),
                    Node::Expand { .. } => "Expand".into(),
                    Node::Permute { axes, .. } => format!("Permute {:?}", axes),
                    Node::Reshape { shape, .. } => format!("Reshape {:?}", shapes[*shape]),
                    Node::PadZeros { padding, .. } => format!("Pad {:?}", padding),
                    Node::Flip { axes, .. } => format!("Flip {:?}", axes),
                    Node::ToDevice { device, time, .. } => format!("ToDevice {:?} time={}", device, time),
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

        for cid in self.classes.ids() {
            cid.hash(&mut hasher);
            self.classes[cid].shape.hash(&mut hasher);
            self.classes[cid].dtype.hash(&mut hasher);
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
        let mut cost: Vec<Option<u64>> = vec![None; n];
        let mut producer: Vec<Option<NodeId>> = vec![None; n];

        for &cid in &order {
            let idx = cid.0 as usize;

            if self.classes[cid].nodes.iter().any(|&nid| matches!(&self.nodes[nid].node, Node::Leaf { .. })) {
                cost[idx] = Some(0);
            }

            for &nid in &self.classes[cid].nodes {
                match &self.nodes[nid].node {
                    Node::Kernel { inputs, outputs, time, .. } => {
                        if inputs.iter().all(|icid| cost[icid.0 as usize].is_some()) {
                            let total: u64 = inputs.iter().map(|icid| cost[icid.0 as usize].unwrap()).sum();
                            let candidate = time + total;
                            for &ocid in outputs {
                                let oidx = ocid.0 as usize;
                                if cost[oidx].is_none_or(|c| candidate < c) {
                                    cost[oidx] = Some(candidate);
                                    producer[oidx] = Some(nid);
                                }
                            }
                        }
                    }
                    Node::ToDevice { x, time, .. } => {
                        if let Some(c) = cost[x.0 as usize] {
                            let candidate = time + c;
                            if cost[idx].is_none_or(|c| candidate < c) {
                                cost[idx] = Some(candidate);
                                producer[idx] = Some(nid);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        for &ocid in outputs {
            let idx = ocid.0 as usize;
            if cost[idx].is_none() {
                for &cid in &order {
                    if cost[cid.0 as usize].is_none() {
                        /*eprint!("{cid:?}:[");
                        for &nid in &self.classes[cid].nodes {
                            match &self.nodes[nid].node {
                                Node::Kernel { inputs, .. } => eprint!("Kernel(inputs={inputs:?}) "),
                                Node::Leaf { .. } => eprint!("Leaf "),
                                Node::ToDevice { .. } => eprint!("ToDevice "),
                                Node::Const(_) => eprint!("Const "),
                                Node::Expand { .. } => eprint!("Expand "),
                                Node::Permute { .. } => eprint!("Permute "),
                                Node::Reshape { .. } => eprint!("Reshape "),
                                Node::PadZeros { .. } => eprint!("PadZeros "),
                                Node::Reduce { .. } => eprint!("Reduce "),
                                Node::Cast { .. } => eprint!("Cast "),
                                Node::Unary { .. } => eprint!("Unary "),
                                Node::Binary { .. } => eprint!("Binary "),
                            }
                        }
                        eprintln!("]");*/
                        if let Some(producer_nid) = producer[cid.0 as usize]
                            && let Node::Kernel { inputs, .. } = &self.nodes[producer_nid].node
                        {
                            for icid in inputs.iter() {
                                eprintln!("  input {icid:?}: cost={:?}", cost[icid.0 as usize]);
                            }
                        }
                    }
                }
                panic!("class {ocid:?} has no valid producer path through Kernel or ToDevice nodes");
            }
        }

        // Only keep producers that are actually needed to compute the outputs.
        // A class may have a (cheap) producer selected even though nothing on the
        // path to an output consumes it — e.g. an AOT cblas matmul whose result
        // feeds into an op that got fused into a bigger kernel. Such dead kernels
        // must not be extracted. Walk backward from the outputs through the
        // selected producers and mark every reachable class as needed.
        let mut needed: Vec<bool> = vec![false; n];
        let mut stack: Vec<ClassId> = outputs.iter().copied().collect();
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
            }
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
}

impl Runtime {
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

        // SAFETY: graph and shapes are separate fields of Runtime, no aliasing, rust is stupid
        let shapes_ptr: *const Slab<ShapeId, Vec<Dim>> = &self.shapes;

        // Pattern match specialized AOT kernels (e.g. matmul -> cblas) so they can
        // compete with the fused zyx kernels in extraction.
        // SAFETY: devices, graphs and shapes are separate fields of Runtime, no aliasing, rust is stupid
        let dev_ids: Vec<DeviceId> = self.devices.ids().collect();
        let graph_ptr: *mut Graph = &mut self.graphs[graph_id];
        for dev_id in dev_ids {
            self.devices[dev_id].match_graph(unsafe { &mut *graph_ptr }, output_set, unsafe { &*shapes_ptr });
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
        self.graphs[graph_id].fill_gaps(&all_kernel_outputs, output_set, unsafe { &*shapes_ptr });

        // Pass 2: for each memory pool, fill the gaps between only that pool's
        // kernels — other pools' kernels are ignored, giving single-pool paths.
        for active_outputs in pool_kernel_outputs.values() {
            self.graphs[graph_id].fill_gaps(active_outputs, output_set, unsafe { &*shapes_ptr });
        }

        // Autotunes custom zyx kernels for all devices and adds kernel nodes for all of them
        self.autotune_jit_kernels(graph_id)?;

        // After all kernels nodes are added, this adds movement ops so extract can pick fastest path
        let devices_ptr: *const Slab<DeviceId, Device> = &self.devices;
        let buffer_map_ptr: *const Map<TensorId, BufferId> = &self.buffer_map;
        self.graphs[graph_id].add_memory_ops(unsafe { &*devices_ptr }, unsafe { &*buffer_map_ptr });

        if self.debug.egraph() {
            self.graphs[graph_id].debug_print(&self.shapes);
        }

        let nodes = self.graphs[graph_id].extract(output_set);

        let plan = ExecPlan::new(&self.graphs[graph_id], &nodes, output_set, &self.devices, &self.shapes);
        if self.debug.egraph() {
            plan.debug();
        }

        Ok(plan)
    }
}
