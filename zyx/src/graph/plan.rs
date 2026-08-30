use std::collections::BTreeSet;

use crate::{
    Map, Set, ZyxError,
    backend::{BufferId, Device, DeviceId, Event, LaunchArg, MemoryPool, PoolId, ProgramId},
    dtype::Constant,
    graph::{ClassId, Graph, Node, NodeId},
    kernel::BOp,
    runtime::Runtime,
    shape::Dim,
    slab::Slab,
};

#[derive(Debug, Clone)]
pub enum ExecNode {
    Allocate {
        class: ClassId,
        pool: PoolId,
        /// Product of the class's static dims, in elements (dynamic dims excluded).
        static_size: Dim,
        dtype_size: Dim,
        /// Dynamic dims of the class's shape, each resolved at execution time
        /// from the scalar value stored in that leaf class's buffer.
        dynamic_dims: Vec<ClassId>,
    },
    Copy {
        dst_class: ClassId,
        src_class: ClassId,
    },
    Deallocate {
        class: ClassId,
    },
    Launch {
        program_id: ProgramId,
        load_classes: Box<[ClassId]>,
        store_classes: Box<[ClassId]>,
    },
    // Binds class_buf[class] = class_buf[to]: an After output aliases the
    // buffer of its base leaf class (in-place assign write). Preplanned by
    // ExecPlan::new so execute_plan only resolves buffers, never decides.
    Alias {
        class: ClassId,
        to: ClassId,
    },
}

#[derive(Debug, Clone)]
pub struct ExecPlan {
    pub nodes: Vec<ExecNode>,
    pub leaf_classes: Vec<ClassId>,
    // Pool each leaf class lived in when the plan was compiled. Leaf pools
    // must not vary across plan reuse, or the preplanned Alias/Allocate/Copy
    // binding would be wrong — debug-asserted in execute_plan.
    pub leaf_pools: Map<ClassId, PoolId>,
}

impl ExecPlan {
    #[must_use]
    pub fn new(
        graph: &Graph,
        nodes: &[NodeId],
        output_set: &BTreeSet<ClassId>,
        devices: &Slab<DeviceId, Device>,
        leaf_pools: &Map<ClassId, PoolId>,
    ) -> Self {
        let mut rc: Map<ClassId, u32> = Map::default();
        for &nid in nodes {
            match &graph.nodes[nid].node {
                Node::Kernel { inputs, .. } => {
                    for &ic in &**inputs {
                        rc.entry(ic).and_modify(|c| *c += 1).or_insert(1);
                    }
                }
                Node::ToDevice { x, .. } => {
                    rc.entry(*x).and_modify(|c| *c += 1).or_insert(1);
                }
                _ => unreachable!(),
            }
        }

        let mut plan_nodes = Vec::new();
        let mut allocated: Set<ClassId> = Set::default();

        // Allocation spec of a class: static element count (product of static
        // dims), dtype byte size and the leaf classes holding dynamic dim
        // values. Dynamic dims cannot be resolved here — their values live in
        // leaf buffers set between plan runs — so execution multiplies them in.
        // Computed dim classes fold recursively; only Const and leaf dims are
        // expected to terminate the walk.
        fn alloc_spec(graph: &Graph, class: ClassId) -> (Dim, Dim, Vec<ClassId>) {
            fn dim_value(graph: &Graph, dim: ClassId, dynamic_dims: &mut Vec<ClassId>) -> Option<Dim> {
                match &graph.nodes[graph.classes[dim].nodes[0]].node {
                    Node::Const { value: c, .. } => {
                        Some(c.as_dim().unwrap_or_else(|| panic!("dim class {dim:?} is not a constant")))
                    }
                    Node::Leaf { .. } => {
                        dynamic_dims.push(dim);
                        None
                    }
                    Node::Binary { x, y, bop: BOp::Add } => {
                        match (dim_value(graph, *x, dynamic_dims), dim_value(graph, *y, dynamic_dims)) {
                            (Some(a), Some(b)) => Some(a + b),
                            _ => None,
                        }
                    }
                    op => todo!("alloc_spec: computed dim class {dim:?} via {op:?}"),
                }
            }
            let dtype_size = Dim::from(graph.dtype(class).bit_size() / 8);
            let mut static_size: Dim = 1;
            let mut dynamic_dims = Vec::new();
            for d in graph.shape(class) {
                if let Some(v) = dim_value(graph, d, &mut dynamic_dims) {
                    static_size *= v;
                }
            }
            (static_size, dtype_size, dynamic_dims)
        }

        // After output classes alias the buffer of x's base leaf class: the
        // assign writes the new buffer version in-place into that leaf buffer,
        // so an After class (x's value after the assign) shares the leaf's
        // buffer. They must not be allocated or deallocated — the leaf's buffer
        // is owned by the realized tensor.
        let mut aliases: Vec<(ClassId, ClassId, Dim, Dim, Vec<ClassId>)> = Vec::new();
        let mut alias_classes: Set<ClassId> = Set::default();
        for cid in graph.classes.ids() {
            for nid in &graph.classes[cid].nodes {
                if let Node::After { x, .. } = &graph.nodes[*nid].node {
                    let base = graph.base_leaf(*x);
                    let (static_size, dtype_size, dynamic_dims) = alloc_spec(graph, cid);
                    aliases.push((cid, base, static_size, dtype_size, dynamic_dims));
                    alias_classes.insert(cid);
                }
            }
        }

        // Pool of the kernel that stores each alias class — precomputed so the
        // binding below is decided at plan time, not execution time.
        let mut store_pool: Map<ClassId, PoolId> = Map::default();
        for &nid in nodes {
            if let Node::Kernel { outputs, program_id, .. } = &graph.nodes[nid].node {
                let pool = devices[program_id.device].memory_pool_id();
                for &oc in &**outputs {
                    store_pool.insert(oc, pool);
                }
            }
        }

        // Bind aliases before any kernel runs. A leaf in the same pool as its
        // assign kernel binds straight to the leaf buffer. A cross-pool leaf
        // needs one kernel-pool copy of itself shared by every alias of that
        // leaf — chained assigns must write the same physical buffer or the
        // intermediate writes are lost. Mirrors eager assign's store-to-target
        // pool handling.
        let mut leaf_copy: Map<ClassId, ClassId> = Map::default();
        for &(class, to, static_size, dtype_size, ref dynamic_dims) in &aliases {
            match store_pool.get(&class) {
                Some(pool) if leaf_pools[&to] != *pool => {
                    let owner = *leaf_copy.entry(to).or_insert_with(|| {
                        plan_nodes.push(ExecNode::Allocate {
                            class,
                            pool: *pool,
                            static_size,
                            dtype_size,
                            dynamic_dims: dynamic_dims.clone(),
                        });
                        plan_nodes.push(ExecNode::Copy { dst_class: class, src_class: to });
                        class
                    });
                    if owner != class {
                        plan_nodes.push(ExecNode::Alias { class, to: owner });
                    }
                }
                _ => plan_nodes.push(ExecNode::Alias { class, to }),
            }
        }

        for &nid in nodes {
            match &graph.nodes[nid].node {
                Node::Kernel { inputs, outputs, program_id, .. } => {
                    let pool = devices[program_id.device].memory_pool_id();
                    for &oc in &**outputs {
                        if !allocated.insert(oc) {
                            continue;
                        }
                        // Realized leaves and after aliases already have buffers
                        // (leaf buffers via leaf_map, aliases share x's leaf
                        // buffer) — never allocate fresh buffers for them.
                        if !graph.leaf_map.contains_key(&oc) && !alias_classes.contains(&oc) {
                            let (static_size, dtype_size, dynamic_dims) = alloc_spec(graph, oc);
                            plan_nodes.push(ExecNode::Allocate { class: oc, pool, static_size, dtype_size, dynamic_dims });
                        }
                    }
                    plan_nodes.push(ExecNode::Launch {
                        program_id: *program_id,
                        load_classes: inputs.clone(),
                        store_classes: outputs.clone(),
                    });
                    for &ic in &**inputs {
                        let c = rc.get_mut(&ic).unwrap();
                        *c -= 1;
                        if *c == 0
                            && !graph.leaf_map.contains_key(&ic)
                            && !output_set.contains(&ic)
                            && !alias_classes.contains(&ic)
                        {
                            plan_nodes.push(ExecNode::Deallocate { class: ic });
                        }
                    }
                }
                &Node::ToDevice { x, device, .. } => {
                    let pool = devices[device].memory_pool_id();
                    let class_of = graph.nodes[nid].class_of;
                    if allocated.insert(class_of) && !graph.leaf_map.contains_key(&class_of) && !alias_classes.contains(&class_of)
                    {
                        let (static_size, dtype_size, dynamic_dims) = alloc_spec(graph, class_of);
                        plan_nodes.push(ExecNode::Allocate { class: class_of, pool, static_size, dtype_size, dynamic_dims });
                    }
                    plan_nodes.push(ExecNode::Copy { dst_class: class_of, src_class: x });
                    let c = rc.get_mut(&x).unwrap();
                    *c -= 1;
                    if *c == 0 && !graph.leaf_map.contains_key(&x) && !output_set.contains(&x) && !alias_classes.contains(&x) {
                        plan_nodes.push(ExecNode::Deallocate { class: x });
                    }
                }
                _ => unreachable!(),
            }
        }

        // Deallocate kernel outputs that are neither consumed by any node nor
        // requested outputs (e.g. the extra stores of a multi-output kernel).
        let allocated: Vec<ClassId> = allocated.iter().copied().collect();
        for c in allocated {
            if !graph.leaf_map.contains_key(&c) && !output_set.contains(&c) && !alias_classes.contains(&c) && !rc.contains_key(&c)
            {
                plan_nodes.push(ExecNode::Deallocate { class: c });
            }
        }

        Self { nodes: plan_nodes, leaf_classes: graph.leaf_classes.clone(), leaf_pools: leaf_pools.clone() }
    }

    #[allow(unused)]
    pub fn debug(&self) {
        let line = "─".repeat(60);
        println!("\n{}", line);
        println!("  ExecPlan");
        println!("{}", line);
        for node in &self.nodes {
            match node {
                ExecNode::Allocate { class, pool, static_size, dtype_size, dynamic_dims } => {
                    println!(
                        "  Allocate class={class:?} pool={pool:?} static={static_size} dtype_size={dtype_size} dyn={dynamic_dims:?}"
                    );
                }
                ExecNode::Copy { dst_class, src_class } => {
                    println!("  Copy dst={dst_class:?} src={src_class:?}");
                }
                ExecNode::Deallocate { class } => {
                    println!("  Deallocate class={class:?}");
                }
                ExecNode::Launch { program_id, load_classes, store_classes } => {
                    println!("  Launch prog={program_id:?} loads={load_classes:?} stores={store_classes:?}");
                }
                ExecNode::Alias { class, to } => {
                    println!("  Alias class={class:?} -> to={to:?}");
                }
            }
        }
        println!("{}\n", line);
    }
}

impl Runtime {
    pub fn execute_plan(
        &mut self,
        cache_key: u64,
        class_buf: &mut Map<ClassId, BufferId>,
        class_vars: &Map<ClassId, Constant>,
    ) -> Result<(), ZyxError> {
        let plan = self.plan_cache.get(&cache_key).unwrap();

        #[cfg(debug_assertions)]
        {
            for (&cid, &pool) in &plan.leaf_pools {
                debug_assert_eq!(
                    class_buf[&cid].pool, pool,
                    "leaf class {cid:?} moved pools since the plan was compiled — preplanned \
                     Alias/Allocate/Copy binding would be wrong"
                );
            }
        }

        for node in &plan.nodes {
            match node {
                ExecNode::Allocate { class, pool, static_size, dtype_size, dynamic_dims } => {
                    // Resolve dynamic dims from their leaf classes' scalar
                    // values (variables bound between plan runs), then size
                    // the buffer: one element per static*dynamic element,
                    // plus one extra trash element.
                    let mut elements = *static_size;
                    for dim in dynamic_dims {
                        let value = class_vars.get(dim).and_then(|c| c.as_dim());
                        debug_assert!(
                            value.is_some(),
                            "dynamic dim class {dim:?} must resolve to a variable at execution time"
                        );
                        let v = value.unwrap_or(0);
                        debug_assert!(v > 0, "dynamic dim class {dim:?} resolved to non-positive value {v}");
                        elements *= v;
                    }
                    debug_assert!(elements > 0, "allocation for class {class:?} would be empty ({elements} elements)");
                    let bytes = (elements + 1) * dtype_size;
                    let (buf, event) = self.pools[*pool].allocate(bytes)?;
                    let buf_id = BufferId { pool: *pool, buffer: buf };
                    class_buf.insert(*class, buf_id);
                    self.events.insert(BTreeSet::from([buf_id]), event);
                }
                ExecNode::Launch { program_id, load_classes, store_classes } => {
                    let pool_id = self.devices[program_id.device].memory_pool_id();
                    let mut args = Vec::new();
                    let mut kernel_bufs = BTreeSet::new();
                    for c in load_classes.iter().chain(store_classes.iter()) {
                        if let Some(&value) = class_vars.get(c) {
                            // Variable leaf: bound from variable_map, no buffer.
                            args.push(LaunchArg::Variable(value));
                            continue;
                        }
                        let Some(buf) = class_buf.get(c) else {
                            panic!(
                                "DEBUG launch: class {c:?} (program {program_id:?}) has no allocated buffer; load_classes={load_classes:?}, store_classes={store_classes:?}"
                            );
                        };
                        args.push(LaunchArg::Buffer(buf.buffer));
                        kernel_bufs.insert(*buf);
                    }
                    let wait_list = drain_events_for_bufs(&mut self.events, &kernel_bufs);
                    if self.debug.dev() {
                        println!("launching kernel {program_id:?}");
                    }
                    let event =
                        self.devices[program_id.device].launch(program_id.program, &mut self.pools[pool_id], &args, wait_list)?;
                    self.events.insert(kernel_bufs, event);
                }
                ExecNode::Copy { dst_class, src_class } => {
                    let src = class_buf[src_class];
                    let dst = class_buf[dst_class];
                    let wait_list = drain_events_for_buf(&mut self.events, src);
                    // SAFETY: src_pool and dst_pool are different PoolIds (checked
                    // below); rust cannot split the Slab borrow across the two
                    // indices.
                    debug_assert_ne!(src.pool, dst.pool);
                    let src_pool: *mut MemoryPool = &mut self.pools[src.pool];
                    let dst_pool = &mut self.pools[dst.pool];
                    let event = dst_pool.pool_to_pool(unsafe { &mut *src_pool }, src.buffer, dst.buffer, wait_list)?;
                    self.pools[dst.pool].sync_events(vec![event])?;
                }
                ExecNode::Deallocate { class } => {
                    let buf = class_buf.remove(class).unwrap();
                    let wait_list = drain_events_for_buf(&mut self.events, buf);
                    self.pools[buf.pool].deallocate(buf.buffer, wait_list);
                }
                ExecNode::Alias { class, to } => {
                    let buf = class_buf[to];
                    class_buf.insert(*class, buf);
                }
            }
        }

        Ok(())
    }
}

pub(crate) fn drain_events_for_buf(events: &mut Map<BTreeSet<BufferId>, Event>, buf: BufferId) -> Vec<Event> {
    let keys: Vec<BTreeSet<BufferId>> = events.keys().filter(|k| k.contains(&buf)).cloned().collect();
    let mut result = Vec::new();
    for key in keys {
        result.push(events.remove(&key).unwrap());
    }
    result
}

fn drain_events_for_bufs(events: &mut Map<BTreeSet<BufferId>, Event>, bufs: &BTreeSet<BufferId>) -> Vec<Event> {
    let keys: Vec<BTreeSet<BufferId>> = events.keys().filter(|k| !k.is_disjoint(bufs)).cloned().collect();
    let mut result = Vec::new();
    for key in keys {
        result.push(events.remove(&key).unwrap());
    }
    result
}
