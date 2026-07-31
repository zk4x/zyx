use std::collections::BTreeSet;

use crate::{
    Map, Set, ZyxError,
    backend::{BufferId, Device, DeviceId, Event, PoolId, ProgramId},
    graph::{ClassId, Graph, Node, NodeId},
    runtime::{Runtime, ShapeId},
    shape::Dim,
    slab::Slab,
};

#[derive(Debug, Clone)]
pub enum ExecNode {
    Allocate {
        class: ClassId,
        pool: PoolId,
        bytes: Dim,
    },
    Copy {
        dst_class: ClassId,
        src_class: ClassId,
        bytes: Dim,
    },
    Deallocate {
        class: ClassId,
    },
    Launch {
        program_id: ProgramId,
        load_classes: Box<[ClassId]>,
        store_classes: Box<[ClassId]>,
    },
}

#[derive(Debug, Clone)]
pub struct ExecPlan {
    pub nodes: Vec<ExecNode>,
    pub leaf_classes: Vec<ClassId>,
}

impl ExecPlan {
    #[must_use]
    pub fn new(
        graph: &Graph,
        nodes: &[NodeId],
        output_set: &BTreeSet<ClassId>,
        devices: &Slab<DeviceId, Device>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
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

        let class_bytes = |cid: ClassId| -> Dim {
            let class = &graph.classes[cid];
            let shape = &shapes[class.shape];
            let numel: Dim = shape.iter().product();
            // Add one trash element
            ((numel + 1) * class.dtype.bit_size() as Dim + 7) / 8
        };

        for &nid in nodes {
            match &graph.nodes[nid].node {
                Node::Kernel { inputs, outputs, program_id, .. } => {
                    let pool = devices[program_id.device].memory_pool_id();
                    for &oc in &**outputs {
                        if allocated.insert(oc) {
                            plan_nodes.push(ExecNode::Allocate { class: oc, pool, bytes: class_bytes(oc) });
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
                        if *c == 0 && !graph.leaf_map.contains_key(&ic) && !output_set.contains(&ic) {
                            plan_nodes.push(ExecNode::Deallocate { class: ic });
                        }
                    }
                }
                Node::ToDevice { x, device, .. } => {
                    let pool = devices[*device].memory_pool_id();
                    let class_of = graph.nodes[nid].class_of;
                    if allocated.insert(class_of) {
                        plan_nodes.push(ExecNode::Allocate { class: class_of, pool, bytes: class_bytes(class_of) });
                    }
                    let cb = class_bytes(class_of);
                    plan_nodes.push(ExecNode::Copy { dst_class: class_of, src_class: *x, bytes: cb });
                    let c = rc.get_mut(x).unwrap();
                    *c -= 1;
                    if *c == 0 && !graph.leaf_map.contains_key(x) && !output_set.contains(x) {
                        plan_nodes.push(ExecNode::Deallocate { class: *x });
                    }
                }
                _ => unreachable!(),
            }
        }

        // Deallocate kernel outputs that are neither consumed by any node nor
        // requested outputs (e.g. the extra stores of a multi-output kernel).
        let allocated: Vec<ClassId> = allocated.iter().copied().collect();
        for c in allocated {
            if !graph.leaf_map.contains_key(&c) && !output_set.contains(&c) && !rc.contains_key(&c) {
                plan_nodes.push(ExecNode::Deallocate { class: c });
            }
        }

        Self { nodes: plan_nodes, leaf_classes: graph.leaf_classes.clone() }
    }

    #[allow(unused)]
    pub fn debug(&self) {
        let line = "─".repeat(60);
        println!("\n{}", line);
        println!("  ExecPlan");
        println!("{}", line);
        for node in &self.nodes {
            match node {
                ExecNode::Allocate { class, pool, bytes } => {
                    println!("  Allocate class={class:?} pool={pool:?} bytes={bytes}");
                }
                ExecNode::Copy { dst_class, src_class, bytes } => {
                    println!("  Copy dst={dst_class:?} src={src_class:?} bytes={bytes}");
                }
                ExecNode::Deallocate { class } => {
                    println!("  Deallocate class={class:?}");
                }
                ExecNode::Launch { program_id, load_classes, store_classes } => {
                    println!("  Launch prog={program_id:?} loads={load_classes:?} stores={store_classes:?}");
                }
            }
        }
        println!("{}\n", line);
    }
}

impl Runtime {
    pub fn execute_plan(&mut self, cache_key: u64, class_buf: &mut Map<ClassId, BufferId>) -> Result<(), ZyxError> {
        let plan = self.plan_cache.get(&cache_key).unwrap();

        for node in &plan.nodes {
            match node {
                ExecNode::Allocate { class, pool, bytes } => {
                    let (buf, event) = self.pools[*pool].allocate(*bytes)?;
                    let buf_id = BufferId { pool: *pool, buffer: buf };
                    class_buf.insert(*class, buf_id);
                    self.events.insert(BTreeSet::from([buf_id]), event);
                }
                ExecNode::Launch { program_id, load_classes, store_classes } => {
                    let pool_id = self.devices[program_id.device].memory_pool_id();
                    let mut args = Vec::new();
                    let mut kernel_bufs = BTreeSet::new();
                    for c in load_classes.iter().chain(store_classes.iter()) {
                        let buf = class_buf[c];
                        args.push(buf.buffer);
                        kernel_bufs.insert(buf);
                    }
                    let wait_list = drain_events_for_bufs(&mut self.events, &kernel_bufs);
                    let event =
                        self.devices[program_id.device].launch(program_id.program, &mut self.pools[pool_id], &args, wait_list)?;
                    self.events.insert(kernel_bufs, event);
                }
                ExecNode::Copy { dst_class, src_class, bytes } => {
                    let src = class_buf[src_class];
                    let dst = class_buf[dst_class];
                    let wait_list = drain_events_for_buf(&mut self.events, src);
                    let mut tmp = vec![0u8; *bytes as usize];
                    self.pools[src.pool].pool_to_host(src.buffer, &mut tmp, wait_list)?;
                    let event = self.pools[dst.pool].host_to_pool(&tmp, dst.buffer, vec![])?;
                    self.pools[dst.pool].sync_events(vec![event])?;
                }
                ExecNode::Deallocate { class } => {
                    let buf = class_buf.remove(class).unwrap();
                    let wait_list = drain_events_for_buf(&mut self.events, buf);
                    self.pools[buf.pool].deallocate(buf.buffer, wait_list);
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
