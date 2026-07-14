use std::collections::BTreeSet;

use crate::{
    Map, Set,
    backend::{Device, DeviceId, PoolId, ProgramId},
    graph::{ClassId, Graph, Node, NodeId},
    runtime::ShapeId,
    shape::Dim,
    slab::Slab,
};

#[derive(Debug)]
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

#[derive(Debug)]
pub struct ExecPlan {
    pub nodes: Vec<ExecNode>,
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

        let leaf_classes: BTreeSet<ClassId> = graph.leaf_map.keys().copied().collect();
        let mut plan_nodes = Vec::new();
        let mut allocated: Set<ClassId> = Set::default();

        let class_bytes: Map<ClassId, Dim> = graph
            .classes
            .ids()
            .map(|cid| {
                let class = &graph.classes[cid];
                let shape = &shapes[class.shape];
                let numel: Dim = shape.iter().product();
                let bytes = (numel * class.dtype.bit_size() as Dim + 7) / 8;
                (cid, bytes)
            })
            .collect();

        let device_pool_map: Map<DeviceId, PoolId> = devices.ids().map(|did| (did, devices[did].memory_pool_id())).collect();

        for &nid in nodes {
            match &graph.nodes[nid].node {
                Node::Kernel { inputs, outputs, program_id, .. } => {
                    for &oc in &**outputs {
                        if allocated.insert(oc) {
                            plan_nodes.push(ExecNode::Allocate {
                                class: oc,
                                pool: device_pool_map[&program_id.device],
                                bytes: class_bytes[&oc],
                            });
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
                        if *c == 0 && !leaf_classes.contains(&ic) && !output_set.contains(&ic) {
                            plan_nodes.push(ExecNode::Deallocate { class: ic });
                        }
                    }
                }
                Node::ToDevice { x, device, .. } => {
                    let class_of = graph.nodes[nid].class_of;
                    if allocated.insert(class_of) {
                        plan_nodes.push(ExecNode::Allocate {
                            class: class_of,
                            pool: device_pool_map[device],
                            bytes: class_bytes[&class_of],
                        });
                    }
                    plan_nodes.push(ExecNode::Copy { dst_class: class_of, src_class: *x, bytes: class_bytes[&class_of] });
                    let c = rc.get_mut(x).unwrap();
                    *c -= 1;
                    if *c == 0 && !leaf_classes.contains(x) && !output_set.contains(x) {
                        plan_nodes.push(ExecNode::Deallocate { class: *x });
                    }
                }
                _ => unreachable!(),
            }
        }

        Self { nodes: plan_nodes }
    }

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
