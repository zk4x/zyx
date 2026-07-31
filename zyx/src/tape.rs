//! Tape-scoped lazy graph for autograd and optimization.
//!
//! [`Tape::new`] creates a lazy computation graph. Operations on promoted tensors
//! build graph nodes instead of executing eagerly. The same graph is shared by
//! the forward pass and autograd — no separate autograd graph.
//!
//! The tape serves two purposes:
//! 1. **Autograd boundary**: Tensors promoted via [`Tape::new`] are retained for
//!    backward pass until the tape is dropped.
//! 2. **Graph caching boundary**: On drop, all alive tensors are realized together.
//!    Promoted tensors are treated as graph inputs — their buffers change each
//!    iteration (e.g. model parameters, inputs, targets). Everything computed from
//!    them inside the scope is static and cached by structural hash across iterations.
//!
//! Think of [`Tape::new(&model)`] as setting `requires_grad` on the model's tensors
//! for the duration of the scope — but it's not only for gradients. The tape also
//! enables egraph-based fusion optimization, device allocation search, and plan
//! caching across structurally identical iterations.
//!
//! ## Caching with Merkle hashes
//!
//! Each graph node carries a Merkle hash of its structural subgraph (node kind, dtype,
//! shape, input hashes — no TensorIds). When the tape realizes all alive tensors on
//! drop, it uses the output tensors' Merkle hashes as the cache key:
//!
//! - **Cache miss** (first pass): compile the subgraph, store the compiled kernel with
//!   its static leaf→buffer bindings.
//! - **Cache hit** (subsequent passes): structural hash match means the same kernel
//!   applies. Only resolve the boundary-crossing leaf buffers. No graph traversal
//!   for the full subgraph — just collect the leaf TensorIds and map to their current
//!   BufferIds.

use std::collections::BTreeSet;

use crate::{
    DType, Map, RT, Tensor, ZyxError,
    backend::{BufferId, Device},
    graph::{ClassId, ExecPlan, Graph, GraphId, Node},
    kernel::{DeviceId, Kernel, Op},
    runtime::{KernelData, KernelId, ShapeId, TensorState},
    shape::Dim,
    slab::{Slab, SlabId},
    tensor::TensorId,
    view::View,
};

/// Tape-scoped lazy graph.
///
/// Promotes tensors to graph mode for autograd and egraph optimization.
/// All alive tensors are realized when the tape is dropped.
/// The Merkle hash cache avoids recompilation on structurally identical iterations.
#[cfg_attr(feature = "py", pyo3::pyclass)]
pub struct Tape {
    graph_id: GraphId,
}

impl Tape {
    /// Create a tape scope, promoting the given tensors to graph mode.
    ///
    /// This is like setting `requires_grad` on those tensors for the scope's
    /// duration — but it's not only for gradients. The tape also enables
    /// egraph-based fusion optimization, device allocation search, and plan
    /// caching across structurally identical iterations.
    ///
    /// Typically you pass the model: `Tape::new(&model)?` promotes all its
    /// parameters. Input tensors (x, target) are auto-detected as boundary
    /// inputs — they don't need to be promoted explicitly.
    pub fn new<'a>(params: impl IntoIterator<Item = &'a Tensor>) -> Result<Tape, ZyxError> {
        let mut rt = RT.lock();

        let graph_id = rt.graphs.push(Graph::new());

        for p in params {
            rt.promote_to_graph(p.id, graph_id)?;
        }

        Ok(Tape { graph_id })
    }

    /// Create a tape scope without promoting any tensors yet.
    /// Use [`Tape::add`] or [`Tape::extend`] to promote tensors later.
    pub fn empty() -> Tape {
        Self::new(std::iter::empty()).unwrap()
    }

    /// Promote a tensor into the tape's graph scope.
    /// All ops on this tensor from now on will be tracked in the graph.
    pub fn add(&self, tensor: &Tensor) -> Result<(), ZyxError> {
        let mut rt = RT.lock();
        rt.promote_to_graph(tensor.id, self.graph_id)?;
        Ok(())
    }

    /// Promote multiple tensors into the tape's graph scope at once.
    pub fn extend<'a>(&self, params: impl IntoIterator<Item = &'a Tensor>) -> Result<(), ZyxError> {
        let mut rt = RT.lock();
        for p in params {
            rt.promote_to_graph(p.id, self.graph_id)?;
        }
        Ok(())
    }
}

impl Tape {
    /// Returns gradients of target derived w.r.t. sources.
    /// Non-differentiable paths return a zero tensor.
    #[must_use]
    pub fn gradient<'a>(&self, target: &Tensor, sources: impl IntoIterator<Item = &'a Tensor>) -> Vec<Tensor> {
        let sources: Vec<TensorId> = sources.into_iter().map(Tensor::id).collect();
        let mut rt = RT.lock();
        let grads: Map<TensorId, TensorId> = rt.gradient(target.id(), sources.iter().copied().collect(), self.graph_id);
        sources
            .into_iter()
            .map(|x: TensorId| {
                let id = match grads.get(&x) {
                    Some(&id) => id,
                    None => {
                        let shape = rt.shape(x).into();
                        let dtype = rt.dtype(x);
                        rt.new_full(shape, dtype.zero_constant())
                    }
                };
                Tensor { id }
            })
            .collect()
    }

    /// Materializes the given graph tensors by compiling and executing the
    /// subgraph they depend on. The tape is consumed — graph mode ends and
    /// all output tensors become realized (buffers allocated).
    pub fn realize<'a>(self, tensors: impl IntoIterator<Item = &'a Tensor>) -> Result<(), ZyxError> {
        let mut rt = RT.lock();
        let graph_id = self.graph_id;

        let output_pairs: Vec<(TensorId, ClassId)> = tensors
            .into_iter()
            .map(|t| match rt.tensors[t.id].state {
                TensorState::Graph { class_id, .. } => (t.id, class_id),
                _ => unreachable!("non-graph tensor in realize"),
            })
            .collect();

        let output_tids: Vec<TensorId> = output_pairs.iter().map(|(tid, _)| *tid).collect();
        let output_classes: Vec<ClassId> = output_pairs.iter().map(|(_, cid)| *cid).collect();

        debug_assert!(rt.graphs.contains_key(graph_id));

        let output_set: BTreeSet<ClassId> = output_classes.iter().copied().collect();
        let cache_key = rt.graphs[graph_id].cache_key(&output_set);

        if let Some(plan) = rt.plan_cache.get(&cache_key) {
            let mut class_buf: Map<ClassId, BufferId> = Map::default();
            for &cid in &plan.leaf_classes {
                let &tid = rt.graphs[graph_id].leaf_map.get(&cid).unwrap();
                debug_assert!(rt.buffer_map.contains_key(&tid), "leaf class {cid:?} tid {tid:?} not in buffer_map");
                class_buf.insert(cid, rt.buffer_map[&tid]);
            }

            rt.execute_plan(cache_key, &mut class_buf)?;
            for (&tid, &cid) in output_tids.iter().zip(output_classes.iter()) {
                rt.buffer_map.insert(tid, class_buf[&cid]);
                rt.eagerify(tid);
            }

            return Ok(());
        }

        // TODO pattern match cublas, cblas, etc. kernels

        for cid in rt.graphs[graph_id].classes.ids() {
            let has_leaf = rt.graphs[graph_id].classes[cid]
                .nodes
                .iter()
                .any(|&nid| matches!(&rt.graphs[graph_id].nodes[nid].node, Node::Leaf { .. }));
            if has_leaf {
                let &tid = rt.graphs[graph_id].leaf_map.get(&cid).expect("class {cid:?} has Leaf node but not in leaf_map");
                assert!(rt.buffer_map.contains_key(&tid), "leaf class {cid:?} tid {tid:?} not in buffer_map");
            } else {
                assert!(!rt.graphs[graph_id].leaf_map.contains_key(&cid), "class {cid:?} has no Leaf node but is in leaf_map");
            }
        }

        // Fills missing places with zyx custom kernels
        // SAFETY: graph and shapes are separate fields of Runtime, no aliasing, rust is stupid
        let shapes_ptr: *const Slab<ShapeId, Vec<Dim>> = &rt.shapes;

        // TODO debug assert that all leafs are realized
        //let realized_nodes: Set<ClassId> = rt.graphs[graph_id].leaf_map.iter().filter(|(_, tid)| rt.buffer_map.contains_key(tid)).map(|(cid, _)| *cid).collect();
        rt.graphs[graph_id].fill_remaining(&output_set, unsafe { &*shapes_ptr });

        // Autotunes custom zyx kernels for all devices and adds kernel nodes for all of them
        rt.autotune_graph_ekernels(graph_id)?;

        // After all kernels nodes are added, this adds movement ops so extract can pick fastest path
        let devices_ptr: *const Slab<DeviceId, Device> = &rt.devices;
        let buffer_map_ptr: *const Map<TensorId, BufferId> = &rt.buffer_map;
        rt.graphs[graph_id].add_memory_ops(unsafe { &*devices_ptr }, unsafe { &*buffer_map_ptr });

        rt.graphs[graph_id].debug_print(&rt.shapes);

        let nodes = rt.graphs[graph_id].extract(&output_set);

        let plan = ExecPlan::new(&rt.graphs[graph_id], &nodes, &output_set, &rt.devices, &rt.shapes);
        //plan.debug();

        let mut class_buf: Map<ClassId, BufferId> = Map::default();
        for &cid in &plan.leaf_classes {
            let &tid = rt.graphs[graph_id].leaf_map.get(&cid).unwrap();
            debug_assert!(rt.buffer_map.contains_key(&tid), "leaf class {cid:?} tid {tid:?} not in buffer_map");
            class_buf.insert(cid, rt.buffer_map[&tid]);
        }

        rt.plan_cache.insert(cache_key, plan);

        rt.execute_plan(cache_key, &mut class_buf)?;
        for (&tid, &cid) in output_tids.iter().zip(output_classes.iter()) {
            rt.buffer_map.insert(tid, class_buf[&cid]);
            rt.eagerify(tid);
        }

        Ok(())
    }

    // TOOD unsure if this should even be provided
    // Materializes ALL graph tensors still alive in the tape scope.
    // The tape is consumed — graph mode ends and every tracked tensor
    // becomes realized (buffers allocated).
    /*pub fn realize_all(self) -> Result<(), ZyxError> {
        todo!()
    }*/
}

impl Drop for Tape {
    fn drop(&mut self) {
        let mut rt = RT.lock();
        rt.graphs.remove(self.graph_id);

        // TODO this is wrong, cleanup should only clean up graph.leafs and outputs,
        // which are accessible in realize and replay methods.
        // We should also debug assert that no intermediate tensors are kept alive
        // after graph realize or replay is finished.

        let tids: Vec<TensorId> = rt.tensors.iter().map(|(id, _)| id).collect();
        for tid in tids {
            let (rc, is_my_graph) = match &rt.tensors[tid].state {
                TensorState::Graph { rc, graph_id, .. } => (*rc, *graph_id == self.graph_id),
                _ => (0, false),
            };
            if !is_my_graph {
                continue;
            }
            if rc == 0 {
                if let Some(buf_id) = rt.buffer_map.remove(&tid) {
                    let keys: Vec<BTreeSet<BufferId>> = rt.events.keys().filter(|k| k.contains(&buf_id)).cloned().collect();
                    let mut wait_list = Vec::new();
                    for key in keys {
                        wait_list.push(rt.events.remove(&key).unwrap());
                    }
                    rt.pools[buf_id.pool].deallocate(buf_id.buffer, wait_list);
                }
                rt.tensors.remove(tid);
            } else if rt.buffer_map.contains_key(&tid) {
                let shape: Vec<Dim> = rt.shape(tid).into();
                let dtype = rt.dtype(tid);
                let op = Op::LoadView(Box::new((dtype, View::contiguous(&shape))));
                let kernel_id = rt.kernels.push(KernelData {
                    outputs: Vec::new(),
                    loads: Vec::new(),
                    stores: Vec::new(),
                    kernel: Kernel::new(DeviceId::AUTO),
                });
                let op_id = rt.kernels[kernel_id].kernel.push_back(op);
                rt.kernels[kernel_id].loads.push(tid);
                rt.tensors[tid].state = TensorState::Eager { kernel_id, op_id, pending: KernelId::NULL };
                for _ in 0..rc {
                    rt.kernels[kernel_id].outputs.push(tid);
                }
            }
        }
    }
}

impl Tape {
    /// Create frozen tape (fixed control flow, minimum overhead)
    pub fn freeze<'a>(self, outputs: impl IntoIterator<Item = &'a Tensor>) -> Result<FrozenTape, ZyxError> {
        let mut rt = RT.lock();
        let graph_id = self.graph_id;

        let outputs: Vec<(ClassId, Vec<Dim>, DType)> = outputs
            .into_iter()
            .map(|t| match rt.tensors[t.id].state {
                TensorState::Graph { class_id, .. } => (class_id, rt.shape(t.id).into(), rt.dtype(t.id)),
                _ => unreachable!("non-graph tensor in realize"),
            })
            .collect();

        debug_assert!(rt.graphs.contains_key(graph_id));

        let output_set: BTreeSet<ClassId> = outputs.iter().map(|x| x.0).collect();
        let cache_key = rt.graphs[graph_id].cache_key(&output_set);

        if rt.plan_cache.contains_key(&cache_key) {
            return Ok(FrozenTape { cache_key, outputs });
        }

        // TODO pattern match cublas, cblas, etc. kernels

        for cid in rt.graphs[graph_id].classes.ids() {
            let has_leaf = rt.graphs[graph_id].classes[cid]
                .nodes
                .iter()
                .any(|&nid| matches!(&rt.graphs[graph_id].nodes[nid].node, Node::Leaf { .. }));
            if has_leaf {
                let &tid = rt.graphs[graph_id].leaf_map.get(&cid).expect("class {cid:?} has Leaf node but not in leaf_map");
                assert!(rt.buffer_map.contains_key(&tid), "leaf class {cid:?} tid {tid:?} not in buffer_map");
            } else {
                assert!(!rt.graphs[graph_id].leaf_map.contains_key(&cid), "class {cid:?} has no Leaf node but is in leaf_map");
            }
        }

        // Fills missing places with zyx custom kernels
        // SAFETY: graph and shapes are separate fields of Runtime, no aliasing, rust is stupid
        let shapes_ptr: *const Slab<ShapeId, Vec<Dim>> = &rt.shapes;

        // TODO debug assert that all leafs are realized
        //let realized_nodes: Set<ClassId> = rt.graphs[graph_id].leaf_map.iter().filter(|(_, tid)| rt.buffer_map.contains_key(tid)).map(|(cid, _)| *cid).collect();
        rt.graphs[graph_id].fill_remaining(&output_set, unsafe { &*shapes_ptr });

        // Autotunes custom zyx kernels for all devices and adds kernel nodes for all of them
        rt.autotune_graph_ekernels(graph_id)?;

        // After all kernels nodes are added, this adds movement ops so extract can pick fastest path
        let devices_ptr: *const Slab<DeviceId, Device> = &rt.devices;
        let buffer_map_ptr: *const Map<TensorId, BufferId> = &rt.buffer_map;
        rt.graphs[graph_id].add_memory_ops(unsafe { &*devices_ptr }, unsafe { &*buffer_map_ptr });

        rt.graphs[graph_id].debug_print(&rt.shapes);

        let nodes = rt.graphs[graph_id].extract(&output_set);

        let plan = ExecPlan::new(&rt.graphs[graph_id], &nodes, &output_set, &rt.devices, &rt.shapes);
        //plan.debug();

        rt.plan_cache.insert(cache_key, plan);

        return Ok(FrozenTape { cache_key, outputs });
    }
}

/// Frozen tape for minimal overhead tape replay, no branching
pub struct FrozenTape {
    cache_key: u64,
    outputs: Vec<(ClassId, Vec<Dim>, DType)>,
}

impl FrozenTape {
    /// Replay the tape
    pub fn replay<'a>(&self, inputs: impl IntoIterator<Item = &'a Tensor>) -> Result<Vec<Tensor>, ZyxError> {
        let mut rt = RT.lock();

        let mut class_buf: Map<ClassId, BufferId> = Map::default();
        for (tid, &cid) in inputs.into_iter().zip(rt.plan_cache[&self.cache_key].leaf_classes.iter()) {
            class_buf.insert(cid, rt.buffer_map[&tid.id]);
        }

        rt.execute_plan(self.cache_key, &mut class_buf)?;

        let output_tids = Vec::new();
        for (cid, shape, dtype) in self.outputs.iter() {
            let view = View::contiguous(shape);
            let tid = rt.new_kernel(Op::LoadView(Box::new((*dtype, view))));
            rt.buffer_map.insert(tid, class_buf[cid]);
        }

        Ok(output_tids)
    }
}
