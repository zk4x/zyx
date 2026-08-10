//! Tape-scoped lazy graph for autograd and optimization.
//!
//! [`Tape::new`] creates a lazy computation graph. Operations on promoted tensors
//! build graph nodes instead of executing eagerly. The same graph is shared by
//! the forward pass and autograd — no separate autograd graph.
//!
//! The tape serves two purposes:
//! 1. **Autograd boundary**: Tensors promoted via [`Tape::new`] are retained for
//!    backward pass until the tape is dropped.
//! 2. **Graph caching boundary**: [`Tape::realize`] realizes the requested output
//!    tensors. Promoted tensors are treated as graph inputs — their buffers change
//!    each iteration (e.g. model parameters, inputs, targets). Everything computed
//!    from them inside the scope is static and cached by structural hash across
//!    iterations.
//!
//! Think of [`Tape::new(&model)`] as setting `requires_grad` on the model's tensors
//! for the duration of the scope — but it's not only for gradients. The tape also
//! enables egraph-based fusion optimization, device allocation search, and plan
//! caching across structurally identical iterations.
//!
//! ## Lifecycle and invariants
//!
//! - **Graph construction** (`Tape::new` until `realize`/`freeze`): ops only build
//!   nodes, never compute. The only realized graph tensors are leaves — the tensors
//!   promoted by `Tape::new` (I2). No other graph tensor may hold a buffer.
//! - **`realize`/`replay`**: the only places that compute. `realize` eagerifies its
//!   output tensors; all other buffers belong to leaves or are released (I3, I4).
//! - **`Drop`**: marks the graph dead, converts alive leaves back to eager, removes
//!   dead leaves. It performs no computation and no scans (I5).
//! - **Reference counting**: every alive graph tensor counts toward its
//!   [`Graph::ref_count`]. The graph stays in the runtime slab until
//!   `dead && ref_count == 0`, so a stale tensor can never observe a reused
//!   [`GraphId`]. Using a tensor from a dead graph panics with "tape scope has
//!   ended".
//!
//! ## Caching with Merkle hashes
//!
//! Each graph node carries a Merkle hash of its structural subgraph (node kind, dtype,
//! shape, input hashes — no TensorIds). `realize` uses the output tensors' Merkle
//! hashes as the cache key:
//!
//! - **Cache miss** (first pass): compile the subgraph, store the compiled kernel with
//!   its static leaf→buffer bindings.
//! - **Cache hit** (subsequent passes): structural hash match means the same kernel
//!   applies. Only resolve the boundary-crossing leaf buffers. No graph traversal
//!   for the full subgraph — just collect the leaf TensorIds and map to their current
//!   BufferIds.

use std::collections::BTreeSet;

use crate::{
    DType, Map, RT, Set, Tensor, ZyxError,
    backend::BufferId,
    graph::{ClassId, Graph, GraphId, plan::drain_events_for_buf},
    runtime::Runtime,
    shape::Dim,
    slab::SlabId,
    tensor::TensorId,
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
            .map(|t| {
                if rt.tensors[t.id].class_id.is_null() {
                    panic!("non-graph tensor in realize")
                } else {
                    (t.id, rt.tensors[t.id].class_id)
                }
            })
            .collect();

        let output_tids: Vec<TensorId> = output_pairs.iter().map(|(tid, _)| *tid).collect();
        let output_classes: Vec<ClassId> = output_pairs.iter().map(|(_, cid)| *cid).collect();

        debug_assert!(rt.graphs.contains_key(graph_id));
        rt.debug_assert_pre_realize(graph_id);

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
            rt.debug_assert_no_stray_buffers(graph_id, &output_tids);

            return Ok(());
        }

        let plan = rt.compile_graph(graph_id, &output_set)?;

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
        rt.debug_assert_no_stray_buffers(graph_id, &output_tids);

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
        let graph_id = self.graph_id;
        rt.graphs[graph_id].dead = true;
        /*eprintln!(
            ">>> Tape::drop graph={graph_id:?} ref_count={} leaf_map_len={}",
            rt.graphs[graph_id].ref_count,
            rt.graphs[graph_id].leaf_map.len()
        );*/

        let leaves: Vec<TensorId> = rt.graphs[graph_id].leaf_map.values().copied().collect();
        for tid in leaves {
            if rt.tensors[tid].graph_id != graph_id {
                continue;
            }
            if rt.tensors[tid].rc == 0 {
                // Dead leaf: rc counts both handles and kernel loads, so rc == 0
                // means nothing references it anymore. Remove it, freeing its
                // buffer if no other tensor maps to the same buffer.
                if let Some(buf_id) = rt.buffer_map.remove(&tid) {
                    let still_used = rt.buffer_map.values().any(|b| b.pool == buf_id.pool && b.buffer == buf_id.buffer);
                    if !still_used {
                        let wait_list = drain_events_for_buf(&mut rt.events, buf_id);
                        rt.pools[buf_id.pool].deallocate(buf_id.buffer, wait_list);
                    }
                }
                rt.tensors.remove(tid);
            } else {
                // Alive leaf: back to eager. Keep its kernel_id/op_id so its
                // value can still be computed; the graph affiliation is what
                // made it a graph tensor, so clearing it reverts the tensor.
                rt.tensors[tid].class_id = ClassId::NULL;
                rt.tensors[tid].graph_id = GraphId::NULL;
                rt.graphs[graph_id].ref_count -= 1;
            }
        }

        if rt.graphs[graph_id].ref_count == 0 {
            rt.remove_dead_graph(graph_id);
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
            .map(|t| {
                if rt.tensors[t.id].class_id.is_null() {
                    panic!("non-graph tensor in realize")
                } else {
                    (rt.tensors[t.id].class_id, rt.shape(t.id).into(), rt.dtype(t.id))
                }
            })
            .collect();

        debug_assert!(rt.graphs.contains_key(graph_id));
        rt.debug_assert_pre_realize(graph_id);

        let output_set: BTreeSet<ClassId> = outputs.iter().map(|x| x.0).collect();
        let cache_key = rt.graphs[graph_id].cache_key(&output_set);

        if rt.plan_cache.contains_key(&cache_key) {
            return Ok(FrozenTape { cache_key, outputs });
        }

        let plan = rt.compile_graph(graph_id, &output_set)?;
        rt.plan_cache.insert(cache_key, plan);

        Ok(FrozenTape { cache_key, outputs })
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

        let mut outputs = Vec::new();
        for (cid, shape, dtype) in self.outputs.iter() {
            let tid = rt.new_eager_tensor(shape.clone(), *dtype);
            rt.buffer_map.insert(tid, class_buf[cid]);
            outputs.push(Tensor::from_id(tid));
        }

        Ok(outputs)
    }
}

impl Runtime {
    fn debug_assert_no_stray_buffers(&self, graph_id: GraphId, outputs: &[TensorId]) {
        if cfg!(debug_assertions) {
            let output_set: Set<TensorId> = outputs.iter().copied().collect();
            for (tid, td) in self.tensors.iter() {
                if td.graph_id == graph_id
                    && !output_set.contains(&tid)
                    && !self.graphs[graph_id].is_leaf(td.class_id)
                    && !self.graphs[graph_id].is_assign_alias(td.class_id)
                {
                    debug_assert!(
                        !self.buffer_map.contains_key(&tid),
                        "non-leaf, non-output graph tensor {tid} realized after execute_plan"
                    );
                }
            }
        }
    }
}
