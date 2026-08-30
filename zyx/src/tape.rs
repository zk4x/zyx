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
    dtype::Constant,
    graph::{ClassId, Graph, GraphId},
    kernel::ParamKind,
    runtime::{Runtime, TensorData},
    shape::Dim,
    slab::SlabId,
    tensor::TensorId,
};

/// Tape-scoped lazy graph.
///
/// Promotes tensors to graph mode for autograd and egraph optimization.
/// All alive tensors are realized when the tape is dropped.
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
                        let shape = rt.resolve_shape(x);
                        let dtype = rt.dtype(x);
                        let ids: Vec<TensorId> = shape.iter().map(|&d| rt.new_constant_tensor(Constant::idx(d))).collect();
                        let stid = if ids.is_empty() {
                            TensorId::NULL
                        } else {
                            let s = rt.stack(&ids).unwrap();
                            for id in &ids {
                                rt.release(*id);
                            }
                            s
                        };
                        rt.new_full(stid, dtype.zero_constant())
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
                let class_id = match rt.tensors[t.id] {
                    TensorData::Graph { class_id, .. } | TensorData::Promoted { class_id, .. } => class_id,
                    // NOTE: never format the `Tensor` itself here (Display
                    // clones + re-locks RT, which deadlocks under this guard);
                    // `TensorData`'s Debug is lock-free.
                    ref td => panic!(
                        "Tape::realize was given a tensor that never entered the tape's graph \
                         (tid {}, data {:?}).\n\
                         This is a caller mistake, not a zyx bug: the tensor is eager — it was \
                         created outside the tape scope, or built entirely from eager inputs, so \
                         there is no graph class for realize to materialize.\n\
                         How to fix: realize only tensors whose computation this tape traced. \
                         Promote tensors you build from with `tape.add(&t)?` before the ops, or \
                         give the chain at least one promoted operand — ops mixing an eager tensor \
                         with a graph tensor are pulled into the graph automatically; an all-eager \
                         chain stays eager.",
                        t.id(),
                        td
                    ),
                };
                (t.id, class_id)
            })
            .collect();

        let output_tids: Vec<TensorId> = output_pairs.iter().map(|(tid, _)| *tid).collect();
        let output_classes: Vec<ClassId> = output_pairs.iter().map(|(_, cid)| *cid).collect();

        debug_assert!(rt.graphs.contains_id(graph_id));
        rt.debug_assert_pre_realize(graph_id);

        let output_set: BTreeSet<ClassId> = output_classes.iter().copied().collect();
        let cache_key = rt.plan_cache_key(graph_id, &output_set);

        if let Some(plan) = rt.plan_cache.get(&cache_key) {
            let mut class_buf: Map<ClassId, BufferId> = Map::default();
            let mut class_vars: Map<ClassId, Constant> = Map::default();
            for &cid in &plan.leaf_classes {
                let &tid = rt.graphs[graph_id].leaf_map.get(&cid).unwrap();
                if let Some(&buf_id) = rt.buffer_map.get(&tid) {
                    class_buf.insert(cid, buf_id);
                } else {
                    // Variable leaf: no buffer anywhere; its scalar value
                    // resolves from variable_map (directly or symbolically).
                    let value = rt.resolve_symbolic(tid).expect("leaf class tid resolves neither to a buffer nor a variable");
                    class_vars.insert(cid, value);
                }
            }

            rt.execute_plan(cache_key, &mut class_buf, &class_vars)?;
            for (&tid, &cid) in output_tids.iter().zip(output_classes.iter()) {
                rt.buffer_map.insert(tid, class_buf[&cid]);
                rt.eagerify(tid);
            }
            rt.debug_assert_no_stray_buffers(graph_id, &output_tids);

            return Ok(());
        }

        let plan = rt.compile_graph(graph_id, &output_set)?;

        let mut class_buf: Map<ClassId, BufferId> = Map::default();
        let mut class_vars: Map<ClassId, Constant> = Map::default();
        for &cid in &plan.leaf_classes {
            let &tid = rt.graphs[graph_id].leaf_map.get(&cid).unwrap();
            if let Some(&buf_id) = rt.buffer_map.get(&tid) {
                class_buf.insert(cid, buf_id);
            } else {
                // Variable leaf: no buffer anywhere; its scalar value
                // resolves from variable_map (directly or symbolically).
                let value = rt.resolve_symbolic(tid).expect("leaf class tid resolves neither to a buffer nor a variable");
                class_vars.insert(cid, value);
            }
        }

        rt.plan_cache.insert(cache_key, plan);

        rt.execute_plan(cache_key, &mut class_buf, &class_vars)?;
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
        /*eprintln!(
            ">>> Tape::drop graph={graph_id:?} ref_count={} leaf_map_len={}",
            rt.graphs[graph_id].ref_count,
            rt.graphs[graph_id].leaf_map.len()
        );*/

        // Revert every tensor still affiliated with this graph back to a state
        // that does not reference the (about-to-die) graph:
        //   - `Promoted` keeps its eager side, so it reverts to `Eager`.
        //   - pure `Graph` has no eager side; it can't be realized, so we just
        //     clear its `graph_id`/`class_id`, orphaning it as a dead handle
        //     that panics on use (and is freed by `release` when its handle drops).
        // We scan the whole tensor slab (a drop is rare next to kernel launches,
        // and the slab is bounded by the allocation high-water mark) rather than
        // tracking affiliations in a per-graph set.
        // TODO: if this full scan ever shows up as a perf bottleneck, replace it
        // with a `Vec<TensorId>` of affiliated tensors kept on `Graph` and pushed
        // in every `promote_to_graph` path (leaf + non-buffer), iterating that
        // instead. A debug-only full scan can stay to assert the set is complete.
        // Leaf edges: one per leaf_map OCCURRENCE — the same tensor id may be
        // promoted multiple times (e.g. as a tape param and again by an in-scope
        // op), and each occurrence carries its own retain. Do NOT deduplicate.
        let leafs: Vec<TensorId> = rt.graphs[graph_id].leaf_map.values().copied().collect();
        // A leaf that is already `Eager` was converted by `realize`'s output
        // eagerify — its graph affiliation (and the ref_count decrement) was
        // deleted then. The visit loop's `Eager` arm exists for tensors
        // re-homed mid-drop by add_store cascades and must not decrement
        // these a second time.
        let eager_leafs: Set<TensorId> =
            leafs.iter().copied().filter(|&tid| matches!(rt.tensors[tid], TensorData::Eager { .. })).collect();
        let affiliated: Vec<TensorId> = rt
            .tensors
            .iter()
            .filter_map(|(tid, td)| match td {
                TensorData::Graph { graph_id: g, .. } | TensorData::Promoted { graph_id: g, .. } if *g == graph_id => {
                    if leafs.contains(&tid) {
                        None
                    } else {
                        Some(tid)
                    }
                }
                _ => None,
            })
            .collect();
        for &tid in affiliated.iter().chain(&leafs) {
            if !rt.tensors.contains_id(tid) {
                // Already dead: a disowned tensor released earlier in this loop
                // can cascade (shared producer kernel dies, its load releases
                // kill its other disowned inputs). Legitimate — its death path
                // already cleared the graph affiliation.
                continue;
            }
            match rt.tensors[tid] {
                TensorData::Promoted { rc, kernel_id, .. } => {
                    if rc > 0 && !kernel_id.is_null() {
                        // Disowned detection: rc equals x's entry count in its own
                        // kernel ⟺ every remaining reference is a kernel load
                        // edge (no handles, no other kernels' entries). A
                        // disowned tensor is *not* in its kernel's `outputs`
                        // (disown removed it); a handle-held one is.
                        let n = rt.kernels[kernel_id].loads.iter().filter(|&&t| t == tid).count() as u16;
                        let disowned = n > 0 && rc == n && !rt.kernels[kernel_id].outputs.contains(&tid);
                        if disowned {
                            // The user handle is gone and the only remaining
                            // references are the producer kernel's load edges.
                            // Nothing to preserve — release it; the death path
                            // clears the graph affiliation.
                            rt.release(tid);
                        } else {
                            rt.eagerify(tid);
                        }
                    }
                }
                TensorData::Graph { graph_id: _g, rc, .. } => {
                    if rc > 0 {
                        rt.graphs[graph_id].ref_count -= 1;
                        match &mut rt.tensors[tid] {
                            TensorData::Graph { graph_id, class_id, .. } => {
                                *graph_id = GraphId::NULL;
                                *class_id = ClassId::NULL;
                            }
                            _ => unreachable!(),
                        }
                    }
                }
                TensorData::Eager { .. } => {
                    // Converted after collection by `add_store` — a materialize
                    // cascade triggered by an earlier release in this loop
                    // re-homed it as a pure eager load. It no longer holds the
                    // graph variant; just drop the graph's affiliation count.
                    // Leafs already `Eager` at collection were eagerified by
                    // `realize` (see `eager_leafs`): their edge is already gone.
                    if !eager_leafs.contains(&tid) {
                        rt.graphs[graph_id].ref_count -= 1;
                    }
                }
                TensorData::Variable { .. } => {
                    // A variable leaf carries no graph state in its
                    // TensorData — only its leaf edge (retain + ref_count).
                    // Drop the ref_count edge; the `leafs` loop below
                    // releases the retain.
                    rt.graphs[graph_id].ref_count -= 1;
                }
                ref t => unreachable!("affiliated tensor changed variant: {t:?}"),
            };
        }
        for tid in leafs {
            if rt.tensors.contains_id(tid) {
                rt.release(tid);
            }
        }

        for tid in affiliated {
            if !rt.tensors.contains_id(tid) {
                // Disowned tensor, released during the revert loop (directly or
                // via a kernel-death cascade) — legitimate death.
                continue;
            }
            let rc = match rt.tensors[tid] {
                TensorData::Promoted { rc, .. } | TensorData::Graph { rc, .. } => rc,
                TensorData::Eager { .. } => continue,
                _ => panic!("affiliated wrong"),
            };
            if rc == 0 {
                panic!("How is this possible?");
                /*if let Some(buf_id) = rt.buffer_map.remove(tid) {
                    let still_used = rt.buffer_map.values().any(|b| b.pool == buf_id.pool && b.buffer == buf_id.buffer);
                    if !still_used {
                        let wait_list = drain_events_for_buf(&mut rt.events, buf_id);
                        rt.pools[buf_id.pool].deallocate(buf_id.buffer, wait_list);
                    }
                }
                rt.graphs[graph_id].ref_count -= 1;
                rt.tensors.remove(*tid);*/
            }
        }

        // The affiliation invariant must hold before the graph is torn down:
        // ref_count equals the number of live tensors still pointing at it.
        rt.assert_graph_inventory(graph_id);

        rt.graphs[graph_id].mark_dead();

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
                let class_id = match rt.tensors[t.id] {
                    TensorData::Graph { class_id, .. } | TensorData::Promoted { class_id, .. } => class_id,
                    ref td => panic!("non-graph tensor in freeze: tid {t} data {td:?}"),
                };
                (class_id, rt.resolve_shape(t.id), rt.dtype(t.id))
            })
            .collect();

        debug_assert!(rt.graphs.contains_id(graph_id));
        rt.debug_assert_pre_realize(graph_id);

        let output_set: BTreeSet<ClassId> = outputs.iter().map(|x| x.0).collect();
        let cache_key = rt.plan_cache_key(graph_id, &output_set);

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
        let mut class_vars: Map<ClassId, Constant> = Map::default();
        for (tid, &cid) in inputs.into_iter().zip(rt.plan_cache[&self.cache_key].leaf_classes.iter()) {
            // The frozen contract: leaf bindings are fixed since `freeze` — a
            // compiled plan bakes pool-dependent decisions (ExecPlan::new's
            // cross-pool alias handling), so replaying with a leaf buffer in a
            // different pool would execute a wrong plan. Loud error instead:
            // re-freeze the tape.
            if let Some(&buf_id) = rt.buffer_map.get(&tid.id) {
                let expected = rt.plan_cache[&self.cache_key].leaf_pools.get(&cid).copied();
                if expected != Some(buf_id.pool) {
                    return Err(ZyxError::frozen_plan_stale(
                        format!(
                            "frozen tape replayed with leaf class {cid:?} in pool {:?}, but the frozen plan compiled it in pool {:?} — bindings changed since freeze, re-freeze the tape",
                            buf_id.pool,
                            expected
                        )
                        .into(),
                    ));
                }
                class_buf.insert(cid, buf_id);
            } else {
                // Variable leaf: no buffer anywhere; its scalar value
                // resolves from variable_map (directly or symbolically).
                let value = rt.resolve_symbolic(tid.id).expect("replay input resolves neither to a buffer nor a variable");
                class_vars.insert(cid, value);
            }
        }

        rt.execute_plan(self.cache_key, &mut class_buf, &class_vars)?;

        let mut outputs = Vec::new();
        for (cid, shape, dtype) in self.outputs.iter() {
            let ids: Vec<TensorId> = shape.iter().map(|&d| rt.new_constant_tensor(Constant::idx(d))).collect();
            let stid = if ids.is_empty() {
                TensorId::NULL
            } else {
                let s = rt.stack(&ids).unwrap();
                for id in &ids {
                    rt.release(*id);
                }
                s
            };
            let tid = rt.new_eager_tensor(stid, *dtype, ParamKind::Global);
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
                let (affiliated, class_id) = match td {
                    TensorData::Graph { class_id: c, graph_id: g, .. }
                    | TensorData::Promoted { class_id: c, graph_id: g, .. } => (*g == graph_id, *c),
                    _ => continue,
                };
                if affiliated
                    && !output_set.contains(&tid)
                    && !self.graphs[graph_id].is_leaf(class_id)
                    && !self.graphs[graph_id].is_after(class_id)
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
