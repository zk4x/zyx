//! Tape-based scope guards for graph boundary detection.
//!
//! The tape serves two purposes:
//! 1. **Autograd boundary**: Tensors created inside a tape scope are retained for
//!    backward pass until the tape is dropped.
//! 2. **Graph caching boundary**: On drop, all alive tensors are realized together.
//!    The tape detects boundary-crossing tensors: any tensor referenced inside the
//!    scope but whose inputs are not tracked by the tape was created outside — these
//!    are the dynamic inputs. Tensors fully internal to the tape are static and
//!    their compiled kernel bindings are cached.
//!
//! ## Input detection
//!
//! The tape maintains a set of all TensorIds created inside its scope. When a tensor
//! is pushed to the graph, its input TensorIds are checked against this set. Any input
//! not in the set is a boundary-crossing tensor — it was created before the tape and
//! its buffers change each iteration (e.g. model inputs, targets).
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
    Map, RT, Tensor, ZyxError,
    backend::BufferId,
    graph::{ClassId, ExecPlan, Graph},
    kernel::{DeviceId, Kernel, Op},
    runtime::{KernelData, ShapeId, TensorState},
    shape::Dim,
    slab::Slab,
    tensor::TensorId,
    view::View,
};

/// Non-differentiating tape scope.
///
/// Same boundary tracking as [`GradientTape`] but without autograd.
/// All alive tensors are realized when the tape is dropped.
/// The Merkle hash cache avoids recompilation on structurally identical iterations.
#[cfg_attr(feature = "py", pyo3::pyclass)]
pub struct Tape {}

impl Tape {
    /// Create gradient tape for automatic differentiation.
    /// Only one tape can exist at a time.
    ///
    /// Tensors created inside this scope are traced and realized on drop.
    /// Use this around inference loops to batch-realize outputs and
    /// enable graph caching across structurally identical iterations.
    pub fn new<'a>(params: impl IntoIterator<Item = &'a Tensor>) -> Result<Tape, ZyxError> {
        let mut rt = RT.lock();

        if rt.graph.is_some() {
            rt.graph.as_mut().unwrap().rc += 1;
        } else {
            let mut graph = Graph::new();
            graph.rc = 1;
            rt.graph = Some(graph);
        }

        for p in params {
            rt.promote_to_graph(p.id)?;
        }

        Ok(Tape {})
    }

    /// Create a tape scope without registering any tensors.
    /// Tensors are promoted to graph by calling add method.
    pub fn empty() -> Tape {
        Self::new(std::iter::empty()).unwrap()
    }

    /// Promote a tensor into the tape's graph scope.
    /// All ops on this tensor from now on will be tracked in the graph.
    pub fn add(&self, tensor: &Tensor) -> Result<(), ZyxError> {
        let mut rt = RT.lock();
        rt.promote_to_graph(tensor.id)?;
        Ok(())
    }

    /// Promote multiple tensors into the tape's graph scope at once.
    pub fn extend<'a>(&self, params: impl IntoIterator<Item = &'a Tensor>) -> Result<(), ZyxError> {
        let mut rt = RT.lock();
        for p in params {
            rt.promote_to_graph(p.id)?;
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
        let grads: Map<TensorId, TensorId> = rt.gradient(target.id(), sources.iter().copied().collect());
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

        let output_pairs: Vec<(TensorId, ClassId)> = tensors
            .into_iter()
            .map(|t| match rt.tensors[t.id].state {
                TensorState::Graph { class_id, .. } => (t.id, class_id),
                _ => unreachable!("non-graph tensor in realize"),
            })
            .collect();

        let output_tids: Vec<TensorId> = output_pairs.iter().map(|(tid, _)| *tid).collect();
        let output_classes: Vec<ClassId> = output_pairs.iter().map(|(_, cid)| *cid).collect();

        debug_assert!(rt.graph.is_some());

        let output_set: BTreeSet<ClassId> = output_classes.iter().copied().collect();
        let cache_key = rt.graph.as_ref().unwrap().cache_key(&output_set);
        if rt.plan_cache.contains_key(&cache_key) {
            return rt.execute_plan(cache_key, &output_tids, &output_classes);
        }

        rt.graph.as_ref().unwrap().debug_print(&rt.shapes);

        // TODO pattern match cublas, cblas, etc. kernels

        // Fills missing places with zyx custom kernels
        // SAFETY: graph and shapes are separate fields of Runtime, no aliasing, rust is stupid
        let shapes_ptr: *const Slab<ShapeId, Vec<Dim>> = &rt.shapes;
        rt.graph.as_mut().unwrap().fill_remaining(&output_set, unsafe { &*shapes_ptr });

        // Autotunes custom zyx kernels for all devices and adds kernel nodes for all of them
        rt.autotune_all_kernels()?;

        // After all kernels nodes are added, this adds movement ops so extract can pick fastest path
        rt.graph.as_mut().unwrap().add_memory_ops();

        rt.graph.as_ref().unwrap().debug_print(&rt.shapes);

        let nodes = rt.graph.as_ref().unwrap().extract(&output_set);

        let plan = ExecPlan::new(rt.graph.as_ref().unwrap(), &nodes, &output_set, &rt.devices, &rt.shapes);

        plan.debug();

        rt.plan_cache.insert(cache_key, plan);

        rt.execute_plan(cache_key, &output_tids, &output_classes)?;

        Ok(())
    }

    /// Materializes ALL graph tensors still alive in the tape scope.
    /// The tape is consumed — graph mode ends and every tracked tensor
    /// becomes realized (buffers allocated).
    pub fn realize_all(self) -> Result<(), ZyxError> {
        todo!()
    }
}

impl Drop for Tape {
    fn drop(&mut self) {
        let mut rt = RT.lock();
        let graph = &mut rt.graph;
        let Some(graph) = graph else { unreachable!() };
        graph.rc -= 1;
        if graph.rc > 0 {
            return;
        }
        rt.graph = None;

        let tids: Vec<TensorId> = rt.tensors.iter().map(|(id, _)| id).collect();
        for tid in tids {
            let rc = match &rt.tensors[tid].state {
                TensorState::Graph { rc, .. } => *rc,
                _ => continue,
            };
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
                rt.tensors[tid].state = TensorState::Eager { kernel_id, op_id, pending_store: false };
                for _ in 0..rc {
                    rt.kernels[kernel_id].outputs.push(tid);
                }
            }
        }
    }
}
