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

use crate::{
    Map, RT, Set, Tensor, ZyxError,
    graph::{ClassId, ExecPlan, Graph, Node},
    runtime::{Runtime, ShapeId, TensorState},
    shape::Dim,
    slab::Slab,
    tensor::TensorId,
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
    pub fn new() -> Result<Tape, ZyxError> {
        let mut rt = RT.lock();
        let mut graph = Graph::new();

        let tids: Vec<TensorId> = rt.tensors.iter().map(|(id, _)| id).collect();

        // Realize any eager tensors that don't have buffers yet.
        // Follow the pattern from Runtime::load: collect all outputs of
        // the tensor's kernel into a deduplicated Set, then add_store each.
        let mut seen: Set<TensorId> = Set::default();
        for tid in &tids {
            if rt.buffer_map.contains_key(tid) {
                continue;
            }
            if let TensorState::Eager { kernel_id, .. } = rt.tensors[*tid].state {
                seen.extend(rt.kernels[kernel_id].outputs.iter().copied());
            }
        }
        for tid in seen {
            rt.add_store(tid)?;
        }

        // Promote all existing tensors to graph Leaf nodes.
        for tid in tids {
            let shape_id = rt.tensors[tid].shape_id;
            let dtype = rt.tensors[tid].dtype;
            let (node_id, class_id) = graph.push(Node::Leaf { dtype, shape: shape_id }, shape_id, dtype);
            rt.tensors[tid].state = TensorState::Graph { node_id, class_id };
        }

        rt.graph = Some(graph);
        Ok(Tape {})
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

        // TODO hash graph to look up compiled plan from cache

        // TODO load compiled plan from cache if exists

        let output_classes: Vec<ClassId> = tensors
            .into_iter()
            .map(|t| match rt.tensors[t.id].state {
                TensorState::Graph { class_id, .. } => class_id,
                _ => unreachable!("non-graph tensor in realize"),
            })
            .collect();

        debug_assert!(rt.graph.is_some());

        // TODO pattern match cublas, cblas, etc. kernels

        // Fills missing places with zyx custom kernels
        // SAFETY: graph and shapes are separate fields of Runtime, no aliasing, rust is stupid
        let shapes_ptr: *const Slab<ShapeId, Vec<Dim>> = &rt.shapes;
        rt.graph.as_mut().unwrap().fill_remaining(&output_classes, unsafe { &*shapes_ptr });

        // Autotunes custom zyx kernels for all devices and adds kernel nodes for all of them
        rt.autotune_all_kernels()?;

        // After all kernels nodes are added, this adds movement ops so extract can pick fastest path
        rt.graph.as_mut().unwrap().add_memory_ops();

        rt.graph.as_ref().unwrap().debug_print(&rt.shapes);

        let nodes = rt.graph.as_ref().unwrap().extract(&output_classes);

        let plan = ExecPlan::new(rt.graph.as_ref().unwrap(), &nodes);

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
        // TODO realize all tensors that are outputs from graph and set graph to none

        RT.lock().graph = None;
    }
}

impl Runtime {
    fn gradient(&mut self, _target: TensorId, _sources: Set<TensorId>) -> Map<TensorId, TensorId> {
        todo!()
    }
}
