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

use crate::{Map, RT, Set, Tensor, tensor::TensorId};

/// Gradient tape for automatic differentiation.
///
/// Graph is always recorded, but when tensor is realized, it's graph is dropped.
/// When [`GradientTape`] is alive, graph is not dropped until [`GradientTape`] is dropped.
///
/// Unlike other deep learning frameworks, there is no need to specify which tensors
/// are differentiable nor is there need to specify multiple gradient tapes to calculate
/// higher order derivatives. In zyx as long as gradient tape is alive, derivatives
/// of all operations then occured since it's creation can be calculated.
///
/// Gradient tape is necessary because without it graph would grow
/// indefinitely with each iteration of training/inference loop.
/// By creating gradient tape in the beginning of each training loop and dropping
/// it at the end, the user ensures that graph of tensors is dropped after each
/// iteration of the training loop.
///
/// Since tensors are realized lazily, intermediate tensors needed for backpropagation
/// are not held in memory.
#[cfg_attr(feature = "py", pyo3::pyclass)]
pub struct GradientTape {}

/// Non-differentiating tape scope.
///
/// Same boundary tracking as [`GradientTape`] but without autograd.
/// All alive tensors are realized when the tape is dropped.
/// The Merkle hash cache avoids recompilation on structurally identical iterations.
#[cfg_attr(feature = "py", pyo3::pyclass)]
pub struct Tape {}

impl Tape {
    /// Create gradient tape for automatic differentiation.
    /// Only one gradient tape can exist at a time.
    pub fn autograd() -> GradientTape {
        let mut rt = RT.lock();
        rt.graph.tape_rc += 1;
        if rt.graph.tape.is_some() {
            return GradientTape {};
        }
        rt.graph.tape = Some(Set::with_capacity_and_hasher(100, Default::default()));
        drop(rt);
        GradientTape {}
    }

    /// Create non-differentiating tape scope.
    ///
    /// Tensors created inside this scope are traced and realized on drop.
    /// Use this around inference loops to batch-realize outputs and
    /// enable graph caching across structurally identical iterations.
    pub fn no_grad() -> Tape {
        let mut rt = RT.lock();
        rt.graph.tape_rc += 1;
        if rt.graph.tape.is_some() {
            return Tape {};
        }
        rt.graph.tape = Some(Set::with_capacity_and_hasher(100, Default::default()));
        drop(rt);
        Tape {}
    }
}

impl GradientTape {
    /// Returns gradients of target derived w.r.t. sources
    /// Any ops following this function will not be traced.
    /// If you want to keep tracing, use [`gradient_persistent`](GradientTape::gradient_persistent)
    #[must_use]
    pub fn gradient<'a>(self, target: &Tensor, sources: impl IntoIterator<Item = &'a Tensor>) -> Vec<Option<Tensor>> {
        self.gradient_persistent(target, sources)
        // here tape is dropped automatically
    }

    /// Returns gradients of target derived w.r.t. sources
    /// This persistent version keeps gradient tape alive and new ops will be traced until [`GradientTape`] gets destroyed.
    /// This function is useful for higher order derivatives or derivating multiple tensors.
    #[must_use]
    pub fn gradient_persistent<'a>(&self, target: &Tensor, sources: impl IntoIterator<Item = &'a Tensor>) -> Vec<Option<Tensor>> {
        let sources: Vec<TensorId> = sources.into_iter().map(Tensor::id).collect();
        //println!("Sources: {sources:?}");
        let grads: Map<TensorId, TensorId> = RT.lock().gradient(target.id(), &sources.iter().copied().collect());
        sources
            .into_iter()
            .map(|x: TensorId| grads.get(&x).copied())
            .map(|id: Option<TensorId>| id.map(|id| Tensor { id }))
            .collect()
    }
}

impl Drop for Tape {
    fn drop(&mut self) {
        //RT.lock().drop_gradient_tape();
        if let Ok(mut rt) = RT.try_lock() {
            rt.drop_gradient_tape();
        } else {
            println!("Warning: Unable to drop GradientTape due to runtime mutex lock.");
        }
    }
}

impl Drop for GradientTape {
    fn drop(&mut self) {
        //RT.lock().drop_gradient_tape();
        if let Ok(mut rt) = RT.try_lock() {
            rt.drop_gradient_tape();
        } else {
            println!("Warning: Unable to drop GradientTape due to runtime mutex lock.");
        }
    }
}
