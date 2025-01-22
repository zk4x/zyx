use crate::{tensor::TensorId, Map, Set, Tensor, RT};

/// Gradient tape
/// 
/// Graph is always recorded, but when tensor is realized, it's graph is dropped.
/// When GradientTape is alive, graph is not dropped until GradientTape is dropped.
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

impl GradientTape {
    /// Create new gradient tape. Only one gradient tape can exist at a time.
    pub fn new() -> Self {
        let mut rt = RT.lock();
        if rt.graph.gradient_tape.is_some() {
            panic!("Only one gradient tape can exist at a time.");
        }
        rt.graph.gradient_tape = Some(Set::with_capacity_and_hasher(100, Default::default()));
        Self {}
    }

    /// Returns gradients of target derived w.r.t. sources
    #[must_use]
    pub fn gradient<'a>(
        &self,
        target: &Tensor,
        sources: impl IntoIterator<Item = &'a Tensor>,
    ) -> Vec<Option<Tensor>> {
        let sources: Vec<TensorId> = sources.into_iter().map(|t| t.id).collect();
        //println!("Sources: {sources:?}");
        let grads: Map<TensorId, TensorId> =
            RT.lock().backward(target.id(), &sources.iter().copied().collect());
        sources
            .into_iter()
            .map(|x: TensorId| grads.get(&x).copied())
            .map(|id: Option<TensorId>| id.map(|id| Tensor { id }))
            .collect()
    }
}

impl Drop for GradientTape {
    fn drop(&mut self) {
        RT.lock().drop_gradient_tape();
    }
}