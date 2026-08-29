// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Repro for `promote_to_graph` panic in `zyx/src/graph/mod.rs:1609`:
//!   "promote_to_graph: tensor {tid} has no eager kernel: Constant { value: 64, rc: 1 }"
//!
//! Root cause: `zyx-derive`'s `Module` derive treats every `Tensor` field as a
//! trainable parameter. When a Module has a `Tensor` field that is a hyperparam
//! (built once at construction via `i64.into()` and therefore a `Constant`
//! tensor with no eager kernel), `Tape::new(&model)` promotes that constant
//! into the graph via `promote_to_graph`, which requires an Eager tensor.
//!
//! This test distills the failure into a minimal `Module` with one
//! hyperparam-style `Tensor` field plus one real Eager parameter. It panics
//! the same way `examples/readme-test` does (which uses `TransformerBlock`
//! whose `MultiheadAttention.embed_dim: Tensor` is the constant 64).
//!
//! Expected: passes. Actual: panics in `promote_to_graph`.

use zyx::{DType, Tape, Tensor, ZyxError};
use zyx_derive::Module;

#[derive(Module)]
struct TinyModel {
    /// Real trainable parameter (Eager tensor).
    weight: Tensor,
    /// Hyperparam stored as a Tensor (a Constant — no eager kernel).
    /// The Module derive currently yields this as a "parameter".
    #[no_param]
    hyperparam: Tensor,
}

#[test]
fn repro_promote_to_graph_hyperparam_tensor() -> Result<(), ZyxError> {
    let model = TinyModel {
        weight: Tensor::randn([4, 4], DType::F32)?,
        hyperparam: 64i64.into(),
    };

    // `Tape::new` promotes every Tensor yielded by the Module's `iter()`,
    // including the `hyperparam` Constant — which has no eager kernel.
    let _tape = Tape::new(&model)?;
    Ok(())
}
