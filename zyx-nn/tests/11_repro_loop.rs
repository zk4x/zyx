// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Second repro for `promote_to_graph` panic at `zyx/src/graph/mod.rs:1599`:
//!   "promote_to_graph: dead-graph tensor {tid} has no eager side to revert to:
//!    Graph { class_id, graph_id, shape_id, rc: 1 }"
//!
//! Independent of the `#[no_param]` fix: a plain `Linear` (no hyperparams) is
//! trained for 2 steps. Step 1 succeeds. Step 2's `Tape::new(&weight)` promotes
//! `weight`, which by now is a `TensorData::Graph` (not `Promoted`) from the
//! just-finished (dead) tape. The revert arm in `promote_to_graph` only
//! matches `Promoted` (which carries the eager `kernel_id`); a plain `Graph`
//! tensor has no eager side → panic.
//!
//! Expected: passes. Actual: panics on step 2.

use zyx::{DType, Tape, Tensor, ZyxError};
use zyx_optim::SGD;

#[test]
fn repro_promote_to_graph_dead_graph_no_eager_side() -> Result<(), ZyxError> {
    let mut weight = Tensor::randn([4, 4], DType::F32)?;
    let x = Tensor::randn([2, 4], DType::F32)?;
    let target = Tensor::randn([2, 4], DType::F32)?;
    let mut optim = SGD::default();

    for _step in 0..2 {
        let tape = Tape::new([&weight])?;
        let y = x.matmul(&weight)?;
        let loss = y.mse_loss(&target)?;
        let grads = tape.gradient(&loss, [&weight]);
        optim.update([&mut weight], grads);
        tape.realize([&weight, &loss])?;
    }
    Ok(())
}
