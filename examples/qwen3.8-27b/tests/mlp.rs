// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! SwiGLU MLP reference-side test (runs on CUDA).
//!
//! Golden: `examples/data/qwen3_8b_mlp.safetensors` from `tests/mlp_ref.py`.
//! Run the dump first: `cd tests && python3.12 mlp_ref.py`.

use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};
use zyx_nn::Linear;

#[test]
fn mlp() -> Result<(), ZyxError> {
    let goldens = Tensor::load("../data/qwen3_8b_mlp.safetensors")?;
    let gate = Linear { weight: goldens["gate"].to(Dev::Cuda(0))?, bias: None };
    let up = Linear { weight: goldens["up"].to(Dev::Cuda(0))?, bias: None };
    let down = Linear { weight: goldens["down"].to(Dev::Cuda(0))?, bias: None };
    let input = goldens["input"].to(Dev::Cuda(0))?;
    let expected = goldens["output"].to_vec::<f32>()?;

    // SwiGLU: down(silu(gate(x)) * up(x)), silu(x) = x * sigmoid(x).
    let g = gate.forward(&input)?;
    let silu = &g * g.sigmoid();
    let out = down.forward(silu * up.forward(&input)?)?.to_vec::<f32>()?;

    assert_eq!(out.len(), expected.len());
    for (i, (&v, &e)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((v - e).abs() < 1e-4, "out[{i}] = {v}, expected {e}");
    }
    Ok(())
}
