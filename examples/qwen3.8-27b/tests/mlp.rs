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
    let gate = Linear {
        weight: goldens["gate"].to(Dev::Cuda(0))?,
        bias: None,
    };
    let up = Linear {
        weight: goldens["up"].to(Dev::Cuda(0))?,
        bias: None,
    };
    let down = Linear {
        weight: goldens["down"].to(Dev::Cuda(0))?,
        bias: None,
    };
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

#[test]
fn mlp_kernel_cuda() -> Result<(), ZyxError> {
    use qwen3_8_27b::mlp_kernel;
    let goldens = Tensor::load("../data/qwen3_8b_mlp.safetensors")?;
    let dev = Dev::Cuda(0);
    let gate_w = goldens["gate"].to(dev)?;
    let up_w = goldens["up"].to(dev)?;
    let input = goldens["input"].to(dev)?;
    let expected = goldens["output"].to_vec::<f32>()?;
    let gate = input.clone().matmul(gate_w.transpose(0, 1)?)?;
    let up = input.clone().matmul(up_w.transpose(0, 1)?)?;
    let g = gate.reshape([8, 128])?.contiguous()?;
    let u = up.reshape([8, 128])?.contiguous()?;
    let mk = mlp_kernel(8, 128).compile()?;
    let mid = mk.forward(&[&g, &u], vec![[8, 128]])?.remove(0);
    let mid = mid.reshape([2, 4, 128])?;
    let down_w = goldens["down"].to(dev)?;
    let out = mid.matmul(down_w.transpose(0, 1)?)?.to_vec::<f32>()?;
    assert_eq!(out.len(), expected.len());
    let mut bad = 0;
    for (i, (&v, &e)) in out.iter().zip(expected.iter()).enumerate() {
        if (v - e).abs() >= 1e-3 {
            if bad < 10 { println!("mlp[{i}] {v} vs {e}"); }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "mlp {bad} mismatches");
    Ok(())
}
