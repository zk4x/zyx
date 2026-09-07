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
fn mlp_timing() -> Result<(), ZyxError> {
    use qwen3_8_27b::mlp_kernel;
    use std::time::Instant;
    let t0 = Instant::now();
    let k = mlp_kernel(8, 128);
    let builder_us = t0.elapsed().as_micros();
    eprintln!("[mlp_timing] builder mlp_kernel(8,128) {}us", builder_us);
    let t1 = Instant::now();
    let ck = k.compile()?;
    let compile_us = t1.elapsed().as_micros();
    eprintln!("[mlp_timing] compile mlp_kernel(8,128) {}us", compile_us);
    eprintln!("[mlp_timing] builder+compile {}us vs budget 50000us", builder_us + compile_us);
    // forward timing isolated (dummy tensors)
    let dev = Dev::Cuda(0);
    let g = Tensor::randn([8, 128], zyx::DType::F32)?.to(dev)?;
    let u = Tensor::randn([8, 128], zyx::DType::F32)?.to(dev)?;
    // warmup
    let _ = ck.forward(&[&g, &u], vec![[8, 128]])?;
    let t2 = Instant::now();
    let mid = ck.forward(&[&g, &u], vec![[8, 128]])?.remove(0);
    let _ = mid.to_vec::<f32>()?;
    let forward_us = t2.elapsed().as_micros();
    eprintln!("[mlp_timing] forward mlp_kernel(8,128) {}us budget 50us/1000us", forward_us);
    // real Qwen shape 6,17408
    let t3 = Instant::now();
    let k2 = mlp_kernel(6, 17408);
    let builder2_us = t3.elapsed().as_micros();
    eprintln!("[mlp_timing] builder mlp_kernel(6,17408) {}us", builder2_us);
    let t4 = Instant::now();
    let ck2 = k2.compile()?;
    let compile2_us = t4.elapsed().as_micros();
    eprintln!("[mlp_timing] compile mlp_kernel(6,17408) {}us total {}us", compile2_us, builder2_us + compile2_us);
    let g2 = Tensor::randn([6, 17408], zyx::DType::F32)?.to(dev)?;
    let u2 = Tensor::randn([6, 17408], zyx::DType::F32)?.to(dev)?;
    let _ = ck2.forward(&[&g2, &u2], vec![[6, 17408]])?;
    let t5 = Instant::now();
    let mid2 = ck2.forward(&[&g2, &u2], vec![[6, 17408]])?.remove(0);
    let _ = mid2.to_vec::<f32>()?;
    let forward2_us = t5.elapsed().as_micros();
    eprintln!("[mlp_timing] forward mlp_kernel(6,17408) {}us", forward2_us);
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
