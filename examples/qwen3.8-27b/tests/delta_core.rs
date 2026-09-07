// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `delta_core_kernel`: gated delta-rule core, token-recurrent.
//! in: mixed [M_PAD, CONV_DIM] f32 (q/k/v packed), b/a [M_PAD, DT_RANK] f32,
//!     exp_a_log/dt_bias [VH] f32. out: [VH, S, VD] f32.

use qwen3_8_27b::delta_core_kernel;
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

#[test]
fn delta_core() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_delta_core.safetensors")?;
    let mixed = goldens["mixed"].to(dev)?;
    let bp = goldens["bp"].to(dev)?;
    let ap = goldens["ap"].to(dev)?;
    let ealog = goldens["ealog"].to(dev)?;
    let dtb = goldens["dtb"].to(dev)?;
    let expected = &goldens["output"];
    let kk = delta_core_kernel();
    let (flops, read, write) = kk.flop_mem_rw();
    let ck = kk.compile()?;
    let t0 = std::time::Instant::now();
    let out = ck.forward(
        &[&mixed, &bp, &ap, &ealog, &dtb],
        vec![[48, 6, 128]],
    )?;
    out[0].sync()?;
    let total_us = t0.elapsed().as_micros() as f64;
    let tflops = if total_us > 0.0 { flops as f64 / total_us / 1e3 } else { 0.0 };
    let gbs = if total_us > 0.0 { (read + write) as f64 / total_us / 1e3 } else { 0.0 };
    eprintln!("delta_core forward+sync {total_us:.0}us, {tflops:.2} TFLOPS, {gbs:.1} GB/s");
    let v: Vec<f32> = out[0].to_vec()?;
    let exp: Vec<f32> = expected.to_vec()?;
    assert_eq!(v.len(), exp.len());
    let mut max_err = 0f32;
    for (i, (&a, &b)) in v.iter().zip(exp.iter()).enumerate() {
        max_err = max_err.max((a - b).abs());
    }
    eprintln!("delta_core max_err {max_err}");
    if max_err > 0.5 {
        for (i, (&a, &b)) in v.iter().zip(exp.iter()).enumerate() {
            if (a - b).abs() > 0.5 {
                eprintln!("mismatch at i={i} got={a} exp={b} diff={}", (a-b).abs());
                if i > 10 { break; }
            }
        }
    }
    // values can grow large (~650k) for random F32 inputs; check relative error
    let mut max_rel = 0f32;
    for (i, (&a, &b)) in v.iter().zip(exp.iter()).enumerate() {
        let denom = a.abs().max(b.abs()).max(1e-3);
        max_rel = max_rel.max((a - b).abs() / denom);
    }
    eprintln!("delta_core max_rel {max_rel}");
    assert!(max_rel < 1e-2, "delta_core max_rel {max_rel}");
    Ok(())
}
