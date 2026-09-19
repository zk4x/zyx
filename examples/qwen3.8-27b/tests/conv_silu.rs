// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `conv_silu_kernel`: depthwise causal conv1d (kernel 4, left pad 3) + SiLU
//! out [M_PAD, CONV_DIM] = silu( sum_k convw[c, k] * inp[max(0, t+k-3), c] )

use qwen3_8_27b::{conv_silu_kernel, CONV_DIM, M_PAD};
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

#[test]
fn conv_silu() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_conv_silu.safetensors")?;
    let input = goldens["input"].to(dev)?;
    let convw = goldens["convw"].to(dev)?;
    let expected = &goldens["output"];
    let kk = conv_silu_kernel();
    let k = kk.compile()?;
    let out = k.forward(&[&input, &convw], vec![[M_PAD, CONV_DIM]])?;
    let v: Vec<f32> = out[0].to_vec()?;
    let exp: Vec<f32> = expected.to_vec()?;
    assert_eq!(v.len(), exp.len());
    let mut max_err = 0f32;
    for (i, (&a, &b)) in v.iter().zip(exp.iter()).enumerate() {
        max_err = max_err.max((a - b).abs());
        assert!(
            (a - b).abs() < 1e-3,
            "conv_silu[{i}] = {a}, expected {b}, max_err {max_err}"
        );
    }
    eprintln!("conv_silu max_err {max_err}");
    Ok(())
}
