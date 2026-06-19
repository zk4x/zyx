// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use std::time::Instant;
use zyx::{DType, Tensor, ZyxError};
use zyx_nn::Conv2d;

#[test]
fn conv_bn_mean() -> Result<(), ZyxError> {
    let conv = Conv2d::new(3, 32, 3, 1, 1, 1, 1, false, DType::F32)?;
    let x = Tensor::rand([128, 3, 32, 32], DType::F32)?;
    let z = conv.forward(x)?;
    let batch_mean = z.mean([0, 2, 3])?;
    let start = Instant::now();
    Tensor::realize([&batch_mean])?;
    let elapsed = start.elapsed();
    eprintln!("conv_bn_mean: {}.{:03}s",
        elapsed.as_secs(),
        elapsed.subsec_millis(),
    );
    Ok(())
}
