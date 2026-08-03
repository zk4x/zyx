// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, Tensor, ZyxError, bf16};

#[test]
fn dbg_mean() -> Result<(), ZyxError> {
    let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Tensor::from(data).cast(DType::BF16);
    let mean = x.mean([0])?;
    let _: f32 = mean.item()?;
    Ok(())
}