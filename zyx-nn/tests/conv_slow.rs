// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, Tensor, ZyxError};
use zyx_nn::Conv2d;

#[test]
fn conv_bn_mean() -> Result<(), ZyxError> {
    let conv = Conv2d::new(3, 32, [3], [1], [1], [1], 1, false, DType::F32)?;
    let x = Tensor::rand([128, 3, 32, 32], DType::F32)?;
    let z = conv.forward(x)?;
    let _batch_mean = z.mean([0, 2, 3])?;
    Ok(())
}
