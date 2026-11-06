// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! SiLU-like `x * sigmoid(x)` over several shapes.

use std::ops::{Add, Div, Mul, Sub};

use zyx::{DType, Tensor, ZyxError};

pub fn bench() -> Result<(), ZyxError> {
    for dims in [
        &[4096i64][..],
        &[16384],
        &[1024, 4096],
        &[512, 1024],
        &[256, 1024, 512],
        &[16, 768, 2048],
    ] {
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        x.sigmoid().mul(&x);
    }
    Ok(())
}
