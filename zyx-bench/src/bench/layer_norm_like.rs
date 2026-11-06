// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Layer-norm-like `(x - mean) / std * gamma + beta`.

use std::ops::{Add, Div, Mul, Sub};

use zyx::{DType, Tensor, ZyxError};

pub fn bench() -> Result<(), ZyxError> {
    for dims in [&[128i64, 768][..], &[64, 1024], &[32, 2048], &[16, 768, 1024]] {
        let last = *dims.last().unwrap();
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        let gamma = Tensor::rand([last], DType::F32)?;
        let beta = Tensor::rand([last], DType::F32)?;
        let mean = x.mean([-1])?.unsqueeze(-1)?;
        let centered = x.sub(&mean);
        let var = centered.var([-1])?.unsqueeze(-1)?;
        let std = var.add(Tensor::from(1e-5)).sqrt();
        let norm = centered.div(&std);
        norm.mul(&gamma).add(&beta);
    }
    Ok(())
}
