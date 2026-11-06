// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Elementwise activations over several shapes.

use std::ops::{Add, Div, Mul, Sub};

use zyx::{DType, Tensor, ZyxError};

pub fn bench() -> Result<(), ZyxError> {
    for dims in [
        &[16384i64][..],
        &[32768i64],
        &[65536i64],
        &[131072i64],
        &[524288i64],
        &[4096i64, 4096i64],
        &[512i64, 2048i64],
        &[128i64, 16384i64],
    ] {
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        x.relu();
        x.sigmoid();
        x.tanh();
        x.exp();
        x.log(Tensor::from(2.0));
    }
    Ok(())
}
