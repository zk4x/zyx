// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Matmul over a spread of shapes.

use zyx::{DType, Tensor, ZyxError};

pub fn bench() -> Result<(), ZyxError> {
    for &(m, n, k) in &[
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (128, 128, 128),
        (128, 256, 64),
        (256, 128, 512),
        (512, 256, 128),
        (1024, 512, 256),
        (256, 1024, 512),
        (512, 1024, 128),
        (128, 512, 1024),
        (2048, 2048, 2048),
    ] {
        let a = Tensor::rand([m, k], DType::F32)?;
        let b = Tensor::rand([n, k], DType::F32)?;
        a.matmul(b.t())?;
    }
    Ok(())
}
