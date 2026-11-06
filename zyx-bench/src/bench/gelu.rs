// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! GELU over several shapes.

use zyx::{DType, Tensor, ZyxError};

pub fn bench() -> Result<(), ZyxError> {
    for dims in [&[4096i64][..], &[16384], &[1024, 4096], &[256, 1024, 512]] {
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        x.gelu();
    }
    Ok(())
}
