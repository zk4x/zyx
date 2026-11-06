// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Reductions over several axes at once.

use zyx::{DType, Tensor, ZyxError};

pub fn bench() -> Result<(), ZyxError> {
    for dims in [&[64i64, 64, 64][..], &[32, 128, 256], &[16, 256, 128], &[8, 512, 64]] {
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        x.sum([0, 1])?;
        x.max([0, 1])?;
        x.mean([1, 2])?;
        x.var([0, 2])?;
    }
    Ok(())
}
