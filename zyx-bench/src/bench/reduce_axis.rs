// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Axis reductions over many shapes.

use zyx::{DType, Tensor, ZyxError};

pub fn bench() -> Result<(), ZyxError> {
    for dims in [
        &[512i64, 512][..],
        &[256, 1024],
        &[128, 2048],
        &[64, 4096],
        &[32, 128, 256],
        &[16, 512, 512],
        &[8, 1024, 1024],
        &[1024, 256],
        &[4096, 128],
        &[64, 16384],
        &[32, 32768],
        &[16, 65536],
        &[8, 128, 512],
        &[4, 256, 256, 256],
        &[1024, 1024],
        &[2048, 64],
        &[128, 128, 256],
        &[256, 128, 128],
        &[16, 256, 256],
        &[8, 64, 64, 64],
    ] {
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        let last = dims.len() as i32 - 1;
        x.sum([last])?;
        x.max([last])?;
        x.min([last])?;
        x.mean([last])?;
        x.var([last])?;
        // Reduce along axis 0 as well.
        if dims.len() > 1 {
            x.sum([0])?;
            x.max([0])?;
            x.mean([0])?;
        }
    }
    Ok(())
}
