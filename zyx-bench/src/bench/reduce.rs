// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Whole-tensor reductions.

use zyx::{DType, Tensor, ZyxError};

pub fn bench() -> Result<(), ZyxError> {
    for dims in [&[4096i64][..], &[16384], &[65536], &[4096, 4096], &[256, 1024, 512], &[64, 64, 64]] {
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        x.sum_all();
        x.mean_all();
        x.max_all();
        x.min_all();
        x.prod_all();
        x.var_all();
        x.std_all();
    }
    Ok(())
}
