// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Log-softmax over one axis of several shapes.

use zyx::{DType, Tensor, ZyxError};

pub fn bench() -> Result<(), ZyxError> {
    for &(dims, axes) in &[
        (&[4096i64][..], &[0i32][..]),
        (&[256, 4096], &[0i32][..]),
        (&[512, 1024], &[1i32][..]),
        (&[32, 768, 1024], &[1i32][..]),
        (&[64, 512], &[0i32][..]),
    ] {
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        x.ln_softmax(axes.iter().copied())?;
    }
    Ok(())
}
