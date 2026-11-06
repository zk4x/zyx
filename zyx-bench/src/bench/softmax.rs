// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Softmax over one axis of several shapes.

use zyx::{DType, Tensor, ZyxError};

pub fn bench() -> Result<(), ZyxError> {
    for &(dims, axes) in &[
        (&[4096i64][..], &[0i32][..]),
        (&[128, 4096], &[0i32][..]),
        (&[512, 1024], &[1i32][..]),
        (&[16, 768, 1024], &[1i32][..]),
        (&[64, 64, 64, 64], &[3i32][..]),
        (&[64, 128, 4096], &[2i32][..]),
        (&[16384i64], &[0i32][..]),
        (&[256, 2048], &[1i32][..]),
        (&[32, 4096], &[0i32][..]),
        (&[8, 512, 2048], &[2i32][..]),
    ] {
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        x.softmax(axes.iter().copied())?;
    }
    Ok(())
}
