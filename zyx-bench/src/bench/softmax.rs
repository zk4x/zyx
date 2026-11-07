// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Softmax over one axis of several shapes.

use zyx::{DType, Tensor, ZyxError};

pub fn bench() -> Result<(), ZyxError> {
    // Temporary: `ZYX_SHAPE="8x512x2048:2"` runs one shape only, so a single
    // failing sequence can be reproduced without the other nine.
    let only = std::env::var("ZYX_SHAPE").ok();
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
        let label = format!(
            "{}:{}",
            dims.iter().map(ToString::to_string).collect::<Vec<_>>().join("x"),
            axes.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
        );
        if let Some(only) = &only {
            if only != &label {
                continue;
            }
        }
        println!("softmax {label}");
        let x = Tensor::rand(dims.iter().copied(), DType::F32)?;
        x.softmax(axes.iter().copied())?;
    }
    Ok(())
}
