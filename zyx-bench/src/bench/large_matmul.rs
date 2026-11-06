// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Large-matmul sweep: square, tall, wide and deep, one launch per shape.

use zyx::{DType, Tensor, ZyxError};

use super::force;

pub fn bench() -> Result<(), ZyxError> {
    let dims: &[u64] = &[
        256, 320, 384, 448, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1792, 2048, 2304, 2560, 3072, 3584, 4096,
        4608, 5120, 6144,
    ];
    // Square.
    for &n in dims {
        let a = Tensor::rand([n, n], DType::F32)?;
        let b = Tensor::rand([n, n], DType::F32)?;
        force(a.matmul(b.t())?)?;
    }
    // Vary M (tall matrices).
    for &m in dims.iter().filter(|&&m| m != 1024) {
        let a = Tensor::rand([m, 1024], DType::F32)?;
        let b = Tensor::rand([1024, 1024], DType::F32)?;
        force(a.matmul(b.t())?)?;
    }
    // Vary N (wide matrices).
    for &n in dims.iter().filter(|&&n| n != 1024) {
        let a = Tensor::rand([1024, 1024], DType::F32)?;
        let b = Tensor::rand([n, 1024], DType::F32)?;
        force(a.matmul(b.t())?)?;
    }
    // Vary K (inner dimension).
    for &k in dims.iter().filter(|&&k| k != 1024) {
        let a = Tensor::rand([1024, k], DType::F32)?;
        let b = Tensor::rand([1024, k], DType::F32)?;
        force(a.matmul(b.t())?)?;
    }
    Ok(())
}
