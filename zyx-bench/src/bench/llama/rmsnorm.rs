// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! llama RMSNorm: `x * rsqrt(mean(x^2) + eps) * w`.

use std::ops::{Add, Div, Mul};

use zyx::{DType, Tensor, ZyxError};

use super::shapes::{EPS, HIDDEN, SIZES};
use crate::bench::{bench_dev, run};

pub fn bench() -> Result<(), ZyxError> {
    let dev = bench_dev();
    for seq in SIZES {
        let x = Tensor::rand([1, seq, HIDDEN], DType::F16)?.to(dev)?;
        let w = Tensor::rand([HIDDEN], DType::F16)?.to(dev)?;
        println!("llama rmsnorm [1x{seq}x{HIDDEN}]");
        let mean_sq = x.clone().mul(&x).mean([-1])?.unsqueeze(-1)?;
        let norm = x.div(&mean_sq.add(Tensor::from(EPS)).rsqrt());
        run(norm.mul(&w))?;
    }
    Ok(())
}
