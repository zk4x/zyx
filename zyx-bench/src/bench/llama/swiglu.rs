// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! llama SwiGLU block: `down(silu(gate(x)) * up(x))`.

use std::ops::Mul;

use zyx::{DType, Tensor, ZyxError};

use super::shapes::{HIDDEN, INTERMEDIATE, SIZES};
use crate::bench::{bench_dev, run};

pub fn bench() -> Result<(), ZyxError> {
    let dev = bench_dev();
    for seq in SIZES {
        let x = Tensor::rand([seq, HIDDEN], DType::F16)?.to(dev)?;
        let gate = Tensor::rand([INTERMEDIATE, HIDDEN], DType::F16)?.to(dev)?;
        let up = Tensor::rand([INTERMEDIATE, HIDDEN], DType::F16)?.to(dev)?;
        let down = Tensor::rand([HIDDEN, INTERMEDIATE], DType::F16)?.to(dev)?;
        println!("llama swiglu [1x{seq}x{HIDDEN}]");
        let g = x.matmul(gate.t())?;
        let u = x.matmul(up.t())?;
        run(g.swish().mul(&u).matmul(down.t())?)?;
    }
    Ok(())
}
