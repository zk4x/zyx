// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! llama embedding lookup: `[1, seq]` indices into the vocab matrix.

use zyx::{DType, Tensor, ZyxError};

use super::shapes::{HIDDEN, SIZES, VOCAB};
use crate::bench::{bench_dev, run};

pub fn bench() -> Result<(), ZyxError> {
    let dev = bench_dev();
    let w = Tensor::rand([VOCAB, HIDDEN], DType::F16)?.to(dev)?;
    for seq in SIZES {
        let idx = Tensor::from(vec![0u32; seq as usize]).unsqueeze(0)?.to(dev)?;
        println!("llama embedding [1x{seq}] of [{VOCAB}x{HIDDEN}]");
        run(w.index_select(0, idx)?)?;
    }
    Ok(())
}
