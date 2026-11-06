// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Embedding lookup (`index_select` on the vocab axis).

use zyx::{DType, Tensor, ZyxError};

pub fn bench() -> Result<(), ZyxError> {
    for &(vocab_size, embed_dim, seq_len) in &[(10000, 768, 64), (50000, 1024, 256), (100000, 768, 128)] {
        let embedding = Tensor::rand([vocab_size, embed_dim], DType::F32)?;
        let idx: Vec<i32> = (0..seq_len).map(|i| (i % vocab_size)).collect();
        let indices = Tensor::from(idx);
        embedding.index_select(0, indices)?;
    }
    Ok(())
}
