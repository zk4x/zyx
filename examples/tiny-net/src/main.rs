// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, Module, Tape, Tensor, ZyxError};
use zyx_optim::SGD;

fn main() -> Result<(), ZyxError> {
    let mut optim = SGD {
        momentum: 0.2,
        nesterov: false,
        weight_decay: 0.0,
        ..Default::default()
    };

    let mut w = Tensor::rand([3, 2], DType::F16)?;
    let x = Tensor::from([2, 3, 1]).cast(DType::F16);
    let _target = Tensor::from([5, 7]).cast(DType::F16);

    for _ in 0..100 {
        let tape = Tape::new([&w])?;
        let y = x.matmul(&w)?.sigmoid();
        let grads = tape.gradient(&y, [&w]);
        optim.update([&mut w], grads);
        tape.realize([&w].into_iter().chain(optim.iter()))?;

        //
    }

    Ok(())
}
