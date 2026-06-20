// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, GradientTape, ReduceOp, Tensor, ZyxError};
use zyx_nn::{BatchNorm, Conv2d, Linear};

fn make_bn(num_features: u64) -> BatchNorm {
    BatchNorm {
        eps: 1e-5, momentum: 0.1, track_running_stats: true,
        weight: Some(Tensor::ones(num_features, DType::F32)),
        bias: Some(Tensor::zeros(num_features, DType::F32)),
        running_mean: Tensor::zeros(num_features, DType::F32),
        running_var: Tensor::ones(num_features, DType::F32),
        num_batches_tracked: Tensor::zeros(1, DType::F32),
    }
}

#[test]
fn conv_bn_backward_1() -> Result<(), ZyxError> {
    // Reproduce slow backward kernel: conv(3->16) + BN + pool + linear + loss + backward
    let conv = Conv2d::new(3, 16, 3, 1, 1, 1, 1, false, DType::F32)?;
    let mut bn = make_bn(16);
    let linear = Linear::new(16, 10, true, DType::F32)?;

    let x = Tensor::randn([128, 3, 32, 32], DType::F32)?;
    let y = Tensor::randint([128], 0i32, 10)?;

    let tape = GradientTape::new();
    let h = conv.forward(&x)?;
    let h = bn.forward(h)?;
    let h = h.relu();
    let h = h.mean([2, 3])?;
    let h = h.reshape([0, 16])?;
    let logits = linear.forward(h)?;
    let loss = logits.cross_entropy(y, ReduceOp::Mean)?;

    let params = vec![&conv.weight, &linear.weight, linear.bias.as_ref().unwrap()];
    let _grads = tape.gradient(&loss, params.into_iter());
    Tensor::realize_all()?;
    Ok(())
}

#[test]
fn conv_weight_backward() -> Result<(), ZyxError> {
    // Conv weight backward kernel only: no BN, no pool, no linear.
    // This isolates the im2col + expand + mul + reduce pattern.
    let conv = Conv2d::new(3, 16, 3, 1, 1, 1, 1, false, DType::F32)?;
    let x = Tensor::rand([128, 3, 32, 32], DType::F32)?;

    let tape = GradientTape::new();
    let h = conv.forward(&x)?;
    let loss = h.sum_all();

    let _grads = tape.gradient(&loss, [&conv.weight].into_iter());
    Tensor::realize_all()?;
    Ok(())
}
