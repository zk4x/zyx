// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use std::collections::HashMap;

use zyx::{DType, ReduceOp, Tape, Tensor, ZyxError};

#[test]
fn mnist() -> Result<(), ZyxError> {
    struct MnistNet {
        l1_weight: Tensor,
        l1_bias: Tensor,
        l2_weight: Tensor,
        l2_bias: Tensor,
    }

    impl MnistNet {
        fn forward(&self, x: &Tensor) -> Tensor {
            let x = x.reshape([0, 784]).unwrap();
            let x = x.matmul(&self.l1_weight.t()).unwrap() + &self.l1_bias;
            let x = x.relu();
            let x = x.matmul(&self.l2_weight.t()).unwrap() + &self.l2_bias;
            x
        }
    }

    let train_dataset: HashMap<String, Tensor> = Tensor::load("../zyx-examples/data/mnist_dataset.safetensors")?;
    let train_x = train_dataset["train_x"].cast(DType::F32) / 255;
    let train_y = train_dataset["train_y"].clone();

    let batch_size = 64;
    let num_train = train_x.shape()[0];

    let net = MnistNet {
        l1_weight: Tensor::randn([128, 784], DType::F32)?,
        l1_bias: Tensor::randn([128], DType::F32)?,
        l2_weight: Tensor::randn([10, 128], DType::F32)?,
        l2_bias: Tensor::randn([10], DType::F32)?,
    };

    for _ in 0..1 {
        for i in (0..num_train as u64).step_by(batch_size) {
            let end = if i + batch_size as u64 <= num_train as u64 {
                i + batch_size as u64
            } else {
                num_train as u64
            };

            let x = train_x.slice([i..end])?;
            let y = train_y.slice([i..end])?;

            let tape = Tape::new([&net.l1_weight, &net.l1_bias, &net.l2_weight, &net.l2_bias])?;
            let logits = net.forward(&x);
            let loss = logits.cross_entropy(y, ReduceOp::Mean)?;
            let grads = tape.gradient(&loss, [&net.l1_weight, &net.l1_bias, &net.l2_weight, &net.l2_bias, &loss]);

            // Simulate SGD update
            let lr = 0.01f32;
            let new_w1 = &net.l1_weight - &grads[0] * lr;
            let new_b1 = &net.l1_bias - &grads[1] * lr;
            let new_w2 = &net.l2_weight - &grads[2] * lr;
            let new_b2 = &net.l2_bias - &grads[3] * lr;

            tape.realize([&new_w1, &new_b1, &new_w2, &new_b2, &loss])?;
            break;
        }
    }
    Ok(())
}
