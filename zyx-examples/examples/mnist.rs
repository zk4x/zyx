// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use std::collections::HashMap;
use zyx::{DType, GradientTape, Module, Tensor, ZyxError};
use zyx_nn::{Linear, Module};
use zyx_optim::SGD;

#[derive(Module)]
struct MnistNet {
    l1: Linear,
    l2: Linear,
}

impl MnistNet {
    fn new(dtype: DType) -> Result<Self, ZyxError> {
        Ok(Self {
            l1: Linear::new(784, 128, true, dtype)?,
            l2: Linear::new(128, 10, true, dtype)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = x.reshape([0, 784]).unwrap();
        let x = self.l1.forward(x).unwrap().relu();
        self.l2.forward(&x).unwrap()
    }
}

fn main() -> Result<(), ZyxError> {
    println!("Loading MNIST...");
    let train_dataset: HashMap<String, Tensor> = Tensor::load("data/mnist_dataset.safetensors")?;
    let train_x = train_dataset["train_images"].clone().reshape([60000, 784])?;
    let train_y = train_dataset["train_labels"].clone();
    let test_x = train_dataset["test_images"].clone().reshape([10000, 784])?;
    let test_y = train_dataset["test_labels"].clone();

    Tensor::realize_all()?;

    let x_mean = train_x.mean_all().item::<f32>();
    let x_max = train_x.max_all().item::<f32>();
    let x_min = train_x.min_all().item::<f32>();
    println!("train_x mean={:.6}, max={:.6}, min={:.6}", x_mean, x_max, x_min);

    let batch_size = 128;
    let n_train = train_x.shape()[0] as u64;

    let mut net = MnistNet::new(DType::F32)?;

    let mut optim = SGD {
        learning_rate: 0.01,
        momentum: 0.9,
        ..Default::default()
    };

    println!("train_x {:?}, train_y {:?}", train_x.shape(), train_y.shape());

    Tensor::realize_all()?;

    println!("Training...");
    for step in 0..200usize {
        Tensor::set_training(true);
        let tape = GradientTape::new();
        let samples = Tensor::uniform(batch_size, 0..n_train)?;
        let x = train_x.index_select(0, &samples)?;
        let y = train_y.index_select(0, &samples)?;

        let logits = net.forward(&x);
        let loss = logits.cross_entropy(y.one_hot(10), [-1])?.mean_all();
        let grads: Vec<_> = tape.gradient(&loss, &net);

        optim.update(&mut net, grads);
        Tensor::realize(net.iter().chain(optim.iter()).chain([&loss]))?;

        if step % 20 == 0 {
            Tensor::set_training(false);
            let acc = net
                .forward(&test_x)
                .argmax_axis(1)?
                .equal(&test_y)?
                .cast(DType::F32)
                .mean_all()
                .item::<f32>();
            println!(
                "step {step}, loss {:.6}, acc {:.2}%",
                loss.item::<f32>(),
                acc * 100.
            );
        }
    }

    Ok(())
}
