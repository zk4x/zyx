// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use std::{collections::HashMap, time::Instant};
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
    let train_x = train_dataset["train_x"].clone().reshape([60000, 784])?;
    let train_y = train_dataset["train_y"].clone();
    let _test_x = train_dataset["test_x"].clone().reshape([10000, 784])?;
    let _test_y = train_dataset["test_y"].clone();

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
    let mut total_ms = 0.0f64;
    let mut count = 0u64;
    for step in 0..7000usize {
        let now = Instant::now();
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

        let elapsed_ms = now.elapsed().as_secs_f64() * 1000.0;
        if step >= 100 {
            total_ms += elapsed_ms;
            count += 1;
        }

        if step.is_multiple_of(500) && step > 0 {
            println!("step {}, loss {:.6}, step_time {:.1}ms", step, loss.item::<f32>(), elapsed_ms);
        }
    }

    let avg_ms = total_ms / count as f64;
    println!("\nAverage step time (steps 100-7000): {:.1}ms", avg_ms);

    Ok(())
}
