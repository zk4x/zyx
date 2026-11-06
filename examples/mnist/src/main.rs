// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use std::{collections::HashMap, time::Instant};
use zyx::{DType, Dev, Module, Reduction, Tape, Tensor, ZyxError};
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
        let x = x.reshape([-1, 784]).unwrap();
        let x = self.l1.forward(x).unwrap().relu();
        self.l2.forward(&x).unwrap()
    }
}

/// The device the dataset is moved to once, before the training loop. Moving
/// it inside the loop is exactly what the runtime warns about: every realize
/// would re-upload the whole dataset. `ZYX_MNIST_DEVICE` overrides the pick
/// (e.g. `cuda:0`, `c`, `vulkan:0`); by default the first available non-CPU
/// device is used.
fn compute_device() -> Dev {
    if let Ok(s) = std::env::var("ZYX_MNIST_DEVICE") {
        return s
            .parse()
            .unwrap_or_else(|e| panic!("ZYX_MNIST_DEVICE: {e}"));
    }
    Dev::all()
        .into_iter()
        .find(|d| !matches!(d, Dev::C))
        .unwrap_or(Dev::C)
}

fn main() -> Result<(), ZyxError> {
    println!("Loading MNIST...");
    let train_dataset: HashMap<String, Tensor> =
        Tensor::load(concat!(env!("CARGO_MANIFEST_DIR"), "/../data/mnist_dataset.safetensors"))?;
    let dev = compute_device();
    let train_x = train_dataset["train_x"]
        .clone()
        .reshape([60000, 784])?
        .to(dev)?;
    let train_y = train_dataset["train_y"].clone().to(dev)?;
    let _test_x = train_dataset["test_x"]
        .clone()
        .reshape([10000, 784])?
        .to(dev)?;
    let _test_y = train_dataset["test_y"].clone().to(dev)?;

    let batch_size = 128;
    let n_train = train_x.shape()[0].item::<i64>();

    let mut net = MnistNet::new(DType::F32)?;

    let mut optim = SGD {
        learning_rate: 0.01,
        momentum: 0.9,
        ..Default::default()
    };

    println!(
        "train_x {:?}, train_y {:?}",
        train_x.shape(),
        train_y.shape()
    );

    println!("Training...");
    Tensor::set_training(true);

    // Warm-up step (step 0). SGD creates its momentum buffers on first use
    // (`bias.push(grad)`); the frozen trace below must see the steady-state
    // update (`bias.assign`, `param.assign`), not the initialization branch,
    // so the buffers have to exist and be buffer-backed before tracing.
    let samples0 = Tensor::randint([batch_size], 0..n_train)?.to(dev)?;
    {
        let tape = Tape::new(&net)?;
        let x = train_x.index_select(0, &samples0)?;
        let y = train_y.index_select(0, &samples0)?;
        let logits = net.forward(&x);
        let loss = logits.cross_entropy(y, Reduction::Mean)?;
        let grads: Vec<_> = tape.gradient(&loss, &net);
        optim.update(&mut net, grads);
        tape.realize(net.iter().chain(optim.iter()).chain([&loss]))?;
    }

    // Trace one step and freeze it. `train_x`/`train_y` are promoted so
    // `index_select` is traced into the graph instead of running eagerly: with
    // both operands looking like plain leaves the data path stays an eager op
    // chain, which is where most of the per-step host time went. `samples` is
    // the replay input (its buffer is rebound every step); the dataset and the
    // optimizer state are captured once, and `net ∪ optim ∪ loss` as outputs
    // is what makes the plan keep the in-place `assign` stores.
    let tape = Tape::new(&net)?;
    tape.add(&train_x)?;
    tape.add(&train_y)?;
    // `freeze` only accepts tape leaves as declared inputs; promoting the
    // sample buffer puts its leaf class in the plan so replay can rebind it.
    tape.add(&samples0)?;
    let x = train_x.index_select(0, &samples0)?;
    let y = train_y.index_select(0, &samples0)?;
    let logits = net.forward(&x);
    let loss = logits.cross_entropy(y, Reduction::Mean)?;
    let grads: Vec<_> = tape.gradient(&loss, &net);
    optim.update(&mut net, grads);
    let outputs: Vec<Tensor> = net.iter().chain(optim.iter()).chain([&loss]).cloned().collect();
    let frozen = tape.freeze([&samples0], &outputs)?;

    let mut total_ms = 0.0f64;
    let mut count = 0u64;
    for step in 1..=200usize {
        let now = Instant::now();
        let samples = Tensor::randint([batch_size], 0..n_train)?.to(dev)?;
        let out = frozen.replay([&samples])?;
        let loss = out.last().expect("frozen tape returns the loss last");
        let elapsed_ms = now.elapsed().as_secs_f64() * 1000.0;
        if step >= 100 {
            total_ms += elapsed_ms;
            count += 1;
        }

        if step.is_multiple_of(5) {
            println!(
                "step {}, loss {:.6}, step_time {:.3}ms",
                step,
                loss.item::<f32>(),
                elapsed_ms
            );
        }
    }

    let avg_ms = total_ms / count as f64;
    println!("\nAverage step time (steps 100-7000): {:.3}ms", avg_ms);

    Ok(())
}
