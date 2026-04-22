// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! RNN training example demonstrating RNNCell usage with gradient descent.
//!
//! This example trains a simple RNN on random data to predict random targets.
//! The RNN forward pass processes a sequence of inputs, updating hidden state at each step.
//!
//! Performance notes:
//! - First few steps are slow due to kernel autotuning (~10s per step)
//! - After warmup, cached kernels run in ~30ms per step
//! - 20 kernels is typical for this workload (8 sequence steps × multiple ops)

use zyx::{DType, GradientTape, Tensor, ZyxError};
use zyx_nn::RNNCell;
use zyx_optim::SGD;

fn main() -> Result<(), ZyxError> {
    let input_size = 16u64;
    let hidden_size = 32u64;
    let batch_size = 64usize;
    let seq_len = 8usize;

    let train_x = Tensor::rand([batch_size as u64, seq_len as u64, input_size], DType::F32)?;
    let target = Tensor::rand([batch_size as u64, hidden_size], DType::F32)?;

    let mut rnn = RNNCell::new(input_size, hidden_size, true, "tanh", Some(DType::F32))?;

    let mut optim = SGD {
        learning_rate: 0.05,
        momentum: 0.9,
        nesterov: true,
        ..Default::default()
    };

    Tensor::realize_all()?;

    println!("Training RNN...");
    for step in 0..50 {
        let tape = GradientTape::new();

        let mut hidden = Tensor::zeros([batch_size as u64, hidden_size], DType::F32);
        for t in 0..seq_len {
            let x_t = train_x.slice((.., t, ..))?;
            hidden = rnn.forward(&x_t, &hidden)?;
        }

        let loss = hidden.mse_loss(&target)?;
        let grads = tape.gradient(&loss, &rnn);
        optim.update(&mut rnn, grads);
        Tensor::realize_all()?;

        println!("step {}, loss {}", step, loss.item::<f32>());
    }

    println!("RNN training completed!");
    Ok(())
}