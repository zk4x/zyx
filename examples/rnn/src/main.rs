// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! RNN training example demonstrating RNNCell usage with gradient descent.
//!
//! This example trains a simple RNN on random data to predict random targets.
//! The RNN forward pass processes a sequence of inputs, updating hidden state at each step.
//!
//! Performance notes:
//! - First few steps are slow due to kernel autotuning (~10s per step)
//! - After warmup, cached kernels run in ~30ms per step
//! - 20 kernels is typical for this workload (8 sequence steps × multiple ops)

use zyx::{DType, Tape, Tensor, ZyxError};
use zyx_nn::RNNCell;
use zyx_optim::SGD;

fn main() -> Result<(), ZyxError> {
    let input_size = 16;
    let hidden_size = 32;
    let batch_size = 64;
    let seq_len = 8;

    let train_x = Tensor::rand([batch_size, seq_len, input_size], DType::F32)?;
    let target = Tensor::rand([batch_size, hidden_size], DType::F32)?;

    let mut rnn = RNNCell::new(input_size, hidden_size, true, "tanh", Some(DType::F32))?;

    let mut optim = SGD {
        learning_rate: 0.05,
        momentum: 0.9,
        nesterov: true,
        ..Default::default()
    };

    println!("Training RNN...");
    for step in 0..50 {
        let tape = Tape::new(&rnn)?;

        let mut hidden = Tensor::zeros([batch_size, hidden_size], DType::F32);
        for t in 0..seq_len {
            let x_t = train_x.slice((.., t, ..))?;
            hidden = rnn.forward(&x_t, &hidden)?;
        }

        let loss = hidden.mse_loss(&target)?;
        let grads = tape.gradient(&loss, &rnn);
        optim.update(&mut rnn, grads);

        tape.realize(rnn.into_iter().chain(optim.into_iter()).chain([&loss]))?;

        println!("step {}, loss {}", step, loss.item::<f32>());
    }

    println!("RNN training completed!");
    Ok(())
}
