use std::collections::HashMap;
use zyx::{DType, GradientTape, Tensor, ZyxError};
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

// ----------------------------------------------------------------------------
// TRAINING LOOP
// ----------------------------------------------------------------------------

fn main() -> Result<(), ZyxError> {
    println!("Loading MNIST...");
    let mut train_dataset: HashMap<String, Tensor> =
        Tensor::load_safetensors("../data/mnist.safetensors")?;
    let train_x = train_dataset.remove("train_x").unwrap();
    let train_y = train_dataset.remove("train_y").unwrap();
    let test_x = train_dataset.remove("test_x").unwrap();
    let test_y = train_dataset.remove("test_y").unwrap();

    let batch_size = 64usize;
    let num_train = train_x.shape()[0];

    let mut net = MnistNet::new(DType::F32)?;
    let mut optim = SGD {
        learning_rate: 0.01,
        momentum: 0.9,
        nesterov: true,
        ..Default::default()
    };

    println!("Training...");
    for epoch in 1..=5 {
        let mut total_loss = 0f32;

        for i in (0..num_train).step_by(batch_size) {
            let end = (i + batch_size).min(num_train);

            let x = train_x.slice([i..end]).unwrap();
            let y = train_y.slice([i..end]).unwrap();
            println!("{y}");

            let tape = GradientTape::new();
            let logits = net.forward(&x);
            let loss = logits.cross_entropy(&y, [-1])?;
            total_loss += loss.item::<f32>();

            let grads = tape.gradient(&loss, &net);
            optim.update(&mut net, grads);

            Tensor::realize(&net)?;
        }

        println!("Epoch {epoch}: loss = {total_loss:.4}");
    }

    // Evaluation Loop
    println!("Evaluating...");
    let logits = net.forward(&test_x);
    let preds: Vec<i64> = logits.argmax_axis(-1)?.try_into()?;

    /*let correct = preds
        .iter()
        .zip(test_y.iter())
        .filter(|(a, b)| a == b)
        .count();

    let accuracy = (correct as f32) / (test_y.len() as f32) * 100.0;
    println!("Test accuracy: {:.2}%", accuracy);*/

    Ok(())
}
