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
        Tensor::load_safetensors("data/mnist_dataset.safetensors")?;
    println!("{:?}", train_dataset.keys());
    let train_x = train_dataset.remove("x_train").unwrap().cast(DType::F32)/255;
    println!("{:.2}", train_x.slice((-5.., ..))?);
    let train_y = train_dataset.remove("y_train").unwrap();
    let test_x = train_dataset.remove("x_test").unwrap().cast(DType::F32)/255;
    let test_y = train_dataset.remove("y_test").unwrap();

    let batch_size = 64usize;
    let num_train = train_x.shape()[0];

    let mut net = MnistNet::new(DType::F32)?;

    let mut optim = SGD {
        learning_rate: 0.0001,
        momentum: 0.9,
        nesterov: true,
        ..Default::default()
    };

    println!("Training...");
    for epoch in 1..=5 {
        let mut total_loss = 0f32;
        let mut iters = 0;

        for i in (0..num_train).step_by(batch_size) {
            let end = (i + batch_size).min(num_train);

            let x = train_x.slice([i..end]).unwrap();
            let y = train_y.slice([i..end]).unwrap();

            let tape = GradientTape::new();
            let logits = net.forward(&x).clamp(-100, 100)?;
            println!("{:?}, {:?}", logits.shape(), y.shape());

            //println!("{}", logits.slice((-5.., ..))?);
            let loss = logits.cross_entropy(y.one_hot(10), [-1])?.mean_all();
            total_loss += loss.item::<f32>();
            println!("Loss is: {:.8}", loss.item::<f32>());

            let grads = tape.gradient(&loss, &net);

            /*for (i, grad_opt) in grads.iter().enumerate() {
                if let Some(grad) = grad_opt {
                    // Compute the L2 norm of the gradient (vectorized)
                    let grad_norm = (grad * grad).sum_all().sqrt();  // ||grad||_2
                    println!("Grad {} L2 norm: {:?}", i, grad_norm);

                    // Compute min/max in a vectorized way
                    let grad_min = grad.min_all();
                    let grad_max = grad.max_all();
                    println!("Grad {} min/max: {}/{}", i, grad_min, grad_max);
                } else {
                    println!("Grad {} is None", i);
                }
            }*/

            optim.update(&mut net, grads);

            Tensor::realize(net.into_iter().chain(&optim))?;
            iters += 1;
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
