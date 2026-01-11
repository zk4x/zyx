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
    //println!("{:?}", train_dataset.keys());
    //println!("{:.2}", train_x.slice((-5.., ..))?);
    /*let train_x = train_dataset["x_train"].cast(DType::F32) / 255;
    let train_y = train_dataset["y_train"].clone();
    let test_x = train_dataset["x_test"].cast(DType::F32) / 255;
    let test_y = train_dataset["y_test"].clone();*/
    let train_x = train_dataset["train_x"].cast(DType::F32) / 255;
    let train_y = train_dataset["train_y"].clone();
    let test_x = train_dataset["test_x"].cast(DType::F32) / 255;
    let test_y = train_dataset["test_y"].clone();

    let batch_size = 64;
    let num_train = train_x.shape()[0];

    let mut net = MnistNet::new(DType::F32)?;
    //net.save("models/mnist.safetensors")?;

    //let mut state_dict = Tensor::load("models/mnist.safetensors")?;
    //net.set_params(&mut state_dict);

    let mut optim = SGD {
        learning_rate: 0.01,
        momentum: 0.6,
        nesterov: false,
        ..Default::default()
    };

    println!("Training...");
    Tensor::realize_all()?;
    for epoch in 1..=5 {
        let mut total_loss = 0f32;
        let mut iters = 0;

        for i in (0..num_train).step_by(batch_size) {
            let end = (i + batch_size).min(num_train);

            let x = train_x.slice([i..end])?;
            let y = train_y.slice([i..end])?;

            let tape = GradientTape::new();
            let logits = net.forward(&x); //.clamp(-100, 100)?;
                                          //println!("{:?}, {:?}", logits.shape(), y.shape());

            //println!("{}", logits.slice((-5.., ..))?);
            let loss = logits.cross_entropy(y.one_hot(10), [-1])?.mean_all();

            let grads = tape.gradient(&loss, &net);

            /*for (i, grad) in grads.iter().enumerate() {
                println!("{i}, grad shape={:?}", grad.as_ref().unwrap().shape());
            }*/

            /*for (i, grad_opt) in grads.iter().enumerate() {
                if let Some(grad) = grad_opt {
                    // Compute the L2 norm of the gradient (vectorized)
                    let grad_norm = (grad * grad).sum_all().sqrt(); // ||grad||_2
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

            Tensor::realize(net.iter().chain(optim.iter()).chain([&loss]))?;
            total_loss += loss.item::<f32>();
            println!("Iters={iters}, loss={:.8}\n\n\n\n", loss.item::<f32>());

            iters += 1;
            //std::thread::sleep(std::time::Duration::from_secs(2));
            //panic!();
        }

        println!("Epoch {epoch}: loss = {total_loss:.4}");
    }
    // Required losses
    //let correct_losses = [2.302276134490967, 2.313948631286621, 2.2944066524505615, 2.3102803230285645 2.307297706604004 2.3003830909729004 2.299680471420288

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
