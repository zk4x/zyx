use std::collections::HashMap;
use zyx::{DType, GradientTape, Module, Tensor, ZyxError};
use zyx_nn::{Conv2d, Linear, Module};
use zyx_optim::SGD;

/*#[derive(Module)]
struct MnistNet {
    l1: Conv2d,
    l2: Conv2d,
    l3: Linear,
}

impl MnistNet {
    fn new(dtype: DType) -> Result<Self, ZyxError> {
        Ok(Self {
            l1: Conv2d::new(1, 32, [3, 3], 1, 0, 1, 1, true, dtype)?,
            l2: Conv2d::new(32, 64, [3, 3], 1, 0, 1, 1, true, dtype)?,
            l3: Linear::new(1600, 10, true, dtype)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        //let x = x.reshape([0, 784]).unwrap();
        let x = self.l1.forward(x).unwrap().relu().max_pool2d([2, 2]).unwrap();
        let x = self.l2.forward(x).unwrap().relu().max_pool2d([2, 2]).unwrap();
        self.l3.forward(&x).unwrap()
    }
}*/

// With linear
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
    let train_x = train_dataset["train_x"].cast(DType::F32) / 255;
    let train_y = train_dataset["train_y"].clone();
    let test_x = train_dataset["test_x"].cast(DType::F32) / 255;
    let test_y = train_dataset["test_y"].clone();

    let batch_size = 128;
    let n_train = train_x.shape()[0] as u64;

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

    println!("train_x {:?}, train_y {:?}", train_x.shape(), train_y.shape());

    Tensor::realize_all()?;

    println!("Training...");
    for step in 0..7000usize {
        Tensor::set_training(true);
        let tape = GradientTape::new();
        let samples = Tensor::uniform(batch_size, 0..n_train)?;
        let x = train_x.gather(0, samples.reshape([batch_size, 1, 1])?.expand([batch_size, 28, 28])?)?;
        let y = train_y.gather(0, samples)?;

        let logits = net.forward(&x);
        let loss = logits.cross_entropy(y.one_hot(10), [-1])?.mean_all();
        let grads = tape.gradient(&loss, &net);
        optim.update(&mut net, grads);
        Tensor::realize(net.iter().chain(optim.iter()).chain([&loss]))?;

        if step.is_multiple_of(10) {
            Tensor::set_training(false);
            let acc = net.forward(&test_x).argmax_axis(1)?.equal(&test_y)?.mean_all().item::<f32>();
            println!("step {step}, loss {}, acc {:.2}%", loss.item::<f32>(), acc*100.)
        }
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
