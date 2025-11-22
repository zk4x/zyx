use std::fs::File;
use std::io::{BufReader, Read};
use zyx::{DType, GradientTape, Tensor, ZyxError};
use zyx_nn::{Linear, Module};
use zyx_optim::SGD;

// ----------------------------------------------------------------------------
// MNIST LOADING (as before, no changes needed here)
// ----------------------------------------------------------------------------

fn load_idx_images(path: &str) -> Result<Tensor, ZyxError> {
    let mut f = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    let mut dims = [0u8; 12];
    f.read_exact(&mut dims)?;

    let num = u32::from_be_bytes([dims[0], dims[1], dims[2], dims[3]]) as usize;
    let rows = u32::from_be_bytes([dims[4], dims[5], dims[6], dims[7]]) as usize;
    let cols = u32::from_be_bytes([dims[8], dims[9], dims[10], dims[11]]) as usize;

    let mut data = vec![0u8; num * rows * cols];
    f.read_exact(&mut data)?;

    // Flatten to [num, 784] and normalize to [0,1]
    let data_f32: Vec<f32> = data.iter().map(|x| *x as f32 / 255.0).collect();

    Tensor::from(data_f32).reshape([num, rows * cols])
}

fn load_idx_labels(path: &str) -> Result<Vec<i64>, ZyxError> {
    let mut f = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    let mut dims = [0u8; 4];
    f.read_exact(&mut dims)?;

    let num = u32::from_be_bytes(dims) as usize;

    let mut data = vec![0u8; num];
    f.read_exact(&mut data)?;

    let data_i64: Vec<i64> = data.into_iter().map(|x| x as i64).collect();

    Ok(data_i64)
}

// ----------------------------------------------------------------------------
// MODEL
// ----------------------------------------------------------------------------

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
        let x = self.l1.forward(x).unwrap().relu();
        self.l2.forward(&x).unwrap()
    }
}

// ----------------------------------------------------------------------------
// TRAINING LOOP
// ----------------------------------------------------------------------------

fn main() -> Result<(), ZyxError> {
    println!("Loading MNIST...");
    let train_x = load_idx_images("data/mnist/train-images-idx3-ubyte")?;
    let train_y = load_idx_labels("data/mnist/train-labels-idx1-ubyte")?;
    let test_x = load_idx_images("data/mnist/t10k-images-idx3-ubyte")?;
    let test_y = load_idx_labels("data/mnist/t10k-labels-idx1-ubyte")?;

    let batch_size = 64usize;
    let num_train = train_x.shape()[0];

    // Convert labels to one-hot encoding
    let train_y_one_hot: Vec<Vec<f32>> = train_y
        .iter()
        .map(|&label| {
            let mut one_hot = vec![0.0; 10];
            one_hot[label as usize] = 1.0;
            one_hot
        })
        .collect();

    let test_y_one_hot: Vec<Vec<f32>> = test_y
        .iter()
        .map(|&label| {
            let mut one_hot = vec![0.0; 10];
            one_hot[label as usize] = 1.0;
            one_hot
        })
        .collect();

    // Convert to Tensor
    let train_y_one_hot = Tensor::from(train_y_one_hot);
    let test_y_one_hot = Tensor::from(test_y_one_hot);

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
            let y = train_y_one_hot.slice([i..end]).unwrap();

            let tape = GradientTape::new();
            let logits = net.forward(&x);
            let loss = logits.cross_entropy(&y, [-1])?;
            total_loss += loss.item();

            let grads = tape.gradient(&loss, &net);
            optim.update(&mut net, grads);

            // Realize updated parameters for safety
            Tensor::realize(&net)?;
        }

        println!("Epoch {epoch}: loss = {total_loss:.4}");
    }

    // Evaluation Loop
    println!("Evaluating...");
    let logits = net.forward(&test_x);
    let preds: Vec<i64> = logits.argmax(-1).unwrap().try_into()?;

    let correct = preds
        .iter()
        .zip(test_y.iter())
        .filter(|(a, b)| a == b)
        .count();

    let accuracy = (correct as f32) / (test_y.len() as f32) * 100.0;
    println!("Test accuracy: {:.2}%", accuracy);

    Ok(())
}
