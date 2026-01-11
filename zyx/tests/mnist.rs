use std::collections::HashMap;

use zyx::{DType, GradientTape, Tensor, ZyxError};

#[test]
fn mnist() -> Result<(), ZyxError> {
    struct MnistNet {
        l1_weight: Tensor,
        l1_bias: Tensor,
        l2_weight: Tensor,
        l2_bias: Tensor,
    }

    impl MnistNet {
        fn forward(&self, x: &Tensor) -> Tensor {
            let x = x.reshape([0, 784]).unwrap();
            //println!("x={}", x.reshape([0, 28, 28]).unwrap().slice((0, 15..20, 15..20)).unwrap());
            //println!("{}", self.l1_weight.slice((0, 0..10)).unwrap());
            let x = x.matmul(&self.l1_weight.t()).unwrap() + &self.l1_bias;
            let x = x.relu();
            let x = x.matmul(&self.l2_weight.t()).unwrap() + &self.l2_bias;
            x
        }
    }

    let state_dict = Tensor::load("../zyx-examples/models/mnist.safetensors")?;

    let net = MnistNet {
        l1_weight: state_dict["l1.weight"].clone(),
        l1_bias: state_dict["l1.bias"].clone(),
        l2_weight: state_dict["l2.weight"].clone(),
        l2_bias: state_dict["l2.bias"].clone(),
    };

    let train_dataset: HashMap<String, Tensor> = Tensor::load("../zyx-examples/data/mnist_dataset.safetensors")?;
    let train_x = train_dataset["train_x"].cast(DType::F32) / 255;
    let train_y = train_dataset["train_y"].clone();
    let test_x = train_dataset["test_x"].cast(DType::F32) / 255;
    let test_y = train_dataset["test_y"].clone();

    let batch_size = 64;
    let num_train = train_x.shape()[0];

    //let mut optim = SGD { learning_rate: 0.01, momentum: 0.6, nesterov: false, ..Default::default() };

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
            println!("{:.4}", logits.slice((0..4, 0..4)).unwrap());

            //println!("{}", logits.slice((-5.., ..))?);
            let loss = logits.cross_entropy(y.one_hot(10), [-1])?.mean_all();
            println!("{loss}");

            let grads = tape.gradient(&loss, [&net.l1_weight, &net.l1_bias, &net.l2_weight, &net.l2_bias, &loss]);

            //println!("{}", grads[1].clone().unwrap());
            println!("{}", grads[3].clone().unwrap());
            println!("{}", grads[4].clone().unwrap());
            panic!();

            /*for (i, grad) in grads.iter().enumerate() {
                println!("{i}, grad shape={:?}", grad.as_ref().unwrap().shape());
            }*/

            /*optim.update(&mut net, grads);

            Tensor::realize(net.iter().chain(optim.iter()).chain([&loss]))?;
            total_loss += loss.item::<f32>();
            println!("Iters={iters}, loss={:.8}\n\n\n\n", loss.item::<f32>());*/

            iters += 1;
            //std::thread::sleep(std::time::Duration::from_secs(2));
            panic!();
        }
        let correct_losses = [
            2.302830696105957,
            2.3023550510406494,
            2.307710647583008,
            2.2971067428588867,
            2.308199405670166,
            2.3134093284606934,
            2.302781820297241,
            2.2912802696228027,
            2.302485704421997,
            2.301023483276367,
        ];

        println!("Epoch {epoch}: loss = {total_loss:.4}");
    }
    panic!();
    Ok(())
}

/*

          x=tensor([[ 0.1464, -0.0082, -0.2147, -0.1245,  0.0447,  0.1138,  0.0383,  0.0569,
                   -0.0029,  0.0660]], grad_fn=<AddBackward0>)
          */
