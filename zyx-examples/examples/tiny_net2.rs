use zyx::{DType, GradientTape, Tensor, ZyxError};
use zyx_nn::{Linear, Module};
use zyx_optim::SGD;

#[derive(Module)]
struct TinyNet {
    l0: Linear,
    l1: Linear,
    lr: f32,
}

impl TinyNet {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.l0.forward(x).unwrap().relu();
        self.l1.forward(x).unwrap().sigmoid()
    }
}

fn main() -> Result<(), ZyxError> {
    let mut net = TinyNet {
        l0: Linear::new(3, 1024, true, DType::F16)?,
        l1: Linear::new(1024, 2, true, DType::F16)?,
        lr: 0.01,
    };

    let mut optim = SGD {
        learning_rate: net.lr,
        momentum: 0.9,
        nesterov: true,
        ..Default::default()
    };

    let x = Tensor::from([2, 3, 1]).cast(DType::F16);
    let target = Tensor::from([5, 7]).cast(DType::F16);

    Tensor::realize(&net)?;
    Tensor::realize([&x, &target])?;
    for _ in 0..100 {
        let tape = GradientTape::new();
        let y = net.forward(&x);
        let loss = y.mse_loss(&target)?;
        let grads = tape.gradient(&loss, &net);
        optim.update(&mut net, grads);
        drop(tape);
        Tensor::realize(&net)?;
    }

    Ok(())
}
