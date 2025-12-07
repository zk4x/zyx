use zyx::{DType, GradientTape, Tensor, ZyxError};
use zyx_nn::{Linear, Module};
use zyx_optim::SGD;

#[derive(Module)]
struct TinyNet {
    l0: Linear,
    l1: Linear,
}

impl TinyNet {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.l0.forward(x).unwrap().relu();
        self.l1.forward(x).unwrap()
    }
}

fn main() -> Result<(), ZyxError> {
    let mut net = TinyNet {
        l0: Linear::new(3, 1024, true, DType::F16)?,
        l1: Linear::new(1024, 2, true, DType::F16)?,
    };

    let mut optim = SGD {
        learning_rate: 0.01,
        momentum: 0.9,
        nesterov: true,
        ..Default::default()
    };

    let x = Tensor::from([2, 3, 1]).cast(DType::F16);
    let target = Tensor::from([5, 7]).cast(DType::F16);

    for _ in 0..100 {
        let tape = GradientTape::new();
        let y = net.forward(&x);
        let loss = y.mse_loss(&target)?;
        let grads = tape.gradient(&loss, &net);
        optim.update(&mut net, grads);
        Tensor::realize(net.into_iter().chain(optim.into_iter()))?;
        //Tensor::realize_all()?;
    }

    Ok(())
}
