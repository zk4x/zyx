use zyx::{DType, GradientTape, Tensor, ZyxError};
use zyx_optim::SGD;

fn main() -> Result<(), ZyxError> {
    let mut optim = SGD {
        momentum: 0.2,
        nesterov: false,
        weight_decay: 0.0,
        ..Default::default()
    };

    let mut w = Tensor::rand([3, 2], DType::F16)?;
    let x = Tensor::from([2, 3, 1]).cast(DType::F16);
    let target = Tensor::from([5, 7]).cast(DType::F16);

    Tensor::realize([&w, &x, &target])?;
    for _ in 0..100 {
        let tape = GradientTape::new();
        let y = x.matmul(&w)?;
        let grads = tape.gradient(&y, [&w]);
        optim.update([&mut w], grads);
        Tensor::realize([&w])?;
        Tensor::realize(optim.bias.iter())?;
        //Tensor::realize_all()?;
    }

    Ok(())
}
