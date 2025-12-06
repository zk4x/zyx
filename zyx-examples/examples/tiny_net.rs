use zyx::{DType, GradientTape, Tensor, ZyxError};
use zyx_optim::SGD;

fn main() -> Result<(), ZyxError> {
    let mut optim = SGD {
        momentum: 0.9,
        nesterov: true,
        ..Default::default()
    };

    let mut w = Tensor::rand([3, 2], DType::F16)?;
    let x = Tensor::from([2, 3, 1]).cast(DType::F16);
    let target = Tensor::from([5, 7]).cast(DType::F16);

    Tensor::realize([&w, &x, &target])?;
    for _ in 0..100 {
        let tape = GradientTape::new();
        let y = x.matmul(&w)?;
        let mut grads = tape.gradient(&y, [&w]);
        w = w - grads.pop().unwrap().unwrap();
        drop(tape);
        Tensor::realize([&w])?;
    }

    Ok(())
}
