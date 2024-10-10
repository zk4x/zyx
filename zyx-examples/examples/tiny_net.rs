use zyx::{DType, ZyxError};
use zyx_nn::{Module, Linear};

#[derive(Module)]
struct TinyNet {
    l0: Linear,
    l1: Linear,
    lr: f32,
}

fn main() -> Result<(), ZyxError> {
    let tiny_net = TinyNet {
        l0: Linear::init(1, 128, true, DType::F32)?,
        l1: Linear::init(1, 128, true, DType::F32)?,
        lr: 0.0,
    };

    for x in tiny_net.into_iter() {
        println!("{x}");
    }

    //let x = Tensor::uniform([2, 3], 0..2);

    //tiny_net.save("file.safetensors");
    //let tiny_net: TinyNet = Tensor::load("file.safetensors").collect();

    //let y = tiny_net.forward();
    //let loss = x.mse_loss(target);
    //let _grads = loss.backward(&tiny_net);

    Ok(())
}
