use zyx::{DType, Tensor};
use zyx_derive::Module;
use zyx_nn::Linear;

#[derive(Module)]
struct TinyNet {
    l0: Linear,
    lr: f32,
    l1: Linear,
}

fn main() {
    let tiny_net = TinyNet {
        l0: Linear::new(128, 128, DType::F32),
        lr: 0.0,
        l1: Linear::new(128, 128, DType::F32),
    };

    let x = Tensor::uniform([2, 3], 0..2);

    let _grads = x.backward(&tiny_net);
}
