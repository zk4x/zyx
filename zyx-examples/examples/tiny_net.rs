use zyx_core::backend::Backend;
use zyx_derive::Module;
use zyx_nn::{Linear, prelude::*};
use std::vec;
use zyx_core::error::ZyxError;

#[derive(Module)]
struct TinyNet<B: Backend> {
    l0: Linear<B>,
    lr: f32,
    l1: Linear<B>,
}

fn main() -> Result<(), ZyxError> {
    let dev = zyx_opencl::device()?;

    let tiny_net = TinyNet {
        l0: dev.linear(128, 128),
        lr: 0.0,
        l1: dev.linear(128, 128),
    };

    for t in tiny_net.into_iter() {
        println!("{}", t.id());
    }

    let x = dev.uniform([2, 3], 0..2);

    let _grads = x.backward(&tiny_net);

    Ok(())
}
