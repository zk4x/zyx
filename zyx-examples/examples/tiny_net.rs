use zyx_core::backend::Backend;
use zyx_derive::Module;
use zyx_nn::{Linear, prelude::*};
use std::vec;
use zyx_core::error::ZyxError;

#[derive(Module)]
struct TinyNet<B: Backend> {
    l0: Linear<B>,
    lr: f32,
}

fn main() -> Result<(), ZyxError> {
    let dev = zyx_opencl::device()?;

    let tiny_net = TinyNet {
        l0: dev.linear(128, 128),
        lr: 0.0,
    };

    for t in tiny_net.into_iter() {
        println!("{}", t.id());
    }

    Ok(())
}