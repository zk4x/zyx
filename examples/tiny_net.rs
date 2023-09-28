#[cfg(feature = "opencl")]
use zyx::{
    context::Context,
    nn::{Linear, Module},
    optim::SGD,
    parameters::{Parameters, IntoParameters},
    tensor::Tensor,
    OutOfMemoryError,
};

#[cfg(feature = "opencl")]
struct TinyNet {
    l0: Linear,
    l1: Linear,
}

#[cfg(feature = "opencl")]
impl Module for TinyNet {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.l0.forward(x).tanh();
        self.l1.forward(&x)
    }

    fn parameters(&mut self) -> Parameters {
        self.l0.parameters().join(self.l1.parameters())
    }
}

#[cfg(feature = "opencl")]
fn main() -> Result<(), OutOfMemoryError> {
    //let mut ctx = Context::new();
    let mut ctx = Context::opencl().unwrap();
    let mut tiny_net = TinyNet {
        l0: ctx.linear(1024, 1024),
        l1: ctx.linear(1024, 1024),
    };
    let mut opt = SGD::new();
    let mut x = ctx.randn((1024, 1024)); //.set_label("x");
    let label = ctx.randn(1024);

    x.realize()?;
    tiny_net.realize()?;
    for _ in 0..5 {
        let out = tiny_net.forward(&x);
        let loss = out.mse(&label);
        loss.backward(tiny_net.parameters());
        opt.step(tiny_net.parameters())?;
    }

    let now = std::time::Instant::now();
    for _ in 0..100 {
        let out = tiny_net.forward(&x);
        let loss = out.mse(&label);
        loss.backward(&mut tiny_net);
        opt.step(&mut tiny_net)?;
    }
    for param in tiny_net.parameters() {
        let _ = param.to_vec();
    }
    let elapsed = now.elapsed();

    std::println!("Elapsed {}ms", elapsed.as_millis());
    Ok(())
}

#[cfg(not(feature = "opencl"))]
fn main() {}

