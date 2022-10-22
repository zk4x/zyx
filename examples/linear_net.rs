//! ## This is an example of linear neural network with sequential model

/*use zyx::prelude::*;
use zyx::accel::cpu;
use zyx::nn::{Linear, Tanh, SoftMax, Sigmoid, MSELoss, Sum};
use zyx::optim;

fn main() {
    let network = (
        Linear::new::<f32>(1, 2000),
        Tanh,
        Linear::new::<f32>(2000, 1000),
        Tanh,
        Linear::new::<f32>(1000, 500),
        Tanh,
        Linear::new::<f32>(500, 1),
        Sigmoid,
    );

    // MSELoss does not reduce it's output, you need to add some reduce function if you want to apply reduce
    let mse_loss = (MSELoss, Sum { dims: () });

    // This looks bad right now, eventually it will look like this:
    //let optimizer = optim::SGD::new(network.parameters());
    let optimizer = optim::SGD::new(<&(Linear<_>, zyx::nn::Tanh, Linear<_>, zyx::nn::Tanh, Linear<_>, zyx::nn::Tanh, Linear<_>, SoftMax<()>) as Module<cpu::Buffer<f32>>>::parameters(&network));

    for i in 0..100 {
        let x = cpu::Buffer::<f32>::cfrom(i as f32 / 10.);
        let y = cpu::Buffer::cfrom((i as f32).sin());
        let y_predicted = network.forward(x);
        let loss = (y_predicted, y).apply(mse_loss);
        //println!("Loss: {}", loss);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}*/

fn main() {}
