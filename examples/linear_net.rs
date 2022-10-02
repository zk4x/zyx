//! ## This is an example of linear neural network with sequential model

use zyx::prelude::*;
use zyx::accel::cpu;
use zyx::nn;
use zyx::optim;

fn main() {
    let network = (
        nn::Linear::new::<f32>(1, 2000),
        nn::Tanh,
        nn::Linear::new::<f32>(2000, 1000),
        nn::Tanh,
        nn::Linear::new::<f32>(1000, 500),
        nn::Tanh,
        nn::Linear::new::<f32>(500, 1),
        nn::Tanh,
    );

    let mse_loss = |x, y| { x - y };
    let optimizer = optim::SGD::new(&network.parameters());

    for i in 0..100 {
        let x = cpu::Buffer::cfrom([[i as f32 / 10.]]);
        let y = cpu::Buffer::cfrom([[(i as f32).sin()]]);
        let y_predicted = network.forward(x);
        let loss = mse_loss(y_predicted, y);
        //println!("Loss: {}", loss);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }
}
