//! ## This is an example of linear neural network with sequential model

use zyx::prelude::*;
use zyx::accel::cpu;
use zyx::nn::{Linear, MSELoss, Mean, ReLU, Sigmoid, Tanh};
use zyx::optim;

fn main() {
    let network = (
        Linear::new::<f32>(1, 20),
        ReLU,
        Linear::new::<f32>(20, 50),
        Tanh,
        Linear::new::<f32>(50, 1),
        Sigmoid,
    );

    // MSELoss does not reduce it's output (it's just (y-yp)^2), you need to add some reduce function if you want to apply reduce
    let mse_loss = (MSELoss, Mean { dims: () });

    // This looks bad right now, eventually it will look like this:
    //let optimizer = optim::SGD::new(network.parameters()).with_learning_rate(0.03);
    let optimizer = optim::SGD::new(<&(
        Linear<cpu::Buffer<f32>, _, cpu::Buffer<f32>, _>, ReLU,
        Linear<cpu::Buffer<f32>, _, cpu::Buffer<f32>, _>, Tanh,
        Linear<cpu::Buffer<f32>, _, cpu::Buffer<f32>, _>, Sigmoid) as Module<cpu::Buffer<f32>>>::parameters(&network)).with_learning_rate(0.03);

    for _ in 0..100 {
        for i in 0..100 {
            let x = cpu::Buffer::cfrom([i as f32 / 10.]);
            let y = cpu::Buffer::cfrom([(i as f32 / 10.).sin()]);

            let y_predicted = network.forward(x);

            let loss = (y_predicted, y).apply(&mse_loss);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
    }
}
