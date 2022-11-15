//! ## This is an example of linear neural network with sequential model

use zyx::prelude::*;
use zyx::accel::cpu;
use zyx::nn::{Linear, MSELoss, Mean, ReLU, Sigmoid, Tanh};
use zyx::optim;

fn main() {
    let mut network = (
        Linear::new(1, 20),
        ReLU,
        Linear::new(20, 50),
        Tanh,
        Linear::new(50, 1),
        Sigmoid,
    );

    // MSELoss does not reduce it's output (it's just (y-yp)^2), you need to add some reduce function if you want to apply reduce
    let mse_loss = (MSELoss, Mean { dims: () });

    let optimizer = optim::SGD::new().with_learning_rate(0.03);

    for _ in 0..100 {
        for i in 0..100 {
            <(Linear<_, _>, zyx::nn::ReLU, Linear<_, _>, zyx::nn::Tanh, Linear<_, _>, Sigmoid) as zyx::module::Module<'_, cpu::Buffer<f32>>>::parameters(&mut network).zero_grad();
            //network.parameters().zero_grad();

            let x = cpu::Buffer::cfrom([i as f32 / 10.]);
            let y = cpu::Buffer::cfrom([(i as f32 / 10.).sin()]);

            let y_predicted = network.forward(x);

            let loss = mse_loss.forward((y_predicted, y));

            loss.backward();

            use cpu::Buffer;
            <(Linear<Buffer<f32>, Buffer<f32>>, ReLU, Linear<Buffer<f32>, Buffer<f32>>, Tanh, Linear<Buffer<f32>, Buffer<f32>>, Sigmoid) as Module<'_, cpu::Buffer<f32>>>::parameters(&mut network).step(&optimizer);
            // Right now it looks bad, but eventually it will look like this:
            //network.parameters().step(&optimizer);
        }
    }
}
