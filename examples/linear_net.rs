//! ## This is an example of linear neural network with sequential model

use zyx::prelude::*;
use zyx::tensor::Tensor;
use zyx::nn;
use zyx::optim;

fn main() {
    let network = (
        nn::Linear::new::<f32>(1, 20),
        nn::Tanh,
        nn::Linear::new::<f32>(20, 10),
        nn::Tanh,
        nn::Linear::new::<f32>(10, 5),
        nn::Tanh,
        nn::Linear::new::<f32>(5, 1),
        nn::Tanh,
    );

    let mse_loss = |x, y| { x - y };
    let optimizer = optim::SGD::new(&network.parameters());

    for i in 0..100000 {
        let x = Tensor::from([i as f32 / 10.]);
        let y = Tensor::from([[(i as f32).sin()]]);
        let y_predicted = network.forward(x);
        let loss = mse_loss(y_predicted, y);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }
}
