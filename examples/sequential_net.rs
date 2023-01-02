//! ## This is an example of linear neural network with sequential model

fn main() {
    use zyx::device::cpu;
    use zyx::nn::{Linear, MSELoss, Mean, ReLU, Sigmoid};
    use zyx::optim;
    use zyx::prelude::*;
    use zyx::shape::Ax1;

    let device = cpu::Device::default();

    let mut network = (
        Linear::<1, 200>::new(&device),
        ReLU,
        Linear::<200, 2000>::new(&device),
        ReLU,
        Linear::<2000, 500>::new(&device),
        ReLU,
        Linear::<500, 1>::new(&device),
        Sigmoid,
    );

    // MSELoss does not reduce it's output (it's just (y-yp)^2), you need to add some reduce function if you want to apply reduce
    let mse_loss = (MSELoss, Mean::<Ax1<0>>::new());

    let optimizer = optim::SGD::new().with_learning_rate(0.03);

    for _ in 0..10 {
        for i in 0..100 {
            network.parameters().zero_grad();

            let x = device.buffer([[i as f32 / 10.]]);
            let y = device.buffer([[(i as f32 / 10.).sin()]]);

            let y_predicted = network.forward(x);

            let loss = mse_loss.forward((y_predicted, y));

            loss.backward();

            network.parameters().step(&optimizer);
        }
    }
}
