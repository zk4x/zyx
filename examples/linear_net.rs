//! ## This is an example of linear neural network with sequential model

fn main() {
    use zyx::prelude::*;
    use zyx::device::cpu::{self, Buffer};
    use zyx::nn::{Linear, MSELoss, Mean, ReLU, Sigmoid, Tanh};
    use zyx::optim;
    use zyx::shape::{Ax1, Sh2};

    let device = cpu::Device::default();

    let mut network = (
        Linear::<1, 20>::new(&device),
        ReLU,
        Linear::<20, 50>::new(&device),
        Tanh,
        Linear::<50, 1>::new(&device),
        Sigmoid,
    );

    // MSELoss does not reduce it's output (it's just (y-yp)^2), you need to add some reduce function if you want to apply reduce
    let mse_loss = (MSELoss, Mean { dims: Ax1::<0>::default() });

    let optimizer = optim::SGD::new().with_learning_rate(0.03);

    for _ in 0..100 {
        for i in 0..100 {
            <(
                Linear<'_, 1, 20, _, _>, ReLU,
                Linear<'_, 20, 50, _, _>, Tanh,
                Linear<'_, 50, 1, _, _>, Sigmoid
            ) as zyx::prelude::Module<'_, Buffer<'_, Sh2<1, 1>>>>::parameters(&mut network).zero_grad();
            //network.parameters().zero_grad(); // Above line will be inferred in future versions

            let x = device.buffer([[i as f32 / 10.]]);
            let y = device.buffer([[(i as f32 / 10.).sin()]]);

            let y_predicted = network.forward(x);

            let loss = mse_loss.forward((y_predicted, y));

            loss.backward();

            <(
                Linear<'_, 1, 20, _, _>, ReLU,
                Linear<'_, 20, 50, _, _>, Tanh,
                Linear<'_, 50, 1, _, _>, Sigmoid
            ) as zyx::prelude::Module<'_, Buffer<'_, Sh2<1, 1>>>>::parameters(&mut network).step(&optimizer);
            // Right now it looks bad, but eventually it will look like this:
            //network.parameters().step(&optimizer);
        }
    }
}
