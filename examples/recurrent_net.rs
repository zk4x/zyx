//! ## This is an example of recurrent neural network
//#![no_std]

//extern crate alloc;

fn main() {
    use zyx::device::cpu;
    use zyx::nn;
    use zyx::optim;
    use zyx::prelude::*;
    use zyx::shape::{Ax2, Sh2};

    let device = cpu::Device::default();

    const HIDDEN: usize = 10;
    const INPUT: usize = 3;
    const OUT: usize = 5;

    // This looks bad right now, eventually types will be elided
    let mut rnn_net = (nn::RNNCell::<INPUT, HIDDEN>::new(&device), nn::ReLU);
    let mut net2 = (nn::Linear::<HIDDEN, OUT>::new(&device), nn::Sigmoid);

    let mut hidden_state: cpu::Buffer<'_, Sh2<1, HIDDEN>> = device.uniform(0., 1.);

    // MSELoss does not reduce it's output, you need to add some reduce function if you want to apply reduce
    // Sum dims () means sum across all dims
    let mse_loss = (nn::MSELoss, nn::Sum::<Ax2<0, 1>>::new());

    let optimizer = optim::SGD::new();

    for i in 0..30000 {
        (rnn_net.parameters(), net2.parameters()).zero_grad();

        let i_f32 = i as f32;
        let data = vec![i_f32 * 1., i_f32 * 2., i_f32 * 3.];

        let y: cpu::Buffer<'_, Sh2<1, INPUT>> =
            device.slice(&data.iter().map(|x| (*x as f32).sin()).collect::<Vec<_>>());
        let x: cpu::Buffer<'_, Sh2<1, INPUT>> = device.slice(&data);

        let hidden_state_t1 = rnn_net.forward((x, hidden_state.clone()));

        // Don't forget to get data on your hidden state to get just Buffer without graph
        hidden_state = hidden_state_t1.data().clone();

        let y_predicted = net2.forward(hidden_state_t1);
        let loss = mse_loss.forward((y_predicted, y));

        loss.backward();

        (rnn_net.parameters(), net2.parameters()).step(&optimizer);
    }

    //println!("hidden state: {}", hidden_state);
}
