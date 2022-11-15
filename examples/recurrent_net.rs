//! ## This is an example of recurrent neural network
#![no_std]

extern crate alloc;

fn main() {
    use zyx::prelude::*;
    use zyx::accel::cpu;
    use zyx::nn::{RNNCell, Linear, SoftMax, MSELoss, Sum, ReLU};
    use zyx::optim;

    let hidden_size = 10;
    let input_size = 3;

    // This looks bad right now, eventually types will be elided
    let mut rnn_net = (
        RNNCell::new(input_size, hidden_size),
        ReLU,
    );
    let mut net2 = (
        Linear::new(hidden_size, 3),
        SoftMax { dims: () },
    );


    let mut hidden_state = cpu::Buffer::uniform((1, hidden_size), 0., 1.);

    // MSELoss does not reduce it's output, you need to add some reduce function if you want to apply reduce
    // Sum dims () means sum across all dims
    let mse_loss = (MSELoss, Sum { dims: () });

    let optimizer = optim::SGD::new();

    for i in 0..30000 {
        use cpu::Buffer;
        (<(RNNCell<Buffer<f32>, Buffer<f32>, Buffer<f32>, Buffer<f32>>, zyx::nn::ReLU) as zyx::module::Module<'_, (Buffer<f32>, Buffer<f32>)>>::parameters(&mut rnn_net), <(Linear<Buffer<f32>, Buffer<f32>>, SoftMax<()>) as zyx::module::Module<'_, Buffer<f32>>>::parameters(&mut net2)).zero_grad();
        // This looks bad right now, eventually types will be elided:
        //(rnn_net.parameters(), net2.parameters()).zero_grad();

        let i_f32 = i as f32;
        let data = alloc::vec![i_f32*1., i_f32*2., i_f32*3.];

        let y = cpu::Buffer::from_vec(data.iter().map(|x| (*x as f32).sin()).collect(), (1, input_size));
        let x = cpu::Buffer::from_vec(data, (1, input_size));

        let hidden_state_t1 = rnn_net.forward((x, hidden_state.clone()));

        // Don't forget to get data on your hidden state to get just Buffer without graph
        hidden_state = hidden_state_t1.data().clone();

        let y_predicted = net2.forward(hidden_state_t1);
        let loss = mse_loss.forward((y_predicted, y));

        loss.backward();

        (<(RNNCell<_, _, _, _>, zyx::nn::ReLU) as zyx::module::Module<'_, (Buffer<f32>, Buffer<f32>)>>::parameters(&mut rnn_net), <(Linear<_, _>, SoftMax<()>) as zyx::module::Module<'_, Buffer<f32>>>::parameters(&mut net2)).step(&optimizer);
        // This looks bad right now, eventually types will be elided:
        //(rnn_net.parameters(), net2.parameters()).step(&optimizer);
    }

    //println!("hidden state: {}", hidden_state);
}
