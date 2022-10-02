//! ## This is an example of recurrent neural network

use zyx::prelude::*;
use zyx::accel::cpu;
use zyx::nn;
use zyx::optim;

fn main() {
    let hidden_size = 10;
    let input_size = 3;

    let rnn_net = (
        nn::RNNCell::new::<f32>(input_size, hidden_size),
        nn::Tanh,
    );
    let net2 = (
        nn::Linear::new::<f32>(hidden_size, 3),
        nn::Tanh,
    );

    let mut params = Vec::new();
    params.extend(rnn_net.parameters().into_iter());
    params.extend(net2.parameters().into_iter());

    let mut hidden_state = cpu::Buffer::uniform(&[1, hidden_size], 0., 1.);

    let mse_loss = |x, y| { x - y };
    let optimizer = optim::SGD::new(&params);

    for i in 0..30000 {
        let i_f32 = i as f32;
        let data = vec![i_f32*1., i_f32*2., i_f32*3.];
        let y = cpu::Buffer::from_vec(data.iter().map(|x| (*x as f32).sin()).collect(), &[1, input_size]);
        let x = cpu::Buffer::from_vec(data, &[1, input_size]);
        // Don't forget to call .detach() on your hidden state to get just Buffer without graph
        hidden_state = rnn_net.forward((x, hidden_state.clone())).detach();
        let y_predicted = net2.forward(hidden_state.clone());
        let loss = mse_loss(y_predicted, y);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }

    //println!("hidden state: {}", hidden_state);
}
