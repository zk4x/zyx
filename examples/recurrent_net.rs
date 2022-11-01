//! ## This is an example of recurrent neural network
/*use zyx::prelude::*;
use zyx::accel::cpu;
use zyx::nn::{RNNCell, Linear, SoftMax, MSELoss, Sum, ReLU};
use zyx::optim;*/

fn main() {
    /*let hidden_size = 10;
    let input_size = 3;

    let rnn_net = (
        RNNCell::new::<f32>(input_size, hidden_size),
        ReLU,
    );
    let net2 = (
        Linear::new::<f32>(hidden_size, 3),
        SoftMax { dims: () },
    );

    // This looks bad right now, eventually it will look like this:
    //let params = (rnn_net.parameters(), net2.parameters());
    let params = (
        <&(RNNCell<cpu::Buffer<f32>>, ReLU) as Module<(cpu::Buffer<f32>, cpu::Buffer<f32>)>>::parameters(&rnn_net),
        <&(Linear<cpu::Buffer<f32>>, SoftMax<()>) as Module<cpu::Buffer<f32>>>::parameters(&net2),
    );

    let mut hidden_state = cpu::Buffer::uniform((1, hidden_size), 0., 1.);

    // MSELoss does not reduce it's output, you need to add some reduce function if you want to apply reduce
    let mse_loss = (MSELoss, Sum { dims: () });

    let optimizer = optim::SGD::new(params);

    for i in 0..30000 {
        let i_f32 = i as f32;
        let data = vec![i_f32*1., i_f32*2., i_f32*3.];

        let y = cpu::Buffer::from_vec(data.iter().map(|x| (*x as f32).sin()).collect(), (1, input_size));
        let x = cpu::Buffer::from_vec(data, (1, input_size));

        let hidden_state_t1 = rnn_net.forward((x, hidden_state.clone()));
        // Don't forget to get data on your hidden state to get just Buffer without graph
        hidden_state = hidden_state_t1.data().clone();

        let y_predicted = net2.forward(hidden_state_t1);
        let loss = (y_predicted, y).apply(&mse_loss);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }*/

    //println!("hidden state: {}", hidden_state);
}
