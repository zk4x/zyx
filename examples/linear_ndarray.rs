//! ## This is an example of linear neural network with sequential model

//#[cfg(feature = "ndarray")]
fn main() {
    /*use zyx::prelude::*;
    use ndarray::{array, OwnedRepr, Dim, ArrayBase, Ix2, Ix1};
    use zyx::nn::{Linear, ReLU};
    use zyx::optim;

    let network = (
        Linear::new::<f32>(1, 20),
        ReLU,
        Linear::new::<f32>(20, 50),
        ReLU,
        Linear::new::<f32>(50, 1),
        ReLU,
    );

    // MSELoss does not reduce it's output (it's just (y-yp)^2), you need to add some reduce function if you want to apply reduce
    let loss_fn = |(x, y)| x - y; //(MSELoss, Mean { dims: () });

    // This looks bad right now, eventually it will look like this:
    //let optimizer = optim::SGD::new(network.parameters()).with_learning_rate(0.03);
    let optimizer = optim::SGD::new(<&(
        Linear<ArrayBase<OwnedRepr<f32>, Ix2>, _, ArrayBase<OwnedRepr<f32>, Ix1>, _>, ReLU,
        Linear<ArrayBase<OwnedRepr<f32>, Ix2>, _, ArrayBase<OwnedRepr<f32>, Ix1>, _>, ReLU,
        Linear<ArrayBase<OwnedRepr<f32>, Ix2>, _, ArrayBase<OwnedRepr<f32>, Ix1>, _>, ReLU) as Module<ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>>>::parameters(&network)).with_learning_rate(0.03);

    for _ in 0..100 {
        for i in 0..100 {
            let x = array![i as f32 / 10.];
            let y = array![(i as f32 / 10.).sin()];

            let y_predicted = network.forward(x);

            let loss = (y_predicted, y).apply(&loss_fn);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
    }*/
}
