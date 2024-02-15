use zyx_core::backend::Backend;
use zyx_core::error::ZyxError;
use zyx_core::scalar::Scalar;

pub fn sgd<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    let mut p0 = dev.tensor([[2, 3, 4], [4, 3, 2]]);
    let mut p1 = dev.tensor([[2, 3, 4], [4, 3, 2], [5, 4, 3]]);
    let l0 = dev.tensor([[2, 3, 4], [4, 3, 2]]);

    //println!("{p0}\n{p1}");

    let loss = (p0.dot(&p1) + &p0 - &l0).pow(2);
    let grads = loss.backward([&p0, &p1]);

    let mut optim = zyx_optim::SGD { learning_rate: 0.01, momentum: 0.9, nesterov: true, ..Default::default() };
    optim.update([&mut p0, &mut p1], grads);

    //println!("{p0}\n{p1}");

    Ok(())
}
