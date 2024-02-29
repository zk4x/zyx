use zyx_core::backend::Backend;
use zyx_core::error::ZyxError;
use zyx_core::scalar::Scalar;

pub fn sgd<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    let mut p0 = dev.tensor([[2f32, 3., 4.], [4., 3., 2.]])?;
    let mut p1 = dev.tensor([[2f32, 3., 4.], [4., 3., 2.], [5., 4., 3.]])?;
    let l0 = dev.tensor([[2f32, 3., 4.], [4., 3., 2.]])?;

    //let p00 = p0.clone();
    //let p10 = p1.clone();

    //println!("{p0}\n{p1}");

    let loss = (p0.dot(&p1) + &p0 - &l0).pow(2);
    let grads = loss.backward([&p0, &p1]);

    let mut optim = zyx_optim::SGD { learning_rate: 0.01, momentum: 0.9, nesterov: true, ..Default::default() };
    optim.update([&mut p0, &mut p1], grads);

    //std::fs::write("graph.dot", dev.plot_graph([&p0, &p1, &p00, &p10])).unwrap();

    //println!("{p0:.6}\n\n{p1:.6}");

    assert_eq!(p0, [[-9.590f32, -9.160, -11.503999], [-6.981999, -8.096, -12.363998]]);
    assert_eq!(p1, [[-5.296f32, -3.764, -2.232], [-3.524, -3.840, -4.156], [-2.751999, -2.916, -3.080]]);

    Ok(())
}

pub fn adam<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    let mut p0 = dev.tensor([[2, 3, 4], [4, 3, 2]])?;
    let mut p1 = dev.tensor([[2, 3, 4], [4, 3, 2], [5, 4, 3]])?;
    let l0 = dev.tensor([[2, 3, 4], [4, 3, 2]])?;

    //println!("{p0}\n{p1}");

    let loss = (p0.dot(&p1) + &p0 - &l0).pow(2);
    let grads = loss.backward([&p0, &p1]);

    let mut optim = zyx_optim::Adam { learning_rate: 0.01, ..Default::default() };
    optim.update([&mut p0, &mut p1], grads);

    //println!("{p0}\n{p1}");

    assert_eq!(p0, [3]);
    assert_eq!(p1, [4]);

    Ok(())
}
