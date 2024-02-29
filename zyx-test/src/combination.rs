use zyx_core::{backend::Backend, error::ZyxError, scalar::Scalar};
use itertools::Itertools;

pub fn t0<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    let x = dev.tensor([[4, 3, 4], [4, 2, 5]])?;
    assert_eq!(x.sum(0), [[8, 5, 9]]);
    assert_eq!(x.sum(1), [[11], [11]]);
    assert_eq!(x.sum(()), [[22]]);
    Ok(())
}

pub fn dot<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    let x = dev.tensor([[2, 4, 3], [5, 2, 4]])?;
    let y = dev.tensor([[2, 2, 4], [1, 2, 1], [3, 4, 2]])?;
    let z = x.dot(&y);
    assert_eq!(z, [[17, 24, 18], [24, 30, 30]]);

    let (x_grad, y_grad) = z.backward([&x, &y]).into_iter().flatten().collect_tuple().unwrap();

    assert_eq!(x_grad, [[8, 4, 9], [8, 4, 9]]);
    //println!("{y_grad}");
    assert_eq!(y_grad, [[7, 7, 7], [6, 6, 6], [7, 7, 7]]);
    Ok(())
}

pub fn t1<T: Scalar>(_: impl Backend, _: T) -> Result<(), ZyxError> {
    //let x = dev.tensor([[2, 4, 3], [5, 2, 4], [3, 1, 2]])?;
    //let y = dev.tensor([[2, 2, 4], [1, 2, 1], [3, 4, 2]]);
    //let z = x.dot(&x) + x.exp() + x.tanh() + x.sum(0);
    //println!("{}", z);
    //assert_eq!([50, 80, ]);
    // TODO
    Ok(())
}

pub fn cat<T: Scalar>(_dev: impl Backend, _: T) -> Result<(), ZyxError> {
    // TODO
    Ok(())
}

pub fn split<T: Scalar>(_dev: impl Backend, _: T) -> Result<(), ZyxError> {
    // TODO
    Ok(())
}
