use zyx_core::{backend::Backend, error::ZyxError, scalar::Scalar};

pub fn t0<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    let x = dev.tensor([[4, 3, 4], [4, 2, 5]]);
    assert_eq!(x.sum(0), [[8, 5, 9]]);
    assert_eq!(x.sum(1), [[11], [11]]);
    println!("{}", x.sum(()));
    assert_eq!(x.sum(()), [[22]]);
    Ok(())
}

pub fn t1<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    Ok(())
}
