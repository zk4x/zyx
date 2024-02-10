use zyx_core::{backend::Backend, error::ZyxError, scalar::Scalar};
use itertools::Itertools;

pub fn t0<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    let x = dev.tensor([[4, 3, 4], [4, 2, 5]]);
    assert_eq!(x.sum(0), [[8, 5, 9]]);
    assert_eq!(x.sum(1), [[11], [11]]);
    assert_eq!(x.sum(()), [[22]]);
    Ok(())
}

pub fn t1<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    let x = dev.tensor([[2, 4, 3], [5, 2, 4]]);
    let y = dev.tensor([[2, 2, 4], [1, 2, 1], [3, 4, 2]]);
    let z = x.dot(&y);
    assert_eq!(z, [[17, 24, 18], [24, 30, 30]]);

    let (xgrad, ygrad) = z.backward([&x, &y]).flatten().collect_tuple().unwrap();
    assert_eq!(xgrad, [[8, 4, 9], [8, 4, 9]]);
    assert_eq!(ygrad, [[7, 7, 7], [6, 6, 6], [7, 7, 7]]);
    Ok(())
}
