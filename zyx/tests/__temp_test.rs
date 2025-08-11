use zyx::{Tensor, ZyxError};

#[test]
fn sum_1() -> Result<(), ZyxError> {
    let x = Tensor::from([[4, 1, 3], [5, 2, 3], [6, 5, 7]]);
    let x0 = x.sum([-1])?;
    assert_eq!(x0, [8, 10, 18]);
    Ok(())
}
