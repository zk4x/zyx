use zyx::{Tensor, ZyxError};

#[test]
fn sum_1() -> Result<(), ZyxError> {
    let x = Tensor::from([[4, 1, 3], [5, 2, 3], [6, 5, 7]]);
    let x0 = x.sum([-1])?;
    assert_eq!(x0, [8, 10, 18]);
    Ok(())
}

#[test]
fn sum_2() -> Result<(), ZyxError> {
    let x = Tensor::from([[4, 1, 3], [5, 2, 3], [6, 5, 7]]);
    let x0 = x.sum([-1])?;
    let x1 = x.sum([-2])?;
    let x2 = x.sum([])?;
    assert_eq!(x0, [8, 10, 18]);
    assert_eq!(x1, [15, 8, 13]);
    assert_eq!(x2, [36]);
    Ok(())
}

#[test]
fn max_1() -> Result<(), ZyxError> {
    let x = Tensor::from([[4, 1, 3], [5, 2, 3], [6, 5, 7]]);
    let x0 = x.max([-1])?;
    let x1 = x.max([-2])?;
    let x2 = x.max([])?;
    assert_eq!(x0, [4, 5, 7]);
    assert_eq!(x1, [6, 5, 7]);
    assert_eq!(x2, [7]);
    Ok(())
}
