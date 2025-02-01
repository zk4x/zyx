use zyx::{Tensor, ZyxError};

#[test]
fn reshape() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[4, 5, 2, 1], [3, 4, 1, 4]]);
    x = x.reshape([8, 1])?;
    x = x.reshape([1, 2, 1, 4])?;
    x = x.reshape([4, 2])?;
    assert_eq!(x, [[4, 5], [2, 1], [3, 4], [1, 4]]);
    Ok(())
}

#[test]
fn expand() -> Result<(), ZyxError> {
    Ok(())
}

#[test]
fn permute() -> Result<(), ZyxError> {
    Ok(())
}

#[test]
fn pad_zeros() -> Result<(), ZyxError> {
    Ok(())
}
