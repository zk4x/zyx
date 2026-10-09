use zyx::{DType, Tensor, ZyxError};

#[test]
fn sigmoid() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 256], DType::BF16)?.tilize()?;

    let z = x.sigmoid().untilize(32, 32)?;

    Ok(())
}

#[test]
fn matmul() -> Result<(), ZyxError> {
    let x = Tensor::rand([256, 256], DType::BF16)?.tilize()?;
    let y = Tensor::rand([256, 256], DType::BF16)?.tilize()?;

    let z = x.matmul(y)?.untilize(32, 32)?;

    Ok(())
}

#[test]
fn softmax() -> Result<(), ZyxError> {
    let x = Tensor::rand([256, 256], DType::BF16)?.tilize()?;

    let z = x.softmax([1])?.untilize(32, 32)?;

    Ok(())
}
