use zyx::{DType, Dev, Scalar, Tensor, ZyxError, bf16};

#[test]
fn sigmoid() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 256], DType::BF16)?;

    let z = x.tilize()?.to(Dev::TT(0))?.sigmoid().to(Dev::C)?.untilize(32, 32)?;

    // Host reference over the matching input region: rows 10..32 are
    // zero padding, so sigmoid(0) = 0.5 there. Compared in BF16, the
    // computation dtype.
    let xv: Vec<bf16> = x.try_into().unwrap();
    let zv: Vec<bf16> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 256);
    assert_eq!(zv.len(), 32 * 32);
    let mut bad = 0;
    for r in 0..32 {
        for c in 0..32 {
            let xi: f32 = if r < 10 { xv[r * 256 + c].into() } else { 0.0 };
            let expected = bf16::from_f32(1.0 / (1.0 + (-xi).exp()));
            let v = zv[r * 32 + c];
            if !v.is_equal(expected) {
                if bad < 10 {
                    println!("z[{r},{c}] = {v}, expected {expected}");
                }
                bad += 1;
            }
        }
    }
    println!("sigmoid bad: {bad} / 1024");
    assert_eq!(bad, 0);

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
