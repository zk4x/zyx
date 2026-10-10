#![cfg(feature = "tenstorrent")]

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
    let x = Tensor::rand([10, 20], DType::BF16)?;
    let y = Tensor::rand([20, 12], DType::BF16)?;

    let z = x.tilize()?.to(Dev::TT(0))?.matmul(&y.tilize()?.to(Dev::TT(0))?)?.to(Dev::C)?.untilize(32, 32)?;

    // Host reference over the matching input regions. Padding is zero,
    // so rows 10.. and cols 12.. accumulate only zeros. Compared in F32;
    // BF16 accumulation over K=20 needs a loose tolerance.
    let xv: Vec<bf16> = x.try_into().unwrap();
    let yv: Vec<bf16> = y.try_into().unwrap();
    let zv: Vec<bf16> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 20);
    assert_eq!(yv.len(), 20 * 12);
    assert_eq!(zv.len(), 32 * 32);
    let mut bad = 0;
    for r in 0..32 {
        for c in 0..32 {
            let mut acc = 0f32;
            for k in 0..20 {
                let xi: f32 = if r < 10 { xv[r * 20 + k].into() } else { 0.0 };
                let yi: f32 = if c < 12 { yv[k * 12 + c].into() } else { 0.0 };
                acc += xi * yi;
            }
            let v: f32 = zv[r * 32 + c].into();
            if (v - acc).abs() >= 1e-1 {
                if bad < 10 {
                    println!("z[{r},{c}] = {v}, expected {acc}");
                }
                bad += 1;
            }
        }
    }
    println!("matmul bad: {bad} / 1024");
    assert_eq!(bad, 0);

    Ok(())
}

#[test]
fn softmax() -> Result<(), ZyxError> {
    let x = Tensor::rand([256, 256], DType::BF16)?.tilize()?;

    let z = x.softmax([1])?.untilize(32, 32)?;

    Ok(())
}
