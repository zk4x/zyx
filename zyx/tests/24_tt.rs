use zyx::{DType, Dev, Scalar, Tensor, ZyxError, bf16};

fn tt() -> Dev {
    Dev::TT(0)
}

#[test]
fn transfer_roundtrip() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 20], DType::BF16)?;
    let z = x.to(tt())?.to(Dev::C)?;

    let xv: Vec<bf16> = x.try_into().unwrap();
    let zv: Vec<bf16> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 20);
    assert_eq!(zv.len(), 10 * 20);
    for i in 0..10 * 20 {
        assert!(zv[i].is_equal(xv[i]), "transfer_roundtrip z[{i}] = {}, expected {}", zv[i], xv[i]);
    }
    Ok(())
}

#[test]
fn matmul() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 20], DType::BF16)?;
    let y = Tensor::rand([20, 12], DType::BF16)?;

    let z = x.to(tt())?.matmul(&y.to(tt())?)?.to(Dev::C)?;

    let xv: Vec<bf16> = x.try_into().unwrap();
    let yv: Vec<bf16> = y.try_into().unwrap();
    let zv: Vec<bf16> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 20);
    assert_eq!(yv.len(), 20 * 12);
    assert_eq!(zv.len(), 10 * 12);
    let mut bad = 0;
    for r in 0..10 {
        for c in 0..12 {
            let mut acc = 0f32;
            for k in 0..20 {
                acc += f32::from(xv[r * 20 + k]) * f32::from(yv[k * 12 + c]);
            }
            let v: f32 = zv[r * 12 + c].into();
            if (v - acc).abs() >= 1e-1 {
                if bad < 10 {
                    println!("z[{r},{c}] = {v}, expected {acc}");
                }
                bad += 1;
            }
        }
    }
    println!("matmul bad: {bad} / {}", 10 * 12);
    assert_eq!(bad, 0);
    Ok(())
}

#[test]
fn sigmoid() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 256], DType::BF16)?;
    let z = x.to(tt())?.sigmoid().to(Dev::C)?;

    let xv: Vec<bf16> = x.try_into().unwrap();
    let zv: Vec<bf16> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 256);
    assert_eq!(zv.len(), 10 * 256);
    let mut bad = 0;
    for i in 0..10 * 256 {
        let xi: f32 = xv[i].into();
        let expected = bf16::from_f32(1.0 / (1.0 + (-xi).exp()));
        let v = zv[i];
        if !v.is_equal(expected) {
            if bad < 10 {
                println!("z[{i}] = {v}, expected {expected}");
            }
            bad += 1;
        }
    }
    println!("sigmoid bad: {bad} / {}", 10 * 256);
    assert_eq!(bad, 0);
    Ok(())
}

#[test]
fn sum() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 20], DType::BF16)?;

    let z = x.to(tt())?.sum_keepdim([1])?.to(Dev::C)?;

    let xv: Vec<bf16> = x.try_into().unwrap();
    let zv: Vec<bf16> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 20);
    assert_eq!(zv.len(), 10);
    let mut bad = 0;
    for r in 0..10 {
        let mut acc = 0f32;
        for k in 0..20 {
            acc += f32::from(xv[r * 20 + k]);
        }
        let v: f32 = zv[r].into();
        if (v - acc).abs() >= 1e-1 {
            if bad < 10 {
                println!("z[{r}] = {v}, expected {acc}");
            }
            bad += 1;
        }
    }
    println!("sum bad: {bad} / 10");
    assert_eq!(bad, 0);
    Ok(())
}
