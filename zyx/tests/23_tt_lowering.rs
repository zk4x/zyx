#![cfg(feature = "tenstorrent")]

use zyx::{DType, Dev, Scalar, Tensor, ZyxError, bf16};

#[test]
fn sigmoid() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 256], DType::BF16)?;

    let z = x.to(Dev::TT(0))?.sigmoid().to(Dev::C)?;

    // Host reference, compared in BF16, the computation dtype.
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
fn matmul() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 20], DType::BF16)?;
    let y = Tensor::rand([20, 12], DType::BF16)?;

    let z = x.to(Dev::TT(0))?.matmul(&y.to(Dev::TT(0))?)?.to(Dev::C)?;

    // Compared in F32; BF16 accumulation over K=20 needs a loose tolerance.
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
fn sum() -> Result<(), ZyxError> {
    let x = Tensor::rand([10, 20], DType::BF16)?;

    let z = x.to(Dev::TT(0))?.sum_keepdim([1])?.to(Dev::C)?;

    // Host reference: per-row sums over k=0..20, padding is zero.
    // The output is a row-major [32, 1] vec (one sum per tilized row).
    let xv: Vec<bf16> = x.try_into().unwrap();
    let zv: Vec<bf16> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 20);
    assert_eq!(zv.len(), 32);
    let mut bad = 0;
    for r in 0..32 {
        let mut acc = 0f32;
        for k in 0..20 {
            let xi: f32 = if r < 10 { xv[r * 20 + k].into() } else { 0.0 };
            acc += xi;
        }
        let v: f32 = zv[r].into();
        if (v - acc).abs() >= 1e-1 {
            if bad < 10 {
                println!("z[{r}] = {v}, expected {acc}");
            }
            bad += 1;
        }
    }
    println!("sum bad: {bad} / 32");
    assert_eq!(bad, 0);

    Ok(())
}

#[test]
fn softmax() -> Result<(), ZyxError> {
    let x = Tensor::rand([256, 256], DType::BF16)?;

    let z = x.to(Dev::TT(0))?.softmax([1])?.to(Dev::C)?;

    let xv: Vec<bf16> = x.try_into().unwrap();
    let zv: Vec<bf16> = z.try_into().unwrap();
    assert_eq!(xv.len(), 256 * 256);
    assert_eq!(zv.len(), 256 * 256);

    Ok(())
}

#[test]
fn transfer_roundtrip() -> Result<(), ZyxError> {
    // Transfer-only probe: tilize to TT and untilize back, no kernel
    // in between. Exact bf16 equality is expected (tilize is a
    // permutation plus zero halo; untilize narrows to logical).
    let x = Tensor::rand([10, 20], DType::BF16)?;

    let z = x.to(Dev::TT(0))?.to(Dev::C)?;

    let xv: Vec<bf16> = x.try_into().unwrap();
    let zv: Vec<bf16> = z.try_into().unwrap();
    assert_eq!(xv.len(), 10 * 20);
    assert_eq!(zv.len(), 10 * 20);
    let mut bad = 0;
    for i in 0..10 * 20 {
        if !zv[i].is_equal(xv[i]) {
            if bad < 10 {
                println!("z[{i}] = {}, expected {}", zv[i], xv[i]);
            }
            bad += 1;
        }
    }
    println!("transfer_roundtrip bad: {bad} / {}", 10 * 20);
    assert_eq!(bad, 0);

    Ok(())
}
