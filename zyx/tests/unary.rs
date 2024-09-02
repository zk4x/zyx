use zyx::{Tensor, ZyxError};

#[test]
fn relu() -> Result<(), ZyxError> {
    let data: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let zdata: Vec<f32> = Tensor::from(data).relu().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert_eq!(x.max(0.), y);
    }
    Ok(())
}

#[test]
fn neg() -> Result<(), ZyxError> {
    let data: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let zdata: Vec<f32> = (-Tensor::from(data)).try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert_eq!(-x, y);
    }
    Ok(())
}

#[test]
fn exp2() -> Result<(), ZyxError> {
    let data: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let zdata: Vec<f32> = Tensor::from(data).exp2().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert_eq!(x.exp2(), y);
    }
    Ok(())
}

#[test]
fn log2() -> Result<(), ZyxError> {
    let data: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let zdata: Vec<f32> = Tensor::from(data).log2().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        if x.sqrt().is_nan() && y.is_nan() {
            continue
        }
        assert_eq!(x.log2(), y);
    }
    Ok(())
}

#[test]
fn inv() -> Result<(), ZyxError> {
    let data: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let zdata: Vec<f32> = Tensor::from(data).inv().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert_eq!(1./x, y);
    }
    Ok(())
}

#[test]
fn sqrt() -> Result<(), ZyxError> {
    let data: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let zdata: Vec<f32> = Tensor::from(data).sqrt().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        if x.sqrt().is_nan() && y.is_nan() {
            continue
        }
        assert_eq!(x.sqrt(), y);
    }
    Ok(())
}

#[test]
fn sin() -> Result<(), ZyxError> {
    let data: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let zdata: Vec<f32> = Tensor::from(data).sin().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert_eq!(x.sin(), y);
    }
    Ok(())
}

#[test]
fn cos() -> Result<(), ZyxError> {
    let data: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let zdata: Vec<f32> = Tensor::from(data).cos().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert_eq!(x.cos(), y);
    }
    Ok(())
}

#[test]
fn not() -> Result<(), ZyxError> {
    let data: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let zdata: Vec<f32> = (!Tensor::from(data)).try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert_eq!(if *x != 0. { 0. } else { 1. }, y);
    }
    Ok(())
}

#[test]
fn nonzero() -> Result<(), ZyxError> {
    let data: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let zdata: Vec<f32> = Tensor::from(data).nonzero().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert_eq!(if *x != 0. { 1. } else { 0. }, y);
    }
    Ok(())
}
