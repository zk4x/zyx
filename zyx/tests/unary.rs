use zyx::{Scalar, Tensor, ZyxError};

#[test]
fn relu() -> Result<(), ZyxError> {
    let data: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let zdata: Vec<f32> = Tensor::from(data).relu().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        println!("{} == {y}", x.max(0.));
        assert!(x.max(0.).is_equal(y));
    }
    Ok(())
}

#[test]
fn neg() -> Result<(), ZyxError> {
    let data: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let zdata: Vec<f32> = (-Tensor::from(data)).try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert!((-x).is_equal(y));
    }
    Ok(())
}

#[test]
fn exp2() -> Result<(), ZyxError> {
    let data: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let zdata: Vec<f32> = Tensor::from(data).exp2().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        //assert_eq!(x.exp2(), y);
        assert!(x.exp2().is_equal(y));
    }
    Ok(())
}

#[test]
fn log2() -> Result<(), ZyxError> {
    let data: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let zdata: Vec<f32> = Tensor::from(data).log2().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert!(x.log2().is_equal(y));
    }
    Ok(())
}

#[test]
fn reciprocal() -> Result<(), ZyxError> {
    let data: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let zdata: Vec<f32> = Tensor::from(data).reciprocal().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert!((1. / x).is_equal(y));
    }
    Ok(())
}

#[test]
fn sqrt() -> Result<(), ZyxError> {
    let data: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let zdata: Vec<f32> = Tensor::from(data).sqrt().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert!(x.sqrt().is_equal(y));
    }
    Ok(())
}

#[test]
fn sin() -> Result<(), ZyxError> {
    let data: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let zdata: Vec<f32> = Tensor::from(data).sin().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert!(x.sin().is_equal(y));
    }
    Ok(())
}

#[test]
fn cos() -> Result<(), ZyxError> {
    let data: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let zdata: Vec<f32> = Tensor::from(data).cos().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        //assert_eq!(x.cos(), y);
        assert!(x.cos().is_equal(y));
    }
    Ok(())
}

#[test]
fn not() -> Result<(), ZyxError> {
    let data: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let zdata: Vec<f32> = (!Tensor::from(data)).try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert_eq!(if *x != 0. { 0. } else { 1. }, y);
    }
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn nonzero() -> Result<(), ZyxError> {
    let data: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let zdata: Vec<bool> = Tensor::from(data).nonzero().try_into()?;
    for (&x, y) in data.iter().zip(zdata) {
        assert_eq!(x != 0., y);
    }
    Ok(())
}

#[test]
fn tanh() -> Result<(), ZyxError> {
    let data: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let zdata: Vec<f32> = Tensor::from(data).tanh().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert!(x.tanh().is_equal(y));
    }
    Ok(())
}
