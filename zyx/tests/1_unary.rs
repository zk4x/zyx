use zyx::{DType, Scalar, Tensor, ZyxError};

#[test]
fn relu_1() -> Result<(), ZyxError> {
    let data: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let zdata: Vec<f32> = Tensor::from(data).relu().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        //println!("{} == {y}", x.max(0.));
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
        //println!("{}, {y}", 1. / x);
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
        assert!(x.sin().is_equal(y), "{} != {y}", x.sin());
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
        assert!(x.cos().is_equal(y), "{} != {y}", x.cos());
    }
    Ok(())
}

#[test]
fn not() -> Result<(), ZyxError> {
    let data: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let y = !Tensor::from(data);
    let x = y.cast(DType::F32);
    drop(y); // We have to drop manually, because rust is very unreliable in calling destructors
    let zdata: Vec<f32> = x.try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert_eq!(if *x != 0. { 0. } else { 1. }, y);
    }
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn nonzero() -> Result<(), ZyxError> {
    let data: [f32; 10] = [-3.285, 0.00, 1.780, 5.675, -8.521, -0.456, 1.215, 0.00, -4.128, -7.657];
    let z = Tensor::from(data).nonzero();
    assert_eq!(z, [true, false, true, true, true, true, true, false, true, true]);
    Ok(())
}

#[test]
fn tanh_1() -> Result<(), ZyxError> {
    let data: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let zdata: Vec<f32> = Tensor::from(data).tanh().try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert!(x.tanh().is_equal(y));
    }
    Ok(())
}

#[test]
fn tanh_2() -> Result<(), ZyxError> {
    let x = Tensor::from([[2, 4, 1], [5, 4, 1]]).cast(DType::F32);
    let x = x.tanh();
    assert_eq!(x, [[0.964028f32, 0.999329, 0.761594], [0.999909, 0.999329, 0.761594]]);
    Ok(())
}
