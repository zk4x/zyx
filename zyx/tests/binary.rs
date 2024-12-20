use zyx::{Scalar, Tensor, ZyxError};

#[test]
fn add() -> Result<(), ZyxError> {
    let datax: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let datay: [f32; 10] = [
        2.772, -8.327, 1.945, 9.286, 3.989, 8.105, -5.307, 2.865, 3.106, 3.111,
    ];
    let x = Tensor::from(datax);
    let y = Tensor::from(datay);
    let z = x + y;
    let dataz: Vec<f32> = z.try_into()?;
    for ((x, y), z) in datax.iter().zip(datay).zip(dataz) {
        assert_eq!(x + y, z);
    }
    Ok(())
}

#[test]
fn sub() -> Result<(), ZyxError> {
    let datax: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let datay: [f32; 10] = [
        2.772, -8.327, 1.945, 9.286, 3.989, 8.105, -5.307, 2.865, 3.106, 3.111,
    ];
    let x = Tensor::from(datax);
    let y = Tensor::from(datay);
    let z = x - y;
    println!("{z}");
    let dataz: Vec<f32> = z.try_into()?;
    for ((x, y), z) in datax.iter().zip(datay).zip(dataz) {
        assert_eq!(x - y, z);
    }
    Ok(())
}

#[test]
fn mul() -> Result<(), ZyxError> {
    let datax: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let datay: [f32; 10] = [
        2.772, -8.327, 1.945, 9.286, 3.989, 8.105, -5.307, 2.865, 3.106, 3.111,
    ];
    let x = Tensor::from(datax);
    let y = Tensor::from(datay);
    let z = x * y;
    let dataz: Vec<f32> = z.try_into()?;
    for ((x, y), z) in datax.iter().zip(datay).zip(dataz) {
        assert_eq!(x * y, z);
    }
    Ok(())
}

#[test]
fn div() -> Result<(), ZyxError> {
    let datax: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let datay: [f32; 10] = [
        2.772, -8.327, 1.945, 9.286, 3.989, 8.105, -5.307, 2.865, 3.106, 3.111,
    ];
    let x = Tensor::from(datax);
    let y = Tensor::from(datay);
    let z = x / y;
    let dataz: Vec<f32> = z.try_into()?;
    for ((x, y), z) in datax.iter().zip(datay).zip(dataz) {
        assert!((x / y - z).abs() < 0.00001);
    }
    Ok(())
}

#[test]
fn pow() -> Result<(), ZyxError> {
    let datax: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let datay: [f32; 10] = [
        2.772, -8.327, 1.945, 9.286, 3.989, 8.105, -5.307, 2.865, 3.106, 3.111,
    ];
    let x = Tensor::from(datax);
    let y = Tensor::from(datay);
    let z = x.pow(y)?;
    let dataz: Vec<f32> = z.try_into()?;
    for ((x, y), z) in datax.iter().zip(datay).zip(dataz) {
        //assert!((x.pow(y) - z).abs() < 0.00001);
        let x = x.pow(y);
        //println!("{x}, {z}");
        assert!(x.is_equal(z));
    }
    Ok(())
}

#[test]
fn max() -> Result<(), ZyxError> {
    let datax: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let datay: [f32; 10] = [
        2.772, -8.327, 1.945, 9.286, 3.989, 8.105, -5.307, 2.865, 3.106, 3.111,
    ];
    let x = Tensor::from(datax);
    let y = Tensor::from(datay);
    let z = x.maximum(y)?;
    let dataz: Vec<f32> = z.try_into()?;
    for ((x, y), z) in datax.iter().zip(datay).zip(dataz) {
        assert_eq!(x.max(y), z);
    }
    Ok(())
}

#[test]
fn cmplt() -> Result<(), ZyxError> {
    let datax: [f32; 10] = [
        -3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657,
    ];
    let datay: [f32; 10] = [
        2.772, -8.327, 1.945, 9.286, 3.989, 8.105, -5.307, 2.865, 3.106, 3.111,
    ];
    let x = Tensor::from(datax);
    let y = Tensor::from(datay);
    // We cast here since not all backends support bool dtype buffers.
    let z = x.cmplt(y)?.cast(zyx::DType::U32);
    let dataz: Vec<u32> = z.try_into()?;
    for ((x, y), z) in datax.iter().zip(datay).zip(dataz) {
        assert_eq!(x.cmplt(y) as u32, z);
    }
    Ok(())
}
