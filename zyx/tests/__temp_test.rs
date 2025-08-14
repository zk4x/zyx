use zyx::{Scalar, Tensor, ZyxError};

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
