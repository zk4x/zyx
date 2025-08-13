use zyx::{DType, Scalar, Tensor, ZyxError};

#[test]
fn matmul_1024() -> Result<(), ZyxError> {
    //let mut xy: Vec<Tensor> = Tensor::load("xy.safetensors").unwrap();
    //let y = xy.pop().unwrap();
    //let x = xy.pop().unwrap();
    let mut xyz: std::collections::HashMap<String, Tensor> = Tensor::load("./tests/xyz2.safetensors")?;
    let z = xyz.remove("z").unwrap();
    let y = xyz.remove("y").unwrap();
    let x = xyz.remove("x").unwrap();
    //println!("{:?}", x.shape());
    //println!("{:?}", y.shape());
    let dataz: Vec<i64> = z.try_into()?;
    let zz = x.matmul(y)?;
    let datazz: Vec<i64> = zz.try_into()?;
    for (i, (x, y)) in dataz.iter().zip(datazz).enumerate() {
        //println!("{x}, {y}");
        assert!(x.is_equal(y), "{x} != {y} at index {i}");
    }
    //println!("{z}");
    Ok(())
}
