use zyx::Tensor;
fn main() -> Result<(), zyx::ZyxError> {
    let a = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    let b = a + 1.0;
    println!("{}", b.item::<f32>() * 4.0);
    Ok(())
}
