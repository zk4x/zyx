pub fn linear<T: Scalar>(_: T) -> Result<(), ZyxError> {
    let l0 = Linear::new(4, 7, T::dtype());
    let x = Tensor::uniform([2, 3, 4], 0f32..1f32)?;
    let z = l0.forward(&x);
    assert_eq!(z, x.dot(&l0.weight) + &l0.bias.unwrap());
    Ok(())
}

pub fn layer_norm<T: Scalar>(_: T) -> Result<(), ZyxError> {
    todo!()
}

pub fn batch_norm<T: Scalar>(_: T) -> Result<(), ZyxError> {
    todo!()
}
