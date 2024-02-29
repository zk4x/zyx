use zyx_core::backend::Backend;
use zyx_core::error::ZyxError;
use zyx_core::scalar::Scalar;

pub fn linear<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    use zyx_nn::prelude::*;
    let l0 = dev.linear(4, 7);
    let x = dev.uniform([2, 3, 4], 0f32..1f32)?;
    let z = l0.forward(&x);
    assert_eq!(z, x.dot(&l0.weight) + &l0.bias.unwrap());
    Ok(())
}

pub fn layer_norm<T: Scalar>(_dev: impl Backend, _: T) -> Result<(), ZyxError> {
    todo!()
}

pub fn batch_norm<T: Scalar>(_dev: impl Backend, _: T) -> Result<(), ZyxError> {
    todo!()
}
