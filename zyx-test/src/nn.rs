use zyx_core::backend::Backend;
use zyx_core::error::ZyxError;
use zyx_core::scalar::Scalar;

pub fn linear<T: Scalar>(_dev: impl Backend, _: T) -> Result<(), ZyxError> {
    todo!()
}

pub fn layer_norm<T: Scalar>(_dev: impl Backend, _: T) -> Result<(), ZyxError> {
    todo!()
}

pub fn batch_norm<T: Scalar>(_dev: impl Backend, _: T) -> Result<(), ZyxError> {
    todo!()
}
