use super::assert_eq;
use zyx_core::{
    scalar::Scalar,
    backend::Backend,
    shape::Shape,
    error::ZyxError,
};
use core::ops::{Add, Sub, Div, Mul};

macro_rules! binary_test {
    ( $dev:expr, $x:tt ) => {{
        // TODO random generation of different shapes
        let shapes: &[Shape] = &[[4, 7, 4, 3].into(), [1, 4].into(), [1801923].into(), [423, 1938].into(), [1024, 1024].into(), [4097, 1049].into()];
        for shape in shapes {
            let x = $dev.randn(shape, T::dtype());
            let y = $dev.randn(shape, T::dtype());
            let vx = x.to_vec()?;
            let vy = y.to_vec()?;
            assert_eq(x.$x(y).to_vec()?, vx.into_iter().zip(vy).map(|(ex, ey): (T, T)| ex.$x(ey)));
        }
        Ok(())
    }};
}

pub fn add<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    binary_test!(dev, add)
}

pub fn sub<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    binary_test!(dev, sub)
}

pub fn mul<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    binary_test!(dev, mul)
}

pub fn div<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    binary_test!(dev, div)
}

pub fn pow<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    binary_test!(dev, pow)
}

pub fn cmplt<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    binary_test!(dev, cmplt)
}
