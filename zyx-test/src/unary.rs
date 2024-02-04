use super::assert_eq;
use zyx_core::{
    backend::Backend,
    error::ZyxError,
    scalar::Scalar,
    shape::Shape,
};
use core::ops::Neg;

macro_rules! unary_test {
    ( $dev:expr, $x:tt ) => {{
        // TODO random generation of different shapes
        let shapes: &[Shape] = &[[4, 7, 4, 3].into(), [1, 4].into(), [1801923].into(), [423, 1938].into(), [1024, 1024].into(), [4097, 1049].into()];
        for shape in shapes {
            let x = $dev.randn(shape, T::dtype());
            let v = x.to_vec()?;
            assert_eq(x.$x().to_vec()?, v.into_iter().map(|e: T| e.$x()));
        }
        Ok(())
    }};
}

pub fn neg<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    unary_test!(dev, neg)
}

pub fn relu<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    unary_test!(dev, relu)
}

pub fn sin<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    unary_test!(dev, sin)
}

pub fn cos<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    unary_test!(dev, cos)
}

pub fn ln<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    unary_test!(dev, ln)
}

pub fn exp<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    unary_test!(dev, exp)
}

pub fn tanh<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    unary_test!(dev, tanh)
}

pub fn sqrt<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    unary_test!(dev, sqrt)
}
