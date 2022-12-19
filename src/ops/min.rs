use super::Min;
use crate::shape::Ax0;
use duplicate::duplicate_item;

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl Min<Ax0> for dtype
{
    type Output = Self;
    fn min(self) -> Self::Output {
        dtype::MIN
    }
}

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl Min<Ax0> for &dtype
{
    type Output = dtype;
    fn min(self) -> Self::Output {
        dtype::MIN
    }
}

#[duplicate_item(
    dtype;
    [i8];
    [i16];
    [i32];
    [i64];
    [i128];
    [u8];
    [u16];
    [u32];
    [u64];
    [u128];
    [isize];
    [usize];
)]

impl Min<Ax0> for dtype
{
    type Output = Self;
    fn min(self) -> Self::Output {
        dtype::MIN
    }
}

#[duplicate_item(
    dtype;
    [i8];
    [i16];
    [i32];
    [i64];
    [i128];
    [u8];
    [u16];
    [u32];
    [u64];
    [u128];
    [isize];
    [usize];
)]

impl Min<Ax0> for &dtype
{
    type Output = dtype;
    fn min(self) -> Self::Output {
        dtype::MIN
    }
}
