use super::Max;
use duplicate::duplicate_item;

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl Max for dtype
{
    type Output = Self;
    fn max(self, _: &[i32]) -> Self::Output {
        dtype::MAX
    }
}

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl Max for &dtype
{
    type Output = dtype;
    fn max(self, _: &[i32]) -> Self::Output {
        dtype::MAX
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

impl Max for dtype
{
    type Output = Self;
    fn max(self, _: &[i32]) -> Self::Output {
        dtype::MAX
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

impl Max for &dtype
{
    type Output = dtype;
    fn max(self, _: &[i32]) -> Self::Output {
        dtype::MAX
    }
}
