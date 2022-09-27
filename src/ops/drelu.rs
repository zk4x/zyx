use super::DReLU;
use duplicate::duplicate_item;

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl DReLU for dtype {
    type Output = Self;
    fn drelu(self) -> Self::Output {
        if self < 0. { 0. } else { 1. }
    }
}

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl DReLU for &dtype {
    type Output = dtype;
    fn drelu(self) -> Self::Output {
        if *self < 0. { 0. } else { 1. }
    }
}

#[duplicate_item(
    dtype;
    [i8];
    [i16];
    [i32];
    [i64];
    [i128];
    [isize];
)]

impl DReLU for dtype {
    type Output = Self;
    fn drelu(self) -> Self::Output {
        dtype::from(self >= 0)
    }
}

#[duplicate_item(
    dtype;
    [i8];
    [i16];
    [i32];
    [i64];
    [i128];
    [isize];
)]

impl DReLU for &dtype {
    type Output = dtype;
    fn drelu(self) -> Self::Output {
        dtype::from(*self >= 0)
    }
}

#[duplicate_item(
    dtypeu;
    [u8];
    [u16];
    [u32];
    [u64];
    [u128];
    [usize];
)]

impl DReLU for dtypeu {
    type Output = Self;
    fn drelu(self) -> Self::Output {
        1
    }
}
