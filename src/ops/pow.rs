use super::Pow;
use duplicate::duplicate_item;

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl Pow for dtype {
    type Output = Self;
    fn pow(self, rhs: Self) -> Self::Output {
        self.powf(rhs)
    }
}

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl Pow for &dtype {
    type Output = dtype;
    fn pow(self, rhs: Self) -> Self::Output {
        self.powf(*rhs)
    }
}

#[duplicate_item(
    dtype;
    [u8];
    [u16];
    [u64];
    [u128];
    [i8];
    [i16];
    [i32];
    [i64];
    [i128];
)]

impl Pow for dtype {
    type Output = Self;
    fn pow(self, rhs: Self) -> Self::Output {
        self.pow(rhs.try_into().unwrap())
    }
}

impl Pow for u32 {
    type Output = Self;
    fn pow(self, rhs: Self) -> Self::Output {
        self.pow(rhs)
    }
}

#[duplicate_item(
    dtype;
    [u8];
    [u16];
    [u64];
    [u128];
    [i8];
    [i16];
    [i32];
    [i64];
    [i128];
)]

impl Pow for &dtype {
    type Output = dtype;
    fn pow(self, rhs: Self) -> Self::Output {
        (*self).pow((*rhs).try_into().unwrap())
    }
}

impl Pow for &u32 {
    type Output = u32;
    fn pow(self, rhs: Self) -> Self::Output {
        (*self).pow(*rhs)
    }
}
