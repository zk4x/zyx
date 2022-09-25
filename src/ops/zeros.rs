use super::Zeros;
use duplicate::duplicate_item;

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl Zeros for dtype {
    fn zeros(_: &[usize]) -> Self {
        0.
    }
}

#[duplicate_item(
    dtypei;
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

impl Zeros for dtypei {
    fn zeros(_: &[usize]) -> Self {
        0
    }
}
