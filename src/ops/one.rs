use super::One;
use duplicate::duplicate_item;

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl One for dtype {
    fn one() -> Self {
        1.
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

impl One for dtype {
    fn one() -> Self {
        1
    }
}

impl One for bool {
    fn one() -> Self {
        true
    }
}
