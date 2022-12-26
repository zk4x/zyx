use super::ZerosLike;
use duplicate::duplicate_item;

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl ZerosLike for dtype {
    fn zeros_like(&self) -> Self {
        0.
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

impl ZerosLike for dtype {
    fn zeros_like(&self) -> Self {
        0
    }
}
