use super::GetShape;
use duplicate::duplicate_item;

#[duplicate_item(
    dtype;
    [f32];
    [f64];
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

impl GetShape for dtype {
    type Output = ();

    fn shape(&self) -> Self::Output {}
}
