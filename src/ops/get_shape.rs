use super::GetShape;
use crate::shape::Shape;
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
    fn shape(&self) -> Shape {
        use super::IntoShape;
        IntoShape::shape(1)
    }
}
