use super::HasMax;
use duplicate::duplicate_item;

#[duplicate_item( dtype; [i8]; [i16]; [i32]; [i64]; [i128]; [u8]; [u16]; [u32]; [u64]; [u128]; [isize]; [usize];)]
impl HasMax for dtype {
    fn max() -> Self {
        Self::MAX
    }
}
