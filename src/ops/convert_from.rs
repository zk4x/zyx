use super::ConvertFrom;
use duplicate::duplicate_item;

#[duplicate_item( dtype; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<f32> for dtype {
    fn cfrom(x: f32) -> Self {
        x as Self
    }
}

impl ConvertFrom<f32> for f32 {
    fn cfrom(x: f32) -> Self {
        x
    }
}

#[duplicate_item( dtype; [f32]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<f64> for dtype {
    fn cfrom(x: f64) -> Self {
        x as Self
    }
}

impl ConvertFrom<f64> for f64 {
    fn cfrom(x: f64) -> Self {
        x
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<i8> for dtype {
    fn cfrom(x: i8) -> Self {
        x as Self
    }
}

impl ConvertFrom<i8> for i8 {
    fn cfrom(x: i8) -> Self {
        x
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<i16> for dtype {
    fn cfrom(x: i16) -> Self {
        x as Self
    }
}

impl ConvertFrom<i16> for i16 {
    fn cfrom(x: i16) -> Self {
        x
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<i32> for dtype {
    fn cfrom(x: i32) -> Self {
        x as Self
    }
}

impl ConvertFrom<i32> for i32 {
    fn cfrom(x: i32) -> Self {
        x
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<i64> for dtype {
    fn cfrom(x: i64) -> Self {
        x as Self
    }
}

impl ConvertFrom<i64> for i64 {
    fn cfrom(x: i64) -> Self {
        x
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<i128> for dtype {
    fn cfrom(x: i128) -> Self {
        x as Self
    }
}

impl ConvertFrom<i128> for i128 {
    fn cfrom(x: i128) -> Self {
        x
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<isize> for dtype {
    fn cfrom(x: isize) -> Self {
        x as Self
    }
}

impl ConvertFrom<isize> for isize {
    fn cfrom(x: isize) -> Self {
        x
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<u8> for dtype {
    fn cfrom(x: u8) -> Self {
        x as Self
    }
}

impl ConvertFrom<u8> for u8 {
    fn cfrom(x: u8) -> Self {
        x
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<u16> for dtype {
    fn cfrom(x: u16) -> Self {
        x as Self
    }
}

impl ConvertFrom<u16> for u16 {
    fn cfrom(x: u16) -> Self {
        x
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u64]; [u128]; [usize])]
impl ConvertFrom<u32> for dtype {
    fn cfrom(x: u32) -> Self {
        x as Self
    }
}

impl ConvertFrom<u32> for u32 {
    fn cfrom(x: u32) -> Self {
        x
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u128]; [usize])]
impl ConvertFrom<u64> for dtype {
    fn cfrom(x: u64) -> Self {
        x as Self
    }
}

impl ConvertFrom<u64> for u64 {
    fn cfrom(x: u64) -> Self {
        x
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [usize])]
impl ConvertFrom<u128> for dtype {
    fn cfrom(x: u128) -> Self {
        x as Self
    }
}

impl ConvertFrom<u128> for u128 {
    fn cfrom(x: u128) -> Self {
        x
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128];)]
impl ConvertFrom<usize> for dtype {
    fn cfrom(x: usize) -> Self {
        x as Self
    }
}

impl ConvertFrom<usize> for usize {
    fn cfrom(x: usize) -> Self {
        x
    }
}
