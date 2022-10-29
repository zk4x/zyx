use super::ConvertFrom;
use duplicate::duplicate_item;

#[duplicate_item( dtype; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<f32> for dtype {
    fn cfrom(x: f32) -> Self {
        x as Self
    }
}

#[duplicate_item( dtype; [f32]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<f64> for dtype {
    fn cfrom(x: f64) -> Self {
        x as Self
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<i8> for dtype {
    fn cfrom(x: i8) -> Self {
        x as Self
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<i16> for dtype {
    fn cfrom(x: i16) -> Self {
        x as Self
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<i32> for dtype {
    fn cfrom(x: i32) -> Self {
        x as Self
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<i64> for dtype {
    fn cfrom(x: i64) -> Self {
        x as Self
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<i128> for dtype {
    fn cfrom(x: i128) -> Self {
        x as Self
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<isize> for dtype {
    fn cfrom(x: isize) -> Self {
        x as Self
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u16]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<u8> for dtype {
    fn cfrom(x: u8) -> Self {
        x as Self
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u32]; [u64]; [u128]; [usize])]
impl ConvertFrom<u16> for dtype {
    fn cfrom(x: u16) -> Self {
        x as Self
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u64]; [u128]; [usize])]
impl ConvertFrom<u32> for dtype {
    fn cfrom(x: u32) -> Self {
        x as Self
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u128]; [usize])]
impl ConvertFrom<u64> for dtype {
    fn cfrom(x: u64) -> Self {
        x as Self
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [usize])]
impl ConvertFrom<u128> for dtype {
    fn cfrom(x: u128) -> Self {
        x as Self
    }
}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128];)]
impl ConvertFrom<usize> for dtype {
    fn cfrom(x: usize) -> Self {
        x as Self
    }
}
