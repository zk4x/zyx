/// Tiny vector for shapes, padding, axes, ...
/// with nice functions for easy of handling
pub union TVec<T: Copy> {
    //array: [T; 8],
    ptr: *mut T,
    cap: u8,
    len: u8,
}

impl<T: Copy> TVec<T> {
}

struct Arena {
    ptr: *mut u8,
}
