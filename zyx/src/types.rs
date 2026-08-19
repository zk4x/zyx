#![forbid(unsafe_op_in_unsafe_fn)]
use std::{
    alloc::{Layout, alloc, dealloc, handle_alloc_error},
    fmt,
    marker::PhantomData,
    mem::{align_of, size_of},
    ptr::{self, NonNull},
    slice, str,
};

use nanoserde::{DeBin, DeBinErr, DeBinErrReason, SerBin};

// ============================================================
// TinyString
//
// Representation:
//     8-byte owning pointer
//
// Allocation:
//     [ length: u8 ][ UTF-8 bytes ... ]
//
// Maximum length: 255 bytes
// ============================================================

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TinyString {
    ptr: NonNull<u8>,
}

// SAFETY:
// TinyString owns a dedicated heap allocation whose bytes are immutable.
// Moving the owner across threads transfers the allocation with it, like
// moving a Box<str>. Sharing a TinyString only yields a &str into immutable
// bytes. Therefore TinyString is unconditionally Send and Sync.
unsafe impl Send for TinyString {}
unsafe impl Sync for TinyString {}

impl TinyString {
    pub const MAX_LEN: usize = 255;

    pub fn new(value: &str) -> Self {
        assert!(value.len() <= Self::MAX_LEN);

        let allocation_size = 1 + value.len();

        unsafe {
            let ptr = allocate(allocation_size, 1);

            // SAFETY:
            // The allocation contains at least allocation_size bytes.
            ptr.as_ptr().write(value.len() as u8);

            // SAFETY:
            // Destination contains value.len() writable bytes.
            // Source is a valid UTF-8 string of exactly value.len() bytes.
            // Source and destination are distinct allocations.
            ptr::copy_nonoverlapping(value.as_ptr(), ptr.as_ptr().add(1), value.len());

            Self { ptr }
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        unsafe {
            // SAFETY:
            // Every TinyString allocation has its length in byte zero.
            self.ptr.as_ptr().read() as usize
        }
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        unsafe {
            // SAFETY:
            // The bytes were originally copied from a valid &str and
            // TinyString has no mutation API that can change them.
            str::from_utf8_unchecked(slice::from_raw_parts(self.ptr.as_ptr().add(1), self.len()))
        }
    }
}

impl Drop for TinyString {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align(1 + self.len(), 1).expect("valid TinyString layout");

            // SAFETY:
            // self.ptr owns this allocation and it was allocated with
            // exactly this size and alignment.
            dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

impl AsRef<str> for TinyString {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Debug for TinyString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("TinyString").field(&self.as_str()).finish()
    }
}

impl Clone for TinyString {
    fn clone(&self) -> Self {
        Self::new(self.as_str())
    }
}

impl SerBin for TinyString {
    fn ser_bin(&self, output: &mut Vec<u8>) {
        self.len().ser_bin(output);
        output.extend_from_slice(self.as_str().as_bytes());
    }
}

impl DeBin for TinyString {
    fn de_bin(offset: &mut usize, bytes: &[u8]) -> Result<Self, DeBinErr> {
        let len = usize::de_bin(offset, bytes)?;
        if len > Self::MAX_LEN {
            return Err(DeBinErr {
                o: *offset,
                msg: DeBinErrReason::Length { expected_length: Self::MAX_LEN, actual_length: len },
            });
        }
        let end = match offset.checked_add(len) {
            Some(end) if end <= bytes.len() => end,
            _ => {
                return Err(DeBinErr {
                    o: *offset,
                    msg: DeBinErrReason::Length { expected_length: len, actual_length: bytes.len() },
                });
            }
        };
        let text = match str::from_utf8(&bytes[*offset..end]) {
            Ok(text) => text,
            Err(_) => {
                return Err(DeBinErr {
                    o: *offset,
                    msg: DeBinErrReason::Length { expected_length: len, actual_length: bytes.len() },
                });
            }
        };
        *offset = end;
        Ok(Self::new(text))
    }
}

// ============================================================
// TinyVec<T>
//
// T must be Copy.
//
// Representation:
//     8-byte owning pointer
//
// Allocation:
//     [ length: u8 ][ padding ][ T ][ T ] ...
//
// The padding makes the T array properly aligned.
//
// Maximum length: 255 elements.
//
// The API is immutable after construction except for mutation
// through iter_mut(). Use iter_mut() for in-place updates.
// ============================================================

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TinyVec<T: Copy> {
    ptr: NonNull<u8>,
    _marker: PhantomData<T>,
}

// SAFETY:
// TinyVec owns a heap allocation of T: Copy values. Moving the owner across
// threads transfers the allocation pointer with it, like moving Box<[T]>.
// Shared access yields &T into fully initialized data. Hence TinyVec<T> is
// Send if T: Send and Sync if T: Sync.
unsafe impl<T: Copy + Send> Send for TinyVec<T> {}
unsafe impl<T: Copy + Sync> Sync for TinyVec<T> {}

impl<T: Copy> TinyVec<T> {
    pub const MAX_LEN: usize = 255;

    pub fn new(values: &[T]) -> Self {
        assert!(values.len() <= Self::MAX_LEN);

        // This implementation intentionally does not support ZSTs.
        assert!(size_of::<T>() != 0);

        let offset = data_offset::<T>();
        let element_bytes = values.len().checked_mul(size_of::<T>()).expect("TinyVec allocation size overflow");

        let allocation_size = offset.checked_add(element_bytes).expect("TinyVec allocation size overflow");

        let alignment = align_of::<T>();

        unsafe {
            let ptr = allocate(allocation_size, alignment);

            // SAFETY:
            // Allocation contains at least one byte because offset >= 1.
            ptr.as_ptr().write(values.len() as u8);

            if !values.is_empty() {
                let destination = ptr.as_ptr().add(offset).cast::<T>();

                // SAFETY:
                // destination is correctly aligned for T.
                // allocation has room for values.len() T values.
                // T: Copy, so copying does not duplicate ownership.
                ptr::copy_nonoverlapping(values.as_ptr(), destination, values.len());
            }

            Self { ptr, _marker: PhantomData }
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        unsafe {
            // SAFETY:
            // Byte zero always contains the initialized length.
            self.ptr.as_ptr().read() as usize
        }
    }

    #[inline]
    fn data_ptr(&self) -> *const T {
        unsafe {
            // SAFETY:
            // data_offset<T>() guarantees proper alignment for T.
            self.ptr.as_ptr().add(data_offset::<T>()).cast::<T>()
        }
    }

    #[inline]
    fn data_ptr_mut(&mut self) -> *mut T {
        unsafe {
            // SAFETY:
            // data_offset<T>() guarantees proper alignment for T.
            self.ptr.as_ptr().add(data_offset::<T>()).cast::<T>()
        }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            return None;
        }

        unsafe {
            // SAFETY:
            // index < len, therefore this points to an initialized T
            // inside the allocation.
            Some(&*self.data_ptr().add(index))
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len() {
            return None;
        }

        unsafe {
            // SAFETY:
            // index < len, therefore this points to an initialized T
            // inside the allocation.
            Some(&mut *self.data_ptr_mut().add(index))
        }
    }

    pub fn iter(&self) -> slice::Iter<'_, T> {
        unsafe {
            // SAFETY:
            // data_ptr is properly aligned and points to len initialized
            // T values.
            slice::from_raw_parts(self.data_ptr(), self.len()).iter()
        }
    }

    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        unsafe {
            // SAFETY:
            // data_ptr_mut is properly aligned and points to len initialized
            // T values.
            slice::from_raw_parts_mut(self.data_ptr_mut(), self.len()).iter_mut()
        }
    }
}

impl<T: Copy> std::ops::Index<usize> for TinyVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("TinyVec index out of bounds")
    }
}

impl<T: Copy> std::ops::IndexMut<usize> for TinyVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("TinyVec index out of bounds")
    }
}

impl<T: Copy> Drop for TinyVec<T> {
    fn drop(&mut self) {
        unsafe {
            let len = self.len();
            let offset = data_offset::<T>();

            let element_bytes = len.checked_mul(size_of::<T>()).expect("TinyVec allocation size overflow");

            let allocation_size = offset.checked_add(element_bytes).expect("TinyVec allocation size overflow");

            let layout = Layout::from_size_align(allocation_size, align_of::<T>()).expect("valid TinyVec layout");

            // SAFETY:
            // self.ptr owns this allocation and the reconstructed layout
            // exactly matches the allocation performed by new().
            dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

impl<T: Copy> Clone for TinyVec<T> {
    fn clone(&self) -> Self {
        unsafe {
            // SAFETY:
            // data_ptr() is properly aligned for T and points to exactly
            // self.len() initialized T values owned by self.
            //
            // The resulting slice is only borrowed for the duration of
            // this call. TinyVec::new() copies those values into a new,
            // independent allocation.
            let values = slice::from_raw_parts(self.data_ptr(), self.len());

            Self::new(values)
        }
    }
}

impl<T: Copy + SerBin> SerBin for TinyVec<T> {
    fn ser_bin(&self, output: &mut Vec<u8>) {
        self.len().ser_bin(output);
        for item in self.iter() {
            item.ser_bin(output);
        }
    }
}

impl<T: Copy + DeBin> DeBin for TinyVec<T> {
    fn de_bin(offset: &mut usize, bytes: &[u8]) -> Result<Self, DeBinErr> {
        let len = usize::de_bin(offset, bytes)?;
        if len > Self::MAX_LEN {
            return Err(DeBinErr {
                o: *offset,
                msg: DeBinErrReason::Length { expected_length: Self::MAX_LEN, actual_length: len },
            });
        }
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(T::de_bin(offset, bytes)?);
        }
        Ok(Self::new(&values))
    }
}

// ============================================================
// Layout helpers
// ============================================================

#[inline]
const fn data_offset<T>() -> usize {
    let alignment = align_of::<T>();

    // Smallest multiple of alignment >= 1.
    (1 + alignment - 1) & !(alignment - 1)
}

unsafe fn allocate(size: usize, alignment: usize) -> NonNull<u8> {
    let layout = Layout::from_size_align(size, alignment).expect("invalid allocation layout");

    let raw = unsafe {
        // SAFETY:
        // layout was constructed by Layout.
        alloc(layout)
    };

    match NonNull::new(raw) {
        Some(ptr) => ptr,
        None => handle_alloc_error(layout),
    }
}

// ============================================================
// Compile-time representation checks
// ============================================================

const _: () = {
    assert!(size_of::<TinyString>() == 8);
    assert!(size_of::<TinyVec<u32>>() == 8);
};
