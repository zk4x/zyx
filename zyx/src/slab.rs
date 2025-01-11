//! Index map using unsinged integer indices to index
//! into vector of T. Pushing new values returns their
//! index. Removing elements is O(1), does not reallocate
//! and it does not change existing indices.

use std::{
    alloc::{alloc, alloc_zeroed, dealloc, handle_alloc_error, realloc, Layout},
    ops::{Index, IndexMut},
    ptr::{self, NonNull},
};

pub type Id = u32;

const BITS_PER_USIZE: Id = (size_of::<usize>() * 8) as Id;

#[derive(Debug)]
pub struct Slab<T> {
    values: NonNull<T>,
    empty: NonNull<usize>,
    cap: Id,
    len: Id,
}

unsafe impl<T: Send> Send for Slab<T> {}
unsafe impl<T: Sync> Sync for Slab<T> {}

#[inline(always)]
fn num_usizes_for(cap: Id) -> usize {
    ((cap + (BITS_PER_USIZE - 1)) / BITS_PER_USIZE) as usize
}

impl<T> Drop for Slab<T> {
    fn drop(&mut self) {
        unsafe {
            for idx in 0..self.len {
                if self.has_element_at(idx) {
                    ptr::drop_in_place(self.get_unchecked_mut(idx));
                }
            }
            for bit_idx in 0..num_usizes_for(self.len) {
                *self.empty.as_ptr().add(bit_idx) = 0;
            }
            self.len = 0;
            if self.cap != 0 {
                if size_of::<T>() != 0 {
                    dealloc(self.values.as_ptr() as *mut _, self.old_elem_layout());
                }
                dealloc(self.empty.as_ptr() as *mut _, self.old_bit_layout());
                self.cap = 0;
            }
        }
    }
}

impl<T> Slab<T> {
    fn has_element_at(&self, idx: Id) -> bool {
        debug_assert!(idx < self.cap);
        let usize_pos = idx / BITS_PER_USIZE;
        let bit_pos = idx % BITS_PER_USIZE;
        let block = unsafe { *self.empty.as_ptr().add(usize_pos as usize) };
        ((block >> bit_pos) & 0b1) != 0
    }

    fn get_unchecked(&self, idx: Id) -> &T {
        debug_assert!(idx < self.cap);
        debug_assert!(self.has_element_at(idx));
        unsafe { &*self.values.as_ptr().add(idx as usize) }
    }

    fn get_unchecked_mut(&mut self, idx: Id) -> &mut T {
        debug_assert!(idx < self.cap);
        debug_assert!(self.has_element_at(idx));
        unsafe { &mut *self.values.as_ptr().add(idx as usize) }
    }

    unsafe fn old_elem_layout(&self) -> Layout {
        Layout::from_size_align_unchecked(self.cap as usize * size_of::<T>(), align_of::<T>())
    }

    unsafe fn old_bit_layout(&self) -> Layout {
        Layout::from_size_align_unchecked(
            size_of::<usize>() * num_usizes_for(self.cap),
            align_of::<usize>(),
        )
    }

    #[inline(never)]
    #[cold]
    unsafe fn realloc(&mut self, new_cap: u32) {
        debug_assert!(new_cap >= self.len);
        debug_assert!(new_cap <= i32::max_value() as u32);
        #[inline(never)]
        #[cold]
        fn capacity_overflow() -> ! {
            panic!(
                "capacity overflow in `stable_vec::BitVecCore::realloc` (attempt \
                to allocate more than `usize::MAX` bytes"
            );
        }
        // Handle special case
        if new_cap == 0 {
            // Due to preconditions, we know that `self.len == 0` and that in
            // turn tells us that there aren't any filled slots. So we can just
            // deallocate the memory.
            if self.cap != 0 {
                if size_of::<T>() != 0 {
                    dealloc(self.values.as_ptr() as *mut _, self.old_elem_layout());
                }
                dealloc(self.empty.as_ptr() as *mut _, self.old_bit_layout());
                self.cap = 0;
            }
            return;
        }
        // ----- (Re)allocate element memory ---------------------------------

        // We only have to allocate if our size are not zero-sized. Else, we
        // just don't do anything.
        if size_of::<T>() != 0 {
            // Get the new number of bytes for the allocation and create the
            // memory layout.
            let size = (new_cap as usize)
                .checked_mul(size_of::<T>())
                .unwrap_or_else(|| capacity_overflow());
            let new_elem_layout = Layout::from_size_align_unchecked(size, align_of::<T>());

            // (Re)allocate memory.
            let ptr = if self.cap == 0 {
                alloc(new_elem_layout)
            } else {
                realloc(self.values.as_ptr() as *mut _, self.old_elem_layout(), size)
            };
            // If the element allocation failed, we quit the program with an
            // OOM error.
            if ptr.is_null() {
                handle_alloc_error(new_elem_layout);
            }
            // We already overwrite the pointer here. It is not read/changed
            // anywhere else in this function.
            self.values = NonNull::new_unchecked(ptr as *mut _);
        };
        // ----- (Re)allocate bitvec memory ----------------------------------
        {
            // Get the new number of required bytes for the allocation and
            // create the memory layout.
            let size = size_of::<usize>() * num_usizes_for(new_cap);
            let new_bit_layout = Layout::from_size_align_unchecked(size, align_of::<usize>());

            // (Re)allocate memory.
            let ptr = if self.cap == 0 {
                alloc_zeroed(new_bit_layout)
            } else {
                realloc(self.empty.as_ptr() as *mut _, self.old_bit_layout(), size)
            };
            let ptr = ptr as *mut usize;
            // If the element allocation failed, we quit the program with an
            // OOM error.
            if ptr.is_null() {
                handle_alloc_error(new_bit_layout);
            }
            // If we reallocated, the new memory is not necessarily zeroed, so
            // we need to do it. TODO: if `alloc` offers a `realloc_zeroed`
            // in the future, we should use that.
            if self.cap != 0 {
                let initialized_usizes = num_usizes_for(self.cap);
                let new_usizes = num_usizes_for(new_cap);
                if new_usizes > initialized_usizes {
                    ptr::write_bytes(
                        ptr.add(initialized_usizes),
                        0,
                        new_usizes - initialized_usizes,
                    );
                }
            }
            self.values = NonNull::new_unchecked(ptr as *mut _);
        }
        self.cap = new_cap;

        // All formal requirements are met now:
        //
        // **Invariants**:
        // - *slot data*: by using `realloc` if `self.cap != 0`, the slot data
        //   (including deleted-flag) was correctly copied.
        // - `self.len()`: indeed didn't change
        //
        // **Postconditons**:
        // - `self.cap() == new_cap`: trivially holds due to last line.
    }
}

impl<T> Slab<T> {
    pub(crate) const fn new() -> Self {
        Self { values: NonNull::dangling(), empty: NonNull::dangling(), cap: 0, len: 0 }
    }

    pub(crate) fn with_capacity(capacity: u32) -> Self {
        #[cfg(debug_assertions)]
        if capacity > i32::max_value() as u32 {
            panic!("Capacity too large.");
        }
        let mut res =
            Self { values: NonNull::dangling(), empty: NonNull::dangling(), cap: 0, len: 0 };
        unsafe {
            res.realloc(capacity);
        }
        res
    }

    pub(crate) fn len(&self) -> u32 {
        self.len
    }

    pub(crate) fn push(&mut self, value: T) -> Id {
        let mut ptr = self.empty.as_ptr();
        let mut i = 0;
        while i < self.cap {
            let x = unsafe { *ptr };
            if x < usize::MAX {
                for j in 0..BITS_PER_USIZE {
                    let found = ((x >> j) & 0b1) != 0;
                    if found {
                        unsafe {
                            *self.empty.as_ptr().add((i / BITS_PER_USIZE) as usize) |= 1 << j;
                            *self.values.as_ptr().add(i as usize) = value;
                        }
                        return i;
                    }
                    i += 1;
                }
            }
            i += BITS_PER_USIZE;
            ptr = unsafe { ptr.add(1) };
        }
        if self.len == self.cap {
            if self.cap == 0 {
                unsafe { self.realloc(32) };
                self.len += 32;
            } else {
                unsafe { self.realloc(self.cap * 2) };
                self.len *= 2;
            }
        }
        let mask = 1 << (i % BITS_PER_USIZE);
        println!("{i}, cap {}, mask {mask}", self.cap);
        unsafe { *self.empty.as_ptr().add((i / BITS_PER_USIZE) as usize) |= mask };
        panic!();
        unsafe { *self.values.as_ptr().add(i as usize) = value };
        return i;
    }

    pub(crate) fn remove(&mut self, id: Id) -> T {
        debug_assert!(id < self.cap);
        debug_assert!(self.has_element_at(id));
        // We first mark the value as deleted and then read the value.
        // Otherwise, a random panic could lead to a double drop.
        let usize_pos = id / BITS_PER_USIZE;
        let bit_pos = id % BITS_PER_USIZE;
        let mask = !(1 << bit_pos);
        unsafe { *self.empty.as_ptr().add(usize_pos as usize) &= mask };
        self.len -= 1;
        unsafe { ptr::read(self.values.as_ptr().add(id as usize)) }
    }

    pub(crate) fn ids(&self) -> impl Iterator<Item = Id> + '_ {
        todo!();
        //(0..Id::try_from(self.values.len()).unwrap()).filter(|x| !self.empty.contains(x))
        Vec::new().into_iter()
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &T> {
        todo!();
        /*self.values
        .iter()
        .enumerate()
        .filter(|(id, _)| !self.empty.contains(&(Id::try_from(*id).unwrap())))
        .map(|(_, x)| unsafe { x.assume_init_ref() })*/
        Vec::new().into_iter()
    }

    pub(crate) fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        todo!();
        /*self.values
        .iter_mut()
        .enumerate()
        .filter(|(id, _)| !self.empty.contains(&(Id::try_from(*id).unwrap())))
        .map(|(_, x)| unsafe { x.assume_init_mut() })*/
        Vec::new().into_iter()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (Id, &T)> {
        todo!();
        /*self.values
        .iter()
        .enumerate()
        .filter(|(id, _)| !self.empty.contains(&(Id::try_from(*id).unwrap())))
        .map(|(id, x)| (Id::try_from(id).unwrap(), unsafe { x.assume_init_ref() }))*/
        Vec::new().into_iter()
    }
}

impl<T> Index<Id> for Slab<T> {
    type Output = T;
    fn index(&self, index: Id) -> &Self::Output {
        //unsafe { self.values.as_ptr().add(index as usize).as_ref() }.unwrap()
        self.get_unchecked(index)
    }
}

impl<T> IndexMut<Id> for Slab<T> {
    fn index_mut(&mut self, index: Id) -> &mut Self::Output {
        self.get_unchecked_mut(index)
    }
}
