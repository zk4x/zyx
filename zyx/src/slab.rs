//! Index map using unsinged integer indices to index
//! into vector of T. Pushing new values returns their
//! index. Removing elements is O(1), does not reallocate
//! and it does not change existing indices.

/*use std::{
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
            if self.cap != 0 {
                for idx in 0..self.cap {
                    if self.has_element_at(idx) {
                        ptr::drop_in_place(self.get_unchecked_mut(idx));
                    }
                }
                /*for bit_idx in 0..num_usizes_for(self.len) {
                    *self.empty.as_ptr().add(bit_idx) = 0;
                }*/
                //self.len = 0;
                if size_of::<T>() != 0 {
                    dealloc(self.values.as_ptr() as *mut _, self.old_elem_layout());
                }
                dealloc(self.empty.as_ptr() as *mut _, self.old_bit_layout());
                //self.cap = 0;
            }
        }
    }
}

impl<T> Slab<T> {
    fn has_element_at(&self, id: Id) -> bool {
        debug_assert!(id < self.cap);
        let usize_pos = id / BITS_PER_USIZE;
        let bit_pos = id % BITS_PER_USIZE;
        let block = unsafe { *self.empty.as_ptr().add(usize_pos as usize) };
        ((block >> bit_pos) & 0b1) != 0
    }

    fn get_unchecked(&self, id: Id) -> &T {
        debug_assert!(id < self.cap, "idx is {id}, but ap is only {}", self.cap);
        debug_assert!(self.has_element_at(id));
        unsafe { &*self.values.as_ptr().add(id as usize) }
    }

    fn get_unchecked_mut(&mut self, id: Id) -> &mut T {
        debug_assert!(id < self.cap);
        debug_assert!(self.has_element_at(id));
        unsafe { &mut *self.values.as_ptr().add(id as usize) }
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
        debug_assert!(new_cap > self.cap);
        #[inline(never)]
        #[cold]
        fn capacity_overflow() -> ! {
            panic!(
                "capacity overflow in `stable_vec::BitVecCore::realloc` (attempt \
                to allocate more than `u32::MAX` bytes"
            );
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
            let ptr = if self.cap == 0 {
                alloc(new_elem_layout)
            } else {
                realloc(self.values.as_ptr() as *mut _, self.old_elem_layout(), size)
            };
            if ptr.is_null() {
                handle_alloc_error(new_elem_layout);
            }
            self.values = NonNull::new_unchecked(ptr as *mut _);
        };
        // ----- (Re)allocate bitvec memory ----------------------------------
        {
            let size = size_of::<usize>() * num_usizes_for(new_cap);
            let new_bit_layout = Layout::from_size_align_unchecked(size, align_of::<usize>());
            let ptr = if self.cap == 0 {
                alloc_zeroed(new_bit_layout)
            } else {
                realloc(self.empty.as_ptr() as *mut _, self.old_bit_layout(), size)
            };
            let ptr = ptr as *mut usize;
            if ptr.is_null() {
                handle_alloc_error(new_bit_layout);
            }
            // If we reallocated, the new memory is not necessarily zeroed, so
            // we need to do it. TODO: if `alloc` offers a `realloc_zeroed`
            // in the future, we should use that.
            /*if self.cap != 0 {
                let initialized_usizes = num_usizes_for(self.cap);
                let new_usizes = num_usizes_for(new_cap);
                if new_usizes > initialized_usizes {
                    ptr::write_bytes(
                        ptr.add(initialized_usizes),
                        0,
                        new_usizes - initialized_usizes,
                    );
                }
            }*/
            self.empty = NonNull::new_unchecked(ptr as *mut _);
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
        'a: while i < self.cap {
            let x = unsafe { *ptr };
            if x < usize::MAX {
                for j in 0..BITS_PER_USIZE {
                    let found = ((x >> j) & 0b1) == 0;
                    if found {
                        unsafe {
                            *ptr |= 1 << j;
                            *self.values.as_ptr().add(i as usize) = value;
                        }
                        self.len += 1;
                        return i;
                    }
                    i += 1;
                    if i > self.cap {
                        break 'a;
                    }
                }
            }
            i += BITS_PER_USIZE;
            ptr = unsafe { ptr.add(1) };
        }
        if self.len == self.cap {
            if self.cap == 0 {
                unsafe { self.realloc(32) };
            } else {
                unsafe { self.realloc(self.cap * 2) };
            }
        }
        self.len += 1;
        let mask = 1 << (i % BITS_PER_USIZE);
        //println!("{i}, cap {}, mask {mask}", self.cap);
        unsafe { *self.empty.as_ptr().add((i / BITS_PER_USIZE) as usize) |= mask };
        unsafe { *self.values.as_ptr().add(i as usize) = value };
        return i;
    }

    pub(crate) fn remove(&mut self, id: Id) {
        debug_assert!(id < self.cap);
        debug_assert!(self.has_element_at(id));
        // We first mark the value as deleted and then read the value.
        // Otherwise, a random panic could lead to a double drop.
        let usize_pos = id / BITS_PER_USIZE;
        let bit_pos = id % BITS_PER_USIZE;
        let mask = !(1 << bit_pos);
        unsafe { *self.empty.as_ptr().add(usize_pos as usize) &= mask };
        self.len -= 1;
    }

    pub(crate) unsafe fn remove_and_use(&mut self, id: Id) -> T {
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
        (0..self.cap).filter(|&id| self.has_element_at(id))
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &T> {
        (0..self.cap).filter(|&id| self.has_element_at(id)).map(|id| self.get_unchecked(id))
    }

    pub(crate) fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        (0..self.cap)
            .filter(|&id| self.has_element_at(id))
            .map(|id| unsafe { &mut *self.values.as_ptr().add(id as usize) })
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (Id, &T)> {
        (0..self.cap).filter(|&id| self.has_element_at(id)).map(|id| (id, self.get_unchecked(id)))
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

#[test]
fn slab_t1() {
    let mut s1 = Slab::new();
    let x = s1.push(17);
    s1.remove(x);
    let x = s1.push(24);
    println!("{}", s1[x]);
    let y = s1.remove(x);
    println!("{:?}", y);
    let _ = s1.push(29);

    for _ in 0..100 {
        let _ = s1.push(21);
    }

    let y = unsafe { s1.remove_and_use(x) };
    println!("{}", y);
}*/

use std::{
    collections::BTreeSet,
    mem::MaybeUninit,
    ops::{Index, IndexMut},
};

pub type Id = u32;

#[derive(Debug)]
pub struct Slab<T> {
    values: Vec<MaybeUninit<T>>,
    empty: BTreeSet<Id>,
}

impl<T> Drop for Slab<T> {
    fn drop(&mut self) {
        let mut id = 0;
        #[allow(clippy::explicit_counter_loop)]
        for x in self.values.iter_mut() {
            if !self.empty.contains(&id) {
                unsafe { x.assume_init_drop() };
            }
            id += 1;
        }
    }
}

impl<T> Slab<T> {
    pub(crate) const fn new() -> Self {
        Self { values: Vec::new(), empty: BTreeSet::new() }
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self { values: Vec::with_capacity(capacity), empty: BTreeSet::new() }
    }

    pub(crate) fn push(&mut self, value: T) -> Id {
        if let Some(id) = self.empty.pop_first() {
            self.values[id as usize] = MaybeUninit::new(value);
            //println!("Pushing to empty {id}");
            id
        } else {
            self.values.push(MaybeUninit::new(value));
            //println!("Pushing {}, empty: {:?}", self.values.len() - 1, self.empty);
            Id::try_from(self.values.len() - 1).unwrap()
        }
    }

    pub(crate) fn remove(&mut self, id: Id) {
        debug_assert!(!self.empty.contains(&id));
        self.empty.insert(id);
        unsafe { self.values[id as usize].assume_init_drop() };
    }

    pub(crate) unsafe fn remove_and_use(&mut self, id: Id) -> T {
        debug_assert!(!self.empty.contains(&id));
        self.empty.insert(id);
        self.values.push(MaybeUninit::uninit());
        unsafe { self.values.swap_remove(id as usize).assume_init() }
    }

    /*pub(crate) fn get(&self, id: Id) -> Option<&T> {
        if Id::try_from(self.values.len()).unwrap() > id && !self.empty.contains(&id) {
            Some(unsafe { self.values[id as usize].assume_init_ref() })
        } else {
            None
        }
    }*/

    /*pub(crate) fn swap(&mut self, x: Id, y: Id) {
        self.values.swap(x as usize, y as usize);
    }*/

    pub(crate) fn ids(&self) -> impl Iterator<Item = Id> + '_ {
        (0..Id::try_from(self.values.len()).unwrap()).filter(|x| !self.empty.contains(x))
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &T> {
        self.values
            .iter()
            .enumerate()
            .filter(|(id, _)| !self.empty.contains(&(Id::try_from(*id).unwrap())))
            .map(|(_, x)| unsafe { x.assume_init_ref() })
    }

    pub(crate) fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.values
            .iter_mut()
            .enumerate()
            .filter(|(id, _)| !self.empty.contains(&(Id::try_from(*id).unwrap())))
            .map(|(_, x)| unsafe { x.assume_init_mut() })
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (Id, &T)> {
        self.values
            .iter()
            .enumerate()
            .filter(|(id, _)| !self.empty.contains(&(Id::try_from(*id).unwrap())))
            .map(|(id, x)| (Id::try_from(id).unwrap(), unsafe { x.assume_init_ref() }))
    }

    /*pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (Id, &mut T)> {
        self.values
            .iter_mut()
            .enumerate()
            .filter(|(id, _)| !self.empty.contains(&(Id::try_from(*id).unwrap())))
            .map(|(id, x)| (Id::try_from(id).unwrap(), unsafe { x.assume_init_mut() }))
    }*/

    /*pub(crate) fn retain(&mut self, func: impl Fn(&Id) -> bool) -> Set<Id> {
        let mut deleted = Set::with_capacity_and_hasher(10, Default::default());
        let mut i = 0;
        for x in &mut self.values {
            if !func(&i) && !self.empty.contains(&i) {
                deleted.insert(i);
                unsafe { x.assume_init_drop() };
                self.empty.insert(i);
            }
            i += 1;
        }
        deleted
    }*/

    // TODO lower max id by searching for it in self.empty
    #[allow(unused)]
    pub(crate) fn max_id(&self) -> Id {
        self.values.len().try_into().unwrap()
    }

    pub(crate) fn len(&self) -> u32 {
        (self.values.len() - self.empty.len()) as u32
    }
}

impl<T> Index<Id> for Slab<T> {
    type Output = T;
    fn index(&self, index: Id) -> &Self::Output {
        debug_assert!(!self.empty.contains(&index));
        unsafe { self.values[index as usize].assume_init_ref() }
    }
}

impl<T> IndexMut<Id> for Slab<T> {
    fn index_mut(&mut self, index: Id) -> &mut Self::Output {
        debug_assert!(!self.empty.contains(&index));
        unsafe { self.values[index as usize].assume_init_mut() }
    }
}

impl<T> FromIterator<(Id, T)> for Slab<T> {
    fn from_iter<I: IntoIterator<Item = (Id, T)>>(iter: I) -> Self {
        let mut values = Vec::new();
        let mut empty = BTreeSet::new();
        let mut i = 0;
        for (id, v) in iter {
            while id != i {
                values.push(MaybeUninit::uninit());
                empty.insert(i);
                i += 1;
            }
            values.push(MaybeUninit::new(v));
            i += 1;
        }
        Self { values, empty }
    }
}

impl<T: PartialEq> PartialEq for Slab<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(x, y)| x == y)
    }
}

impl<T: Eq> Eq for Slab<T> {}

impl<T: PartialOrd> PartialOrd for Slab<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let mut iter = self.iter().zip(other.iter());
        // TODO perhaps we can do the comparison over more than one element if the first elements
        // are equal
        if let Some((x, y)) = iter.next() {
            x.partial_cmp(&y)
        } else {
            Some(self.values.len().cmp(&other.values.len()))
        }
    }
}

impl<T: Ord> Ord for Slab<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let mut iter = self.iter().zip(other.iter());
        if let Some((x, y)) = iter.next() {
            x.cmp(&y)
        } else {
            self.values.len().cmp(&other.values.len())
        }
    }
}

impl<T: Clone> Clone for Slab<T> {
    fn clone(&self) -> Self {
        Self {
            values: self
                .values
                .iter()
                .enumerate()
                .map(|(id, x)| {
                    if self.empty.contains(&(Id::try_from(id).unwrap())) {
                        MaybeUninit::uninit()
                    } else {
                        MaybeUninit::new(unsafe { x.assume_init_ref() }.clone())
                    }
                })
                .collect(),
            empty: self.empty.clone(),
        }
    }
}
