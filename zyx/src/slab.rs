//! Index map using unsinged integer indices to index
//! into vector of T. Pushing new values returns their
//! index. Removing elements is O(1), does not reallocate
//! and it does not change existing indices.

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

    pub(crate) fn remove(&mut self, id: Id) -> Option<T> {
        if Id::try_from(self.values.len()).unwrap() > id && !self.empty.contains(&id) {
            self.empty.insert(id);
            self.values.push(MaybeUninit::uninit());
            Some(unsafe { self.values.swap_remove(id as usize).assume_init() })
        } else {
            None
        }
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

    pub(crate) fn len(&self) -> usize {
        self.values.len() - self.empty.len()
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
