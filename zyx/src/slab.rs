//! Index map using unsinged integer indices to index
//! into vector of T. Pushing new values returns their
//! index. Removing elements is O(1), does not reallocate
//! and it does not change existing indices.

use std::{
    collections::BTreeSet,
    mem::MaybeUninit,
    ops::{Index, IndexMut},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Id(u32);

impl Id {
    const fn index(self) -> usize {
        self.0 as usize
    }

    const fn from_usize(id: usize) -> Self {
        Id(id as u32)
    }

    const fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug)]
pub struct Slab<T> {
    values: Vec<MaybeUninit<T>>,
    empty: BTreeSet<Id>,
}

struct IdIter<'a> {
    id: Id,
    max: Id,
    empty: &'a BTreeSet<Id>,
}

impl<'a> IdIter<'a> {
    fn new(empty: &'a BTreeSet<Id>, max: Id) -> Self {
        Self {
            id: Id(0),
            max,
            empty,
        }
    }
}

impl Iterator for IdIter<'_> {
    type Item = Id;

    fn next(&mut self) -> Option<Self::Item> {
        // TODO make it fater later like this
        /*let mut id = 0;
        for x in &self.empty {
            while x.0 as usize > id {
                unsafe { self.values[id].assume_init_drop() };
                id += 1;
            }
            id += 1;
        }
        while id < self.values.len() {
            unsafe { self.values[id].assume_init_drop() };
            id += 1;
        }*/

        if self.id >= self.max {
            return None;
        }
        while self.empty.contains(&self.id) {
            self.id.inc();
        }
        let id = self.id;
        self.id.inc();
        Some(id)
    }
}

impl<T> Drop for Slab<T> {
    fn drop(&mut self) {
        // Drops those that are not in self.empty
        for id in IdIter::new(&self.empty, Id::from_usize(self.values.len())) {
            unsafe { self.values[id.index()].assume_init_drop() };
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
            self.values[id.index()] = MaybeUninit::new(value);
            //println!("Pushing to empty {id}");
            id
        } else {
            self.values.push(MaybeUninit::new(value));
            //println!("Pushing {}, empty: {:?}", self.values.len() - 1, self.empty);
            Id::from_usize(self.values.len() - 1)
        }
    }

    pub(crate) fn remove(&mut self, id: Id) {
        debug_assert!(!self.empty.contains(&id));
        self.empty.insert(id);
        unsafe { self.values[id.index()].assume_init_drop() };
    }

    pub(crate) unsafe fn remove_and_return(&mut self, id: Id) -> T {
        debug_assert!(!self.empty.contains(&id));
        self.empty.insert(id);
        self.values.push(MaybeUninit::uninit());
        unsafe { self.values.swap_remove(id.index()).assume_init() }
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
        IdIter::new(&self.empty, Id::from_usize(self.values.len()))
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &T> {
        IdIter::new(&self.empty, Id::from_usize(self.values.len()))
            .map(|id| unsafe { self.values[id.index()].assume_init_ref() })
    }

    pub(crate) fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.values
            .iter_mut()
            .enumerate()
            .filter(|(id, _)| !self.empty.contains(&(Id::from_usize(*id))))
            .map(|(_, x)| unsafe { x.assume_init_mut() })
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (Id, &T)> {
        self.values
            .iter()
            .enumerate()
            .filter(|(id, _)| !self.empty.contains(&(Id::from_usize(*id))))
            .map(|(id, x)| (Id::from_usize(id), unsafe { x.assume_init_ref() }))
    }

    pub(crate) fn contains_key(&self, id: Id) -> bool {
        id < Id::from_usize(self.values.len()) && !self.empty.contains(&id)
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
        Id::from_usize(self.values.len())
    }

    pub(crate) fn len(&self) -> u32 {
        u32::try_from(self.values.len() - self.empty.len()).unwrap()
    }
}

impl<T> Index<Id> for Slab<T> {
    type Output = T;
    fn index(&self, index: Id) -> &Self::Output {
        debug_assert!(!self.empty.contains(&index));
        unsafe { self.values[index.index()].assume_init_ref() }
    }
}

impl<T> IndexMut<Id> for Slab<T> {
    fn index_mut(&mut self, index: Id) -> &mut Self::Output {
        debug_assert!(!self.empty.contains(&index));
        unsafe { self.values[index.index()].assume_init_mut() }
    }
}

impl<T> FromIterator<(Id, T)> for Slab<T> {
    fn from_iter<I: IntoIterator<Item = (Id, T)>>(iter: I) -> Self {
        let mut values = Vec::new();
        let mut empty = BTreeSet::new();
        let mut i = Id(0);
        for (id, v) in iter {
            while id != i {
                values.push(MaybeUninit::uninit());
                empty.insert(i);
                i.inc();
            }
            values.push(MaybeUninit::new(v));
            i.inc();
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
                    if self.empty.contains(&(Id::from_usize(id))) {
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
