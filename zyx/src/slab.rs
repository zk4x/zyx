//! Index map using unsinged integer indices to index
//! into vector of T. Pushing new values returns their
//! index. Removing elements is O(1), does not reallocate
//! and it does not change existing indices.

use std::{
    collections::BTreeSet,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Index, IndexMut},
};

pub trait SlabId:
    std::fmt::Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + From<usize> + Into<usize>
{
    const ZERO: Self;
    fn inc(&mut self);
}

#[derive(Debug)]
pub struct Slab<Id: SlabId, T> {
    values: Vec<MaybeUninit<T>>,
    empty: BTreeSet<Id>,
    _index: PhantomData<Id>,
}

struct IdIter<'a, Id> {
    id: Id,
    max_exclusive: Id,
    empty: &'a BTreeSet<Id>,
}

impl<'a, Id: SlabId> IdIter<'a, Id> {
    const fn new(empty: &'a BTreeSet<Id>, max_exclusive: Id) -> Self {
        Self { id: Id::ZERO, max_exclusive, empty }
    }
}

impl<Id: SlabId> Iterator for IdIter<'_, Id> {
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

        let mut id;
        loop {
            id = self.id;
            self.id.inc();
            if !self.empty.contains(&id) {
                break;
            }
        }
        if id >= self.max_exclusive {
            None
        } else {
            Some(id)
        }
    }
}

impl<Id: SlabId, T> Drop for Slab<Id, T> {
    fn drop(&mut self) {
        // Drops those that are not in self.empty
        for id in IdIter::new(&self.empty, Id::from(self.values.len())) {
            unsafe { self.values[id.into()].assume_init_drop() };
        }
    }
}

impl<Id: SlabId, T> Slab<Id, T> {
    pub(crate) const fn new() -> Self {
        Self { values: Vec::new(), empty: BTreeSet::new(), _index: PhantomData }
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self { values: Vec::with_capacity(capacity), empty: BTreeSet::new(), _index: PhantomData }
    }

    pub(crate) fn push(&mut self, value: T) -> Id {
        if let Some(id) = self.empty.pop_first() {
            self.values[id.into()] = MaybeUninit::new(value);
            //println!("Pushing to empty {id}");
            id
        } else {
            self.values.push(MaybeUninit::new(value));
            //println!("Pushing {}, empty: {:?}", self.values.len() - 1, self.empty);
            Id::from(self.values.len() - 1)
        }
    }

    pub(crate) fn remove(&mut self, id: Id) {
        debug_assert!(!self.empty.contains(&id));
        self.empty.insert(id);
        unsafe { self.values[id.into()].assume_init_drop() };
    }

    pub(crate) unsafe fn remove_and_return(&mut self, id: Id) -> T {
        debug_assert!(!self.empty.contains(&id));
        self.empty.insert(id);
        self.values.push(MaybeUninit::uninit());
        unsafe { self.values.swap_remove(id.into()).assume_init() }
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
        IdIter::new(&self.empty, Id::from(self.values.len()))
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &T> {
        IdIter::new(&self.empty, Id::from(self.values.len()))
            .map(|id| unsafe { self.values[id.into()].assume_init_ref() })
    }

    pub(crate) fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.values
            .iter_mut()
            .enumerate()
            .filter(|(id, _)| !self.empty.contains(&(Id::from(*id))))
            .map(|(_, x)| unsafe { x.assume_init_mut() })
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (Id, &T)> {
        self.values
            .iter()
            .enumerate()
            .filter(|(id, _)| !self.empty.contains(&(Id::from(*id))))
            .map(|(id, x)| (Id::from(id), unsafe { x.assume_init_ref() }))
    }

    pub(crate) fn contains_key(&self, id: Id) -> bool {
        id < Id::from(self.values.len()) && !self.empty.contains(&id)
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
        Id::from(self.values.len())
    }

    pub(crate) fn len(&self) -> Id {
        Id::from(self.values.len() - self.empty.len())
    }
}

impl<Id: SlabId, T> Index<Id> for Slab<Id, T> {
    type Output = T;
    fn index(&self, index: Id) -> &Self::Output {
        debug_assert!(!self.empty.contains(&index));
        unsafe { self.values[index.into()].assume_init_ref() }
    }
}

impl<Id: SlabId, T> IndexMut<Id> for Slab<Id, T> {
    fn index_mut(&mut self, index: Id) -> &mut Self::Output {
        debug_assert!(!self.empty.contains(&index));
        unsafe { self.values[index.into()].assume_init_mut() }
    }
}

impl<Id: SlabId, T> FromIterator<(Id, T)> for Slab<Id, T> {
    fn from_iter<I: IntoIterator<Item = (Id, T)>>(iter: I) -> Self {
        let mut values = Vec::new();
        let mut empty = BTreeSet::new();
        let mut i = Id::ZERO;
        for (id, v) in iter {
            while id != i {
                values.push(MaybeUninit::uninit());
                empty.insert(i);
                i.inc();
            }
            values.push(MaybeUninit::new(v));
            i.inc();
        }
        Self { values, empty, _index: PhantomData }
    }
}

impl<Id: SlabId, T: PartialEq> PartialEq for Slab<Id, T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(x, y)| x == y)
    }
}

impl<Id: SlabId, T: Eq> Eq for Slab<Id, T> {}

impl<Id: SlabId, T: PartialOrd> PartialOrd for Slab<Id, T> {
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

impl<Id: SlabId, T: Ord> Ord for Slab<Id, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let mut iter = self.iter().zip(other.iter());
        if let Some((x, y)) = iter.next() {
            x.cmp(&y)
        } else {
            self.values.len().cmp(&other.values.len())
        }
    }
}

impl<T: Clone, Id: SlabId> Clone for Slab<Id, T> {
    fn clone(&self) -> Self {
        Self {
            values: self
                .values
                .iter()
                .enumerate()
                .map(|(id, x)| {
                    if self.empty.contains(&(Id::from(id))) {
                        MaybeUninit::uninit()
                    } else {
                        MaybeUninit::new(unsafe { x.assume_init_ref() }.clone())
                    }
                })
                .collect(),
            empty: self.empty.clone(),
            _index: PhantomData,
        }
    }
}
