//! Index map using unsinged integer indices to index
//! into vector of T. Pushing new values returns their
//! index. Removing elements is O(1), does not reallocate
//! and it does not change existing indices.

/*use std::{
    collections::BTreeMap,
    ops::{Index, IndexMut},
};

type Id = usize;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct IndexMap<T>(BTreeMap<Id, T>);

impl<T> IndexMap<T> {
    pub(crate) const fn new() -> IndexMap<T> {
        IndexMap(BTreeMap::new())
    }

    pub(crate) fn ids(&self) -> impl Iterator<Item = Id> + '_ {
        self.0.keys().copied()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (Id, &T)> {
        self.0.iter().map(|(x, y)| (*x, y))
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (Id, &mut T)> {
        self.0.iter_mut().map(|(x, y)| (*x, y))
    }

    pub(crate) fn push(&mut self, value: T) -> Id {
        if self.0.is_empty() {
            self.0.insert(0, value);
            return 0;
        }
        if self.0.len() < *self.0.last_key_value().unwrap().0 {
            for id in 0..self.0.len() {
                if !self.0.contains_key(&id) {
                    self.0.insert(id, value);
                    return id;
                }
            }
            return 0;
        } else {
            let id = self.0.len();
            self.0.insert(id, value);
            return id;
        }
    }

    pub(crate) fn remove(&mut self, id: Id) -> Option<T> {
        self.0.remove(&id)
    }

    pub(crate) fn swap(&mut self, x: Id, y: Id) {
        let value1 = self.0.remove(&x).unwrap();
        let value2 = self.0.remove(&y).unwrap();
        self.0.insert(x, value2);
        self.0.insert(y, value1);
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &T> {
        self.0.values()
    }
}

impl<T> Index<Id> for IndexMap<T> {
    type Output = T;

    fn index(&self, index: Id) -> &Self::Output {
        &self.0[&index]
    }
}

impl<T> IndexMut<Id> for IndexMap<T> {
    fn index_mut(&mut self, index: Id) -> &mut Self::Output {
        self.0.get_mut(&index).unwrap()
    }
}

impl<T> FromIterator<(Id, T)> for IndexMap<T> {
    fn from_iter<I: IntoIterator<Item = (Id, T)>>(iter: I) -> IndexMap<T> {
        IndexMap(iter.into_iter().collect())
    }
}*/

use std::{
    collections::BTreeSet,
    mem::MaybeUninit,
    ops::{Index, IndexMut},
};

type Id = usize;

#[derive(Debug)]
pub(crate) struct IndexMap<T> {
    values: Vec<MaybeUninit<T>>,
    empty: BTreeSet<Id>,
}

impl<T> Drop for IndexMap<T> {
    fn drop(&mut self) {
        for (id, x) in self.values.iter_mut().enumerate() {
            if !self.empty.contains(&id) {
                unsafe { x.assume_init_drop() };
            }
        }
    }
}

impl<T> IndexMap<T> {
    pub(crate) const fn new() -> IndexMap<T> {
        IndexMap {
            values: Vec::new(),
            empty: BTreeSet::new(),
        }
    }

    pub(crate) fn push(&mut self, value: T) -> Id {
        if let Some(id) = self.empty.pop_first() {
            self.values[id] = MaybeUninit::new(value);
            //println!("Pushing to empty {id}");
            id
        } else {
            self.values.push(MaybeUninit::new(value));
            //println!("Pushing {}, empty: {:?}", self.values.len() - 1, self.empty);
            self.values.len() - 1
        }
    }

    pub(crate) fn remove(&mut self, id: Id) -> Option<T> {
        if self.values.len() > id && !self.empty.contains(&id) {
            self.empty.insert(id);
            self.values.push(MaybeUninit::uninit());
            Some(unsafe { self.values.swap_remove(id).assume_init() })
        } else {
            None
        }
    }

    pub(crate) fn swap(&mut self, x: Id, y: Id) {
        self.values.swap(x, y);
    }

    pub(crate) fn ids(&self) -> impl Iterator<Item = Id> + '_ {
        (0..self.values.len()).filter(|x| !self.empty.contains(x))
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &T> {
        self.values
            .iter()
            .enumerate()
            .filter(|(id, _)| !self.empty.contains(id))
            .map(|(_, x)| unsafe { x.assume_init_ref() })
    }

    pub(crate) fn iter<'a>(&'a self) -> impl Iterator<Item = (Id, &'a T)> {
        self.values
            .iter()
            .enumerate()
            .filter(|(id, _)| !self.empty.contains(id))
            .map(|(id, x)| (id, unsafe { x.assume_init_ref() }))
    }

    pub(crate) fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (Id, &'a mut T)> {
        self.values
            .iter_mut()
            .enumerate()
            .filter(|(id, _)| !self.empty.contains(id))
            .map(|(id, x)| (id, unsafe { x.assume_init_mut() }))
    }

    pub(crate) fn len(&self) -> usize {
        self.values.len() - self.empty.len()
    }
}

impl<T> Index<Id> for IndexMap<T> {
    type Output = T;
    fn index(&self, index: Id) -> &Self::Output {
        assert!(!self.empty.contains(&index));
        unsafe { self.values[index].assume_init_ref() }
    }
}

impl<T> IndexMut<Id> for IndexMap<T> {
    fn index_mut(&mut self, index: Id) -> &mut Self::Output {
        assert!(!self.empty.contains(&index));
        unsafe { self.values[index].assume_init_mut() }
    }
}

impl<T> FromIterator<(Id, T)> for IndexMap<T> {
    fn from_iter<I: IntoIterator<Item = (Id, T)>>(iter: I) -> IndexMap<T> {
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
        IndexMap { values, empty }
    }
}

impl<T: PartialEq> PartialEq for IndexMap<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(x, y)| x == y)
    }
}

impl<T: Eq> Eq for IndexMap<T> {}

impl<T: PartialOrd> PartialOrd for IndexMap<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let mut iter = self.iter().zip(other.iter());
        loop {
            if let Some((x, y)) = iter.next() {
                return x.partial_cmp(&y);
            } else {
                return Some(self.values.len().cmp(&other.values.len()));
            }
        }
    }
}

impl<T: Ord> Ord for IndexMap<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let mut iter = self.iter().zip(other.iter());
        loop {
            if let Some((x, y)) = iter.next() {
                return x.cmp(&y);
            } else {
                return self.values.len().cmp(&other.values.len());
            }
        }
    }
}

impl<T: Clone> Clone for IndexMap<T> {
    fn clone(&self) -> Self {
        Self {
            values: self
                .values
                .iter()
                .enumerate()
                .map(|(id, x)| {
                    if self.empty.contains(&id) {
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
