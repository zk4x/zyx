use std::{collections::BTreeSet, mem::MaybeUninit, ops::{Index, IndexMut}};

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
        self.values.iter().enumerate().filter(|(id, _)| !self.empty.contains(id)).map(|(_, x)| unsafe { x.assume_init_ref() })
    }

    pub(crate) fn iter<'a>(&'a self) -> impl Iterator<Item = (Id, &'a T)> {
        self.values.iter().enumerate().filter(|(id, _)| !self.empty.contains(id)).map(|(id, x)| (id, unsafe { x.assume_init_ref() }))
    }

    pub(crate) fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (Id, &'a mut T)> {
        self.values.iter_mut().enumerate().filter(|(id, _)| !self.empty.contains(id)).map(|(id, x)| (id, unsafe { x.assume_init_mut() }))
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
        IndexMap {
            values,
            empty,
        }
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
                return x.partial_cmp(&y)
            } else {
                return Some(self.values.len().cmp(&other.values.len()))
            }
        }
    }
}

impl<T: Ord> Ord for IndexMap<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let mut iter = self.iter().zip(other.iter());
        loop {
            if let Some((x, y)) = iter.next() {
                return x.cmp(&y)
            } else {
                return self.values.len().cmp(&other.values.len())
            }
        }
    }
}

impl<T: Clone> Clone for IndexMap<T> {
    fn clone(&self) -> Self {
        Self {
            values: self.values.iter().enumerate().map(|(id, x)| if self.empty.contains(&id) { MaybeUninit::uninit() } else { MaybeUninit::new(unsafe { x.assume_init_ref() }.clone()) }).collect(),
            empty: self.empty.clone(),
        }
    }
}
