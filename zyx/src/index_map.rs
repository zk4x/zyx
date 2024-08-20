use std::ops::{Index, IndexMut};

type Id = usize;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub(crate) struct IndexMap<T> {
    values: Vec<T>,
    empty: Vec<Id>,
}

impl<T> IndexMap<T> {
    pub(crate) const fn new() -> IndexMap<T> {
        IndexMap {
            values: Vec::new(),
            empty: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, value: T) -> Id {
        if let Some(id) = self.empty.pop() {
            self.values[id] = value;
            id
        } else {
            self.values.push(value);
            self.values.len() - 1
        }
    }

    pub(crate) fn remove(&mut self, id: Id) -> Option<T> {
        if self.values.len() > id && !self.empty.contains(&id) {
            self.empty.push(id);
            self.values.push(unsafe { std::mem::zeroed() });
            Some(self.values.swap_remove(id))
        } else {
            None
        }
    }

    /// Returns true if newly inserted, false if replaced old value
    pub(crate) fn insert(&mut self, id: Id, value: T) -> bool
    where
        T: Default,
    {
        if let Some(v) = self.values.get_mut(id) {
            self.empty.retain(|x| *x != id);
            *v = value;
            false
        } else {
            while self.values.len() <= id {
                self.values.push(T::default());
            }
            self.values[id] = value;
            true
        }
    }

    /*pub(crate) fn contains_id(&self, id: Id) -> bool {
        if self.values.len() > id && !self.empty.contains(&id) {
            true
        } else {
            false
        }
    }*/

    pub(crate) fn ids(&self) -> impl Iterator<Item = Id> + '_ {
        (0..self.values.len()).skip_while(|x| self.empty.contains(x))
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &T> {
        self.values.iter()
    }

    pub(crate) fn iter<'a>(&'a self) -> impl Iterator<Item = (Id, &'a T)> {
        self.values.iter().enumerate().skip_while(|(x, _)| self.empty.contains(x)).collect::<Vec<(Id, &'a T)>>().into_iter()
    }

    pub(crate) fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (Id, &'a mut T)> {
        self.values.iter_mut().enumerate().skip_while(|(x, _)| self.empty.contains(x)).collect::<Vec<(Id, &'a mut T)>>().into_iter()
    }
}

impl<T> Index<Id> for IndexMap<T> {
    type Output = T;
    fn index(&self, index: Id) -> &Self::Output {
        &self.values[index]
    }
}

impl<T> IndexMut<Id> for IndexMap<T> {
    fn index_mut(&mut self, index: Id) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl<'a, T> IntoIterator for &'a IndexMap<T> {
    type Item = (Id, &'a T);
    type IntoIter = std::vec::IntoIter<(Id, &'a T)>;
    fn into_iter(self) -> Self::IntoIter {
        // TODO make this faster, probably using custom iterator struct
        self.values.iter().enumerate().skip_while(|(x, _)| self.empty.contains(x)).collect::<Vec<(Id, &'a T)>>().into_iter()
    }
}

impl<'a, T> IntoIterator for &'a mut IndexMap<T> {
    type Item = (Id, &'a mut T);
    type IntoIter = std::vec::IntoIter<(Id, &'a mut T)>;
    fn into_iter(self) -> Self::IntoIter {
        // TODO make this faster, probably using custom iterator struct
        self.values.iter_mut().enumerate().skip_while(|(x, _)| self.empty.contains(x)).collect::<Vec<(Id, &'a mut T)>>().into_iter()
    }
}

impl<T: Default> FromIterator<(Id, T)> for IndexMap<T> {
    fn from_iter<I: IntoIterator<Item = (Id, T)>>(iter: I) -> IndexMap<T> {
        let mut values = Vec::new();
        let mut empty = Vec::new();
        let mut i = 0;
        for (id, v) in iter {
            while id != i {
                values.push(T::default());
                empty.push(i);
                i += 1;
            }
            values.push(v);
            i += 1;
        }
        IndexMap {
            values,
            empty,
        }
    }
}