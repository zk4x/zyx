//! Shape for multidimensional data structures
//! 
//! Stores dimension sizes for multidimensional data structures.
//! Shape is a trait that is implemented for some basic data types,
//! such as tuples and arrays.
//! 

use crate::ops::Permute;

pub trait Shape<'a> {
    fn dims(self) -> &'a [usize];
    fn index(self, idx: i32) -> usize;
    fn strides(self) -> Vec<usize>;
    fn argsort(self) -> Vec<i32>;
    fn numel(self) -> usize;
    fn ndim(self) -> usize;
    //fn is_empty(&self) -> bool { self.numel() == 0 }
}

impl<'a> Shape<'a> for &'a [usize] {
    fn dims(self) -> &'a [usize] {
        self
    }

    fn index(self, idx: i32) -> usize {
        let n = self.len();
        self[(n as i32 + idx) as usize % n]
    }

    fn strides(self) -> Vec<usize> {
        let mut product = 1;
        let mut res = vec![0; self.len()];
        for (i, dim) in self.iter().enumerate().rev() {
            res[i] = product;
            product *= dim;
        }
        res
    }

    fn argsort(self) -> Vec<i32> {
        let mut indices: Vec<i32> = (0..self.len() as i32).collect();
        indices.sort_by_key(|&i| &self[i as usize]);
        indices
    }

    fn numel(self) -> usize {
        self.iter().product()
    }

    fn ndim(self) -> usize {
        self.len()
    }
}

impl Permute for &[usize] {
    type Output = Vec<usize>;
    fn permute(self, dims: &[i32]) -> Self::Output {
        let mut res = self.to_vec();
        for (i, dim) in dims.iter().enumerate() {
            res[i] = self.index(*dim);
        }
        res
    }
}

pub trait Dims<'a> {
    fn dims(self) -> &'a [i32];
    fn index(self, idx: i32) -> i32;
    fn strides(self) -> Vec<i32>;
    fn argsort(self) -> Vec<i32>;
    fn numel(self) -> usize;
    fn ndim(self) -> usize;
    //fn is_empty(&self) -> bool { self.numel() == 0 }
}

impl<'a> Dims<'a> for &'a [i32] {
    fn dims(self) -> &'a [i32] {
        self
    }

    fn index(self, idx: i32) -> i32 {
        let n = self.len();
        self[(n as i32 + idx) as usize % n]
    }

    fn strides(self) -> Vec<i32> {
        let mut product = 1;
        let mut res = vec![0; self.len()];
        for (i, dim) in self.iter().enumerate().rev() {
            res[i] = product;
            product *= dim;
        }
        res
    }

    fn argsort(self) -> Vec<i32> {
        let mut indices: Vec<i32> = (0..self.len() as i32).collect();
        indices.sort_by_key(|&i| &self[i as usize]);
        indices
    }

    fn numel(self) -> usize {
        self.iter().product::<i32>() as usize
    }

    fn ndim(self) -> usize {
        self.len()
    }
}
