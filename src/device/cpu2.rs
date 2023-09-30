extern crate alloc;
use alloc::boxed::Box;

use crate::shape::{Strides, Shape};

struct CpuStorage<T> {
    data: Box<[T]>,
    view: View,
}

struct View {
    strides0: Strides,
    shape1: Shape,
    strides1: Strides,
    shape2: Shape,
    strides2: Strides,
    contiguous: bool,
}

impl View {
    fn get_idx(&self, idx: usize) -> usize {
        if self.contiguous {
            return idx
        }
        let mut res = 0;
        for i in 0..self.shape2.rank() {
            res += idx/self.strides2[i]%self.shape2[i]*self.strides1[i];
        }
        for i in 0..self.shape1.rank() {
            res += idx/self.strides1[i]%self.shape1[i]*self.strides0[i];
        }
        res
    }
}
