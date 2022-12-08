mod bin_op;
mod const_index;

pub use bin_op::BinOpShape;
pub use const_index::ConstIndex;

/// # Dim
///
/// Single dimension of a [Shape].
pub trait Dim {}
impl Dim for usize {}
impl Dim for i32 {}

/// # Shape
/// 
/// Stores size of dimensions of multidimensional data structures.
/// Shape can be tuple or array of usize or i32.
/// Shape of usize describes dimensions of multidimensional data structures.
/// For example, if we have matrix like following,
/// ```txt
/// [2 3 4
///  3 1 4]
/// ```
/// it's shape is 2x3, which can be written as (2, 3) tuple of usize or [2, 3] array of usize.
/// Whether you use tuple of array depends on your personal preference. Most people prefer
/// to use tuples, however these are implemented only up to 5 dimensions.
/// If you want to use more dimensions than that, you have to use arrays.
///
/// Scalars have shape (1usize) or [1usize];
///
/// Shape<i32> is used to describe the order of dimensions in Shape<usize>.
/// This is used for example when you want to permute tensor.
/// ```
/// # use zyx::prelude::*;
/// # use zyx::accel::cpu;
/// let x = cpu::Buffer::from([[[2, 3, 4], [2, 1, 3]], [][4, 1, 3], [3, 1, 2]]);
/// // Here we specify, that we want to exchange first (0) and last dimensions (-1).
/// let y = x.permute((0, -1))
/// ```
///
/// Shape<i32> is also used to specify along which dimensions we want to perform reduce operations.
/// ```
/// # use zyx::prelude::*;
/// # use zyx::accel::cpu;
/// # let x = cpu::Buffer::from([[[2, 3, 4], [2, 1, 3]], [][4, 1, 3], [3, 1, 2]]);
/// // Perform reduce operation using sum along last two dimensions.
/// let y = x.sum((-1, -2));
/// ```
///
/// Shape<usize> is used more often, mainly to initialize new tensors.
/// ## Example
/// To create a [Buffer](crate::accel::cpu::Buffer) with random values,
/// we need to pass it some shape, in this case 2x3.
/// ```
/// # use zyx::prelude::*;
/// # use zyx::accel::cpu;
/// let x = cpu::Buffer::<f32>::randn((2, 3));
/// ```
pub trait Shape: Clone + Copy {
    type D: Dim;
    const N: usize;
    fn strides(&self) -> Self;
    fn argsort(&self) -> Self;
    fn numel(&self) -> usize;
    fn is_empty(&self) -> bool { self.numel() == 0 }
    fn at(&self, idx: usize) -> Self::D;
    fn ati(&self, idx: i32) -> Self::D;
    fn mut_at(&mut self, idx: usize) -> &mut Self::D;
    fn mut_ati(&mut self, idx: i32) -> &mut Self::D;
}

impl Shape for () {
    type D = i32;
    const N: usize = 1;

    fn strides(&self) -> Self {
        ()
    }

    fn argsort(&self) -> Self {
        ()
    }

    fn numel(&self) -> usize {
        0
    }

    fn at(&self, idx: usize) -> Self::D {
        panic!("Index out of range, the index is {}, but the length is {}", idx, 0);
    }

    fn ati(&self, idx: i32) -> Self::D {
        panic!("Index out of range, the index is {}, but the length is {}", idx, 0);
    }

    fn mut_at(&mut self, idx: usize) -> &mut Self::D {
        panic!("Index out of range, the index is {}, but the length is {}", idx, 0);
    }

    fn mut_ati(&mut self, idx: i32) -> &mut Self::D {
        panic!("Index out of range, the index is {}, but the length is {}", idx, 0);
    }
}

impl crate::ops::Permute<Sh> for () {
}

impl Shape for usize {
    type D = usize;
    const N: usize = 1;

    fn strides(&self) -> Self {
        1
    }

    fn argsort(&self) -> Self {
        0
    }

    fn numel(&self) -> usize {
        *self
    }

    fn at(&self, idx: usize) -> Self::D {
        match idx {
            0 => *self,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }

    fn ati(&self, idx: i32) -> Self::D {
        match idx {
            0 => *self,
            -1 => *self,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }

    fn mut_at(&mut self, idx: usize) -> &mut Self::D {
        match idx {
            0 => self,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }

    fn mut_ati(&mut self, idx: i32) -> &mut Self::D {
        match idx {
            0 => self,
            -1 => self,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }
}

impl Shape for i32 {
    type D = i32;
    const N: usize = 1;

    fn strides(&self) -> Self {
        1
    }

    fn argsort(&self) -> Self {
        0
    }

    fn numel(&self) -> usize {
        *self as usize
    }

    fn at(&self, idx: usize) -> Self::D {
        match idx {
            0 => *self,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }

    fn ati(&self, idx: i32) -> Self::D {
        match idx {
            0 => *self,
            -1 => *self,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }

    fn mut_at(&mut self, idx: usize) -> &mut Self::D {
        match idx {
            0 => self,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }

    fn mut_ati(&mut self, idx: i32) -> &mut Self::D {
        match idx {
            0 => self,
            -1 => self,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }
}

impl Shape for (usize, usize) {
    type D = usize;
    const N: usize = 2;

    fn strides(&self) -> Self {
        (self.0, 1)
    }

    fn argsort(&self) -> Self {
        if self.0 > self.1 {
            (0, 1)
        } else {
            (1, 0)
        }
    }

    fn numel(&self) -> usize {
        self.0 * self.1
    }

    fn at(&self, idx: usize) -> Self::D {
        match idx {
            1 => self.1,
            0 => self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1)
        }
    }

    fn ati(&self, idx: i32) -> Self::D {
        match idx {
            1 => self.1,
            0 => self.0,
            -1 => self.1,
            -2 => self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1)
        }
    }

    fn mut_at(&mut self, idx: usize) -> &mut Self::D {
        match idx {
            1 => &mut self.1,
            0 => &mut self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }

    fn mut_ati(&mut self, idx: i32) -> &mut Self::D {
        match idx {
            1 => &mut self.1,
            0 => &mut self.0,
            -1 => &mut self.1,
            -2 => &mut self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }
}

impl Shape for (i32, i32) {
    type D = i32;
    const N: usize = 2;

    fn strides(&self) -> Self {
        (self.0, 1)
    }

    fn argsort(&self) -> Self {
        if self.0 > self.1 {
            (0, 1)
        } else {
            (1, 0)
        }
    }

    fn numel(&self) -> usize {
        (self.0 * self.1) as usize
    }

    fn at(&self, idx: usize) -> Self::D {
        match idx {
            1 => self.1,
            0 => self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }

    fn ati(&self, idx: i32) -> Self::D {
        match idx {
            1 => self.1,
            0 => self.0,
            -1 => self.1,
            -2 => self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }

    fn mut_at(&mut self, idx: usize) -> &mut Self::D {
        match idx {
            1 => &mut self.1,
            0 => &mut self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }

    fn mut_ati(&mut self, idx: i32) -> &mut Self::D {
        match idx {
            1 => &mut self.1,
            0 => &mut self.0,
            -1 => &mut self.1,
            -2 => &mut self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }
}

impl Shape for (usize, usize, usize) {
    type D = usize;
    const N: usize = 3;

    fn strides(&self) -> Self {
        (self.1*self.0, self.0, 1)
    }

    fn argsort(&self) -> Self {
        todo!()
    }

    fn numel(&self) -> usize {
        self.2 * self.1 * self.0
    }

    fn at(&self, idx: usize) -> Self::D {
        match idx {
            2 => self.2,
            1 => self.1,
            0 => self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1)
        }
    }

    fn ati(&self, idx: i32) -> Self::D {
        match idx {
            2 => self.2,
            1 => self.1,
            0 => self.0,
            -1 => self.2,
            -2 => self.1,
            -3 => self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1)
        }
    }

    fn mut_at(&mut self, idx: usize) -> &mut Self::D {
        match idx {
            2 => &mut self.2,
            1 => &mut self.1,
            0 => &mut self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }

    fn mut_ati(&mut self, idx: i32) -> &mut Self::D {
        match idx {
            2 => &mut self.2,
            1 => &mut self.1,
            0 => &mut self.0,
            -1 => &mut self.2,
            -2 => &mut self.1,
            -3 => &mut self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }
}


impl Shape for (i32, i32, i32) {
    type D = i32;
    const N: usize = 3;

    fn strides(&self) -> Self {
        (self.1*self.0, self.0, 1)
    }

    fn argsort(&self) -> Self {
        todo!()
    }

    fn numel(&self) -> usize {
        (self.2 * self.1 * self.0) as usize
    }

    fn at(&self, idx: usize) -> Self::D {
        match idx {
            2 => self.2,
            1 => self.1,
            0 => self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1)
        }
    }

    fn ati(&self, idx: i32) -> Self::D {
        match idx {
            2 => self.2,
            1 => self.1,
            0 => self.0,
            -1 => self.2,
            -2 => self.1,
            -3 => self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1)
        }
    }

    fn mut_at(&mut self, idx: usize) -> &mut Self::D {
        match idx {
            2 => &mut self.2,
            1 => &mut self.1,
            0 => &mut self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }

    fn mut_ati(&mut self, idx: i32) -> &mut Self::D {
        match idx {
            2 => &mut self.2,
            1 => &mut self.1,
            0 => &mut self.0,
            -1 => &mut self.2,
            -2 => &mut self.1,
            -3 => &mut self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }
}

impl Shape for (usize, usize, usize, usize) {
    type D = usize;
    const N: usize = 4;

    fn strides(&self) -> Self {
        (self.2 * self.1 * self.0, self.1*self.0, self.0, 1)
    }

    fn argsort(&self) -> Self {
        todo!()
    }

    fn numel(&self) -> usize {
        self.3 * self.2 * self.1 * self.0
    }

    fn at(&self, idx: usize) -> Self::D {
        match idx {
            3 => self.3,
            2 => self.2,
            1 => self.1,
            0 => self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1)
        }
    }

    fn ati(&self, idx: i32) -> Self::D {
        match idx {
            3 => self.3,
            2 => self.2,
            1 => self.1,
            0 => self.0,
            -1 => self.3,
            -2 => self.2,
            -3 => self.1,
            -4 => self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1)
        }
    }

    fn mut_at(&mut self, idx: usize) -> &mut Self::D {
        match idx {
            3 => &mut self.3,
            2 => &mut self.2,
            1 => &mut self.1,
            0 => &mut self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }

    fn mut_ati(&mut self, idx: i32) -> &mut Self::D {
        match idx {
            3 => &mut self.3,
            2 => &mut self.2,
            1 => &mut self.1,
            0 => &mut self.0,
            -1 => &mut self.3,
            -2 => &mut self.2,
            -3 => &mut self.1,
            -4 => &mut self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }
}

impl Shape for (i32, i32, i32, i32) {
    type D = i32;
    const N: usize = 4;

    fn strides(&self) -> Self {
        (self.2 * self.1 * self.0, self.1*self.0, self.0, 1)
    }

    fn argsort(&self) -> Self {
        todo!()
    }

    fn numel(&self) -> usize {
        (self.3 * self.2 * self.1 * self.0) as usize
    }

    fn at(&self, idx: usize) -> Self::D {
        match idx {
            3 => self.3,
            2 => self.2,
            1 => self.1,
            0 => self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1)
        }
    }

    fn ati(&self, idx: i32) -> Self::D {
        match idx {
            3 => self.3,
            2 => self.2,
            1 => self.1,
            0 => self.0,
            -1 => self.3,
            -2 => self.2,
            -3 => self.1,
            -4 => self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1)
        }
    }

    fn mut_at(&mut self, idx: usize) -> &mut Self::D {
        match idx {
            3 => &mut self.3,
            2 => &mut self.2,
            1 => &mut self.1,
            0 => &mut self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }

    fn mut_ati(&mut self, idx: i32) -> &mut Self::D {
        match idx {
            3 => &mut self.3,
            2 => &mut self.2,
            1 => &mut self.1,
            0 => &mut self.0,
            -1 => &mut self.3,
            -2 => &mut self.2,
            -3 => &mut self.1,
            -4 => &mut self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }
}

impl Shape for (usize, usize, usize, usize, usize) {
    type D = usize;
    const N: usize = 5;

    fn strides(&self) -> Self {
        (self.3 * self.2 * self.1 * self.0, self.2 * self.1 * self.0, self.1*self.0, self.0, 1)
    }

    fn argsort(&self) -> Self {
        todo!()
    }

    fn numel(&self) -> usize {
        self.4 * self.3 * self.2 * self.1 * self.0
    }

    fn at(&self, idx: usize) -> Self::D {
        match idx {
            4 => self.4,
            3 => self.3,
            2 => self.2,
            1 => self.1,
            0 => self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1)
        }
    }

    fn ati(&self, idx: i32) -> Self::D {
        match idx {
            4 => self.4,
            3 => self.3,
            2 => self.2,
            1 => self.1,
            0 => self.0,
            -1 => self.4,
            -2 => self.3,
            -3 => self.2,
            -4 => self.1,
            -5 => self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1)
        }
    }

    fn mut_at(&mut self, idx: usize) -> &mut Self::D {
        match idx {
            4 => &mut self.4,
            3 => &mut self.3,
            2 => &mut self.2,
            1 => &mut self.1,
            0 => &mut self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }

    fn mut_ati(&mut self, idx: i32) -> &mut Self::D {
        match idx {
            4 => &mut self.4,
            3 => &mut self.3,
            2 => &mut self.2,
            1 => &mut self.1,
            0 => &mut self.0,
            -1 => &mut self.4,
            -2 => &mut self.3,
            -3 => &mut self.2,
            -4 => &mut self.1,
            -5 => &mut self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }
}

impl Shape for (i32, i32, i32, i32, i32) {
    type D = i32;
    const N: usize = 5;

    fn strides(&self) -> Self {
        (self.3*self.2*self.1*self.0, self.2*self.1*self.0, self.1*self.0, self.0, 1)
    }

    fn argsort(&self) -> Self {
        todo!()
    }

    fn numel(&self) -> usize {
        (self.4 * self.3 * self.2 * self.1 * self.0) as usize
    }

    fn at(&self, idx: usize) -> Self::D {
        match idx {
            4 => self.4,
            3 => self.3,
            2 => self.2,
            1 => self.1,
            0 => self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1)
        }
    }

    fn ati(&self, idx: i32) -> Self::D {
        match idx {
            4 => self.4,
            3 => self.3,
            2 => self.2,
            1 => self.1,
            0 => self.0,
            -1 => self.4,
            -2 => self.3,
            -3 => self.2,
            -4 => self.1,
            -5 => self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1)
        }
    }

    fn mut_at(&mut self, idx: usize) -> &mut Self::D {
        match idx {
            4 => &mut self.4,
            3 => &mut self.3,
            2 => &mut self.2,
            1 => &mut self.1,
            0 => &mut self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }

    fn mut_ati(&mut self, idx: i32) -> &mut Self::D {
        match idx {
            4 => &mut self.4,
            3 => &mut self.3,
            2 => &mut self.2,
            1 => &mut self.1,
            0 => &mut self.0,
            -1 => &mut self.4,
            -2 => &mut self.3,
            -3 => &mut self.2,
            -4 => &mut self.1,
            -5 => &mut self.0,
            _ => panic!("Index out of range, the index is {}, but the length is {}", idx, 1),
        }
    }
}

impl<const N: usize> Shape for [usize; N] {
    type D = usize;
    const N: usize = N;

    fn strides(&self) -> Self {
        let mut product = 1;
        let mut res = [0; N];
        self.clone().into_iter().enumerate().rev().for_each(
            |(i, dim)| {
                res[i] = product;
                product *= dim;
            });
        res
    }

    fn argsort(&self) -> Self {
        let mut indices: [usize; N] = core::array::from_fn(|i| i);
        indices.sort_by_key(|&i| &self[i]);
        indices
    }

    fn numel(&self) -> usize {
        self.iter().product()
    }

    fn at(&self, idx: usize) -> Self::D {
        self[idx]
    }

    fn ati(&self, idx: i32) -> Self::D {
        self[(N as i32 + idx) as usize % N]
    }

    fn mut_at(&mut self, idx: usize) -> &mut Self::D {
        &mut self[idx]
    }

    fn mut_ati(&mut self, idx: i32) -> &mut Self::D {
        &mut self[(N as i32 + idx) as usize % N]
    }
}

impl<const N: usize> Shape for [i32; N] {
    type D = i32;
    const N: usize = N;

    fn strides(&self) -> Self {
        let mut product = 1;
        let mut res = [0; N];
        self.clone().into_iter().enumerate().rev().for_each(
            |(i, dim)| {
                res[i] = product;
                product *= dim;
            });
        res
    }

    fn argsort(&self) -> Self {
        let mut indices: [i32; N] = core::array::from_fn(|i| i as i32);
        indices.sort_by_key(|&i| &self[i as usize]);
        indices
    }

    fn numel(&self) -> usize {
        self.iter().product::<i32>() as usize
    }

    fn at(&self, idx: usize) -> Self::D {
        self[idx]
    }

    fn ati(&self, idx: i32) -> Self::D {
        self[(N as i32 + idx) as usize % N]
    }

    fn mut_at(&mut self, idx: usize) -> &mut Self::D {
        &mut self[idx]
    }

    fn mut_ati(&mut self, idx: i32) -> &mut Self::D {
        &mut self[(N as i32 + idx) as usize % N]
    }
}
