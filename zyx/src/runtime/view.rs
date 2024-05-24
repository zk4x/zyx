use alloc::vec::Vec;
use alloc::vec;
use alloc::collections::BTreeMap;
use alloc::string::String;

#[derive(Debug, Clone, Copy)]
struct Dimension {
    // size of the dimension in shape
    size: usize,
    stride: usize,
    // len is the actual length of this buffer without padding
    len: usize,
    // shift of size and len
    // shift=0 => no element from actual buffer is taken, it's all just padding
    // buffer with shape of 10, 2 elements, padding of (-2, 10)
    // size 10:    xxxxxxxxxx
    // len 2:    xx
    // shift 0
    // buffer with shape of 8, 3 elements, padding of (4, 1)
    // size 8:  xxxxxxxx
    // len 3:       xxx
    // shift 7
    // shift is essentially the position of the last element of the buffer
    // with respect to first element of the shape
    shift: usize,
}

/// View represents movement ops applied on tensors
#[derive(Debug, Clone)]
pub struct View {
    shapes: Vec<Vec<Dimension>>,
    // TODO perhaps binds for each shape in shapes? But probably not.
    binds: Vec<usize>,
}

impl View {
    /// Create new View from given shape
    #[must_use]
    pub fn from(shape: &[usize]) -> Self {
        let mut stride = 1;
        let mut first_shape: Vec<Dimension> = shape.iter().rev().map(|size| {
            let temp = Dimension { size: *size, stride, len: *size, shift: *size };
            stride *= size;
            temp
        }).collect();
        first_shape.reverse();
        Self {
            binds: alloc::vec![0; first_shape.len()],
            shapes: alloc::vec![first_shape],
        }
    }

    /// Shape
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        self.shapes[0].iter().map(|dim| dim.size).collect()
    }

    /// Rank
    #[must_use]
    pub fn rank(&self) -> usize {
        self.shapes[0].len()
    }

    /// Numel
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shapes[0].iter().map(|dim| dim.size).product()
    }

    #[must_use]
    fn is_last_shape_contiguous(&self) -> bool {
        let mut stride = 1;
        let mut temp = true;
        for dim in self.shapes[0].iter().rev() {
            if stride != dim.stride || dim.size != dim.len || dim.size != dim.shift {
                temp = false;
                break;
            }
            stride *= dim.size;
        }
        temp
    }

    /// Reshape view into different shape
    pub fn reshape(&mut self, shape: &[usize]) {
        assert_eq!(self.numel(), shape.iter().product());
        if self.shape() == *shape { return }
        // TODO perhaps we can merge more shapes with the previous shape

        // TODO merge split and join dimension reshapes

        // Merge unsqueeze
        /*if sh.rank() > self.shapes[0].len() {
            let mut merge_possible = true;
            let mut new_shape = Vec::new();
            let mut new_binds = Vec::new();
            let mut i = self.shapes[0].len();
            for sh_dim in sh.iter().rev() {
                if *sh_dim != 1 {
                    if *sh_dim != self.shapes[0][i].size {
                        merge_possible = false;
                        break;
                    } else {
                        new_binds.insert(0, self.binds[i]);
                        new_shape.insert(0, self.shapes[0][i]);
                        i -= 1;
                    }
                } else {
                    if *sh_dim != self.shapes[0][i].size {
                        new_binds.insert(0, 0);
                        new_shape.insert(0, Dimension {
                            size: 1,
                            stride: self.shapes[0][i].stride,
                            left_mask: 0,
                            right_mask: 0,
                        });
                    } else {
                        i -= 1;
                    }
                }
            }
            if merge_possible {
                self.shapes[0] = new_shape;
                return
            }
        }

        // Merge squeeze
        if sh.rank() < self.shapes[0].len() {
            let mut merge_possible = true;
            let mut new_shape = Vec::new();
            let mut new_binds = self.binds.clone();
            let mut i = sh.rank();
            for dim in self.shapes[0].iter().rev() {
                if dim.size != 1 {
                    if dim.size != sh[i] {
                        merge_possible = false;
                        break;
                    } else {
                        new_shape.insert(0, *dim);
                        i -= 1;
                    }
                } else {
                    if dim.size == sh[i] {
                        new_shape.insert(0, *dim);
                        i -= 1;
                    } else {
                        new_binds.remove(i);
                    }
                }
            }
            if merge_possible {
                self.shapes[0] = new_shape;
                return
            }
        }*/

        let mut new_stride = 1;
        let mut new_shape: Vec<Dimension> = shape.iter().copied().rev().map(|size| {
            let stride = new_stride;
            new_stride *= size;
            Dimension {
                size,
                stride,
                len: size,
                shift: size
            }
        }).collect();
        new_shape.reverse();
        // TODO fix binds
        let binds = vec![0; new_shape.len()];
        if self.is_last_shape_contiguous() {
            // Merge contiguous shape
            self.shapes[0] = new_shape;
        } else {
            self.shapes.insert(0, new_shape);
        }
    }

    /// Pad view with padding.
    /// This function assumes standard padding beginning at last dimension.
    pub fn pad(&mut self, padding: &[isize]) {
        use itertools::Itertools;
        debug_assert!(self.shapes[0].len() >= padding.len());
        for (dim, (lp, rp)) in self.shapes[0].iter_mut().rev().zip(padding.into_iter().tuples()) {
            dim.size = (<usize as TryInto<isize>>::try_into(dim.size).unwrap() + lp + rp).try_into().unwrap();
            dim.shift = (<usize as TryInto<isize>>::try_into(dim.shift).unwrap() + lp) as usize;
        }
    }

    /// Expand view into different shape
    pub fn expand(&mut self, shape: &[usize]) {
        debug_assert_eq!(self.shapes[0].len(), shape.iter().product());
        for (dim, sh_dim) in self.shapes[0].iter_mut().zip(shape) {
            if dim.size != *sh_dim {
                debug_assert_eq!(dim.size, 1);
                dim.size = *sh_dim;
                dim.stride = 0;
            }
        }
    }

    /// Permute view with axes
    pub fn permute(&mut self, axes: &[usize]) {
        debug_assert_eq!(self.shapes[0].len(), axes.len());
        self.shapes[0] = axes.iter().map(|axis| self.shapes[0][*axis]).collect();
    }

    /// Bind index id to dimension, this is used for index names when generating indexes
    /// for buffers in kernels (cidx function). Overwrites previous binds if any.
    pub fn bind(&mut self, index: usize, dim: usize) {
        self.binds[dim] = index;
    }

    /// Get index from this view, using bound indices, this function panics if some dimensions
    /// don't have bound indices.
    pub fn cidx(&self) -> Index {
        todo!()
    }
}

// With this representation of index, we can find repeating
// multipliers and extract them out.

/// Virtual representation of index into
pub enum Index {
    /// Expanded and/or padded
    /// Pairs of index id and multiplier.
    /// Can use wide loads directly with pointer casts.
    Contiguous {
        /// Dimension and multiplier
        dims: BTreeMap<usize, usize>,
        /// When should the padding get applied?
        padding_condition: String,
    },
    /// Expanded and/or permuted
    /// Pairs of index id and multiplier.
    /// Wide loads are possible only if we can transpose it in the kernel
    Strided {
        /// Dimension and multiplier
        dims: BTreeMap<usize, usize>,
        /// When should the padding get applied?
        padding_condition: String,
    },
    /// Expanded, permuted, reshaped and/or padded
    /// Only if reshape could not be merged.
    Reshaped {
        /// Multiple dimension and multipliers
        dims: Vec<BTreeMap<usize, usize>>,
        /// When should the padding get applied?
        padding_condition: String,
    },
}
