use alloc::vec::Vec;
use alloc::collections::BTreeMap;
use alloc::string::String;
use crate::axes::Axes;
use crate::shape::Shape;

#[derive(Debug, Clone, Copy)]
struct Dimension {
    size: usize,
    stride: usize,
    // if left_mask < size, its indexing, else it's padding
    // left_mask goes in reverse direction, from biggest to smallest value
    left_mask: usize,
    // if right_mask < size, its indexing, else it's padding
    right_mask: usize,
}

/// View represents movement ops applied on tensors
#[derive(Debug)]
pub struct View {
    shapes: Vec<Vec<Dimension>>,
    binds: Vec<usize>,
}

impl View {
    /// Create new View from given shape
    #[must_use]
    pub fn from(shape: &Shape) -> Self {
        let mut stride = 1;
        let mut first_shape: Vec<Dimension> = shape.iter().rev().map(|size| {
            let temp = Dimension { size: *size, stride, left_mask: 0, right_mask: 0 };
            stride *= size;
            temp
        }).collect();
        first_shape.reverse();
        Self {
            shapes: alloc::vec![first_shape],
            binds: alloc::vec![0; first_shape.len()],
        }
    }

    #[must_use]
    fn numel(&self) -> usize {
        self.shapes[0].iter().map(|Dimension { size, .. }| size).product()
    }

    #[must_use]
    fn is_last_shape_contiguous(&self) -> bool {
        let mut stride = 1;
        let mut temp = true;
        for dim in self.shapes[0].iter().rev() {
            if stride != dim.stride {
                temp = false;
                break;
            }
            stride *= dim.size;
        }
        temp
    }

    /// Reshape view into different shape
    pub fn reshape(&mut self, sh: &Shape) {
        assert_eq!(self.numel(), sh.numel());
        // TODO perhaps we can merge more shapes with the previous shape

        // TODO merge split and join dimension reshapes

        // Merge unsqueeze
        if sh.rank() > self.shapes[0].len() {
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
            let mut new_binds = Vec::new();
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
                    }
                }
            }
            if merge_possible {
                self.shapes[0] = new_shape;
                return
            }
        }

        let mut new_stride = 1;
        let mut new_shape: Vec<Dimension> = sh.iter().copied().rev().map(|size| {
            let stride = new_stride;
            new_stride *= size;
            Dimension {
                size,
                stride,
                left_mask: 0,
                right_mask: 0
            }
        }).collect();
        new_shape.reverse();
        if self.is_last_shape_contiguous() {
            // Merge contiguous shape
            self.shapes[0] = new_shape;
        } else {
            self.shapes.insert(0, new_shape);
        }
    }

    /// Pad view with padding
    pub fn pad(&mut self, padding: &[(usize, usize)]) {
        debug_assert_eq!(self.shapes[0].len(), padding.len());
        for (dim, (lp, rp)) in self.shapes[0].iter_mut().zip(padding) {
            dim.left_mask = *lp;
            dim.right_mask = *rp;
        }
    }

    /// Expand view into different shape
    pub fn expand(&mut self, sh: &Shape) {
        debug_assert_eq!(self.shapes[0].len(), sh.rank());
        for (dim, sh_dim) in self.shapes[0].iter_mut().zip(sh) {
            if dim.size != *sh_dim {
                dim.size = *sh_dim;
                dim.stride = 0;
            }
        }
    }

    /// Permute view with axes
    pub fn permute(&mut self, axes: &Axes) {
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

/// Virtual representation of index into
pub enum Index {
    /// Expanded and/or padded
    /// Pairs of index id and multiplier.
    /// Can use wide loads directly with pointer casts.
    Contiguous {
        dims: BTreeMap<usize, usize>,
        padding_condition: String,
    },
    /// Expanded and/or permuted
    /// Pairs of index id and multiplier.
    /// Wide loads are possible only if we can transpose it in the kernel
    Strided {
        dims: BTreeMap<usize, usize>,
        padding_condition: String,
    },
    /// Expanded, permuted and/or reshaped
    Reshaped {
        padding_condition: String,
    },
    /// Expanded, permuted, reshaped and/or padded
    ReshapedAndPadded {
        padding_condition: String,
    },
}
