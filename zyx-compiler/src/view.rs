use alloc::vec::Vec;
use zyx_core::axes::Axes;
use zyx_core::shape::Shape;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Dimension {
    size: usize,
    stride: usize,
    // if left_mask < size, its indexing, else it's padding
    // left_mask goes in reverse direction, from biggest to smallest value
    left_mask: usize,
    // if right_mask < size, its indexing, else it's padding
    right_mask: usize,
}

#[derive(Debug)]
pub struct View(pub(crate) Vec<Vec<Dimension>>);

impl View {
    #[must_use]
    fn numel(&self) -> usize {
        self.0[0].iter().map(|Dimension { size, .. }| size).product()
    }

    #[must_use]
    fn is_last_shape_contiguous(&self) -> bool {
        let mut stride = 1;
        let mut temp = true;
        for dim in self.0[0].iter().rev() {
            if stride != dim.stride {
                temp = false;
                break;
            }
            stride *= dim.size;
        }
        temp
    }

    pub(crate) fn reshape(&mut self, sh: &Shape) {
        assert_eq!(self.numel(), sh.numel());
        // TODO perhaps we can merge more shapes with the previous shape

        // Merge unsqueeze
        if sh.rank() > self.0[0].len() {
            let mut merge_possible = true;
            let mut new_shape = Vec::new();
            let mut i = self.0[0].len();
            for sh_dim in sh.iter().rev() {
                if *sh_dim != 1 {
                    if *sh_dim != self.0[0][i].size {
                        merge_possible = false;
                        break;
                    } else {
                        new_shape.insert(0, self.0[0][i]);
                        i -= 1;
                    }
                } else {
                    if *sh_dim != self.0[0][i].size {
                        new_shape.insert(0, Dimension {
                            size: 1,
                            stride: self.0[0][i].stride,
                            left_mask: 0,
                            right_mask: 0,
                        });
                    } else {
                        i -= 1;
                    }
                }
            }
            if merge_possible {
                self.0[0] = new_shape;
                return
            }
        }

        // Merge squeeze
        if sh.rank() < self.0[0].len() {
            let mut merge_possible = true;
            let mut new_shape = Vec::new();
            let mut i = sh.rank();
            for dim in self.0[0].iter().rev() {
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
                self.0[0] = new_shape;
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
            self.0[0] = new_shape;
        } else {
            self.0.insert(0, new_shape);
        }
    }

    pub(crate) fn pad(&mut self, padding: &[(usize, usize)]) {
        debug_assert_eq!(self.0[0].len(), padding.len());
        for (dim, (lp, rp)) in self.0[0].iter_mut().zip(padding) {
            dim.left_mask = *lp;
            dim.right_mask = *rp;
        }
    }

    pub(crate) fn expand(&mut self, sh: &Shape) {
        debug_assert_eq!(self.0[0].len(), sh.rank());
        for (dim, sh_dim) in self.0[0].iter_mut().zip(sh) {
            if dim.size != *sh_dim {
                dim.size = *sh_dim;
                dim.stride = 0;
            }
        }
    }

    pub(crate) fn permute(&mut self, axes: &Axes) {
        debug_assert_eq!(self.0[0].len(), axes.len());
        self.0[0] = axes.iter().map(|axis| self.0[0][*axis]).collect();
    }
}
