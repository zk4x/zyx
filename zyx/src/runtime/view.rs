use crate::runtime::compiler::HWInfo;
use crate::scalar::Scalar;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Ordering;

#[cfg(feature = "debug1")]
use std::println;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
    //binds: Vec<usize>,
}

impl View {
    /// Create new View from given shape
    #[must_use]
    pub fn from(shape: &[usize]) -> Self {
        assert!(shape.len() > 0);
        let mut stride = 1;
        let mut first_shape: Vec<Dimension> = shape
            .iter()
            .rev()
            .map(|size| {
                let temp = Dimension {
                    size: *size,
                    stride,
                    len: *size,
                    shift: *size,
                };
                stride *= size;
                return temp;
            })
            .collect();
        first_shape.reverse();
        return Self {
            //binds: alloc::vec![0; first_shape.len()],
            shapes: alloc::vec![first_shape],
        };
    }

    /// Shape
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        return self.shapes[0].iter().map(|dim| dim.size).collect();
    }

    /// Strides
    #[must_use]
    pub fn strides(&self) -> Vec<usize> {
        return self.shapes[0].iter().map(|dim| dim.stride).collect();
    }

    /// Rank
    #[must_use]
    pub fn rank(&self) -> usize {
        return self.shapes[0].len();
    }

    /// Numel
    #[must_use]
    pub fn numel(&self) -> usize {
        return self.shapes[0].iter().map(|dim| dim.size).product();
    }

    /// Was the last shape expanded?
    /// This is used for local memory tiling.
    /// It may later be updated to work with expands
    /// in all shapes, not only the last one.
    pub fn is_expanded(&self) -> bool {
        //println!("Strides for is_expanded: {:?}", self.strides());
        return self.strides().contains(&0);
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
        return temp;
    }

    #[must_use]
    pub(crate) fn is_contiguous(&self) -> bool {
        return self.shapes.len() == 1 && self.is_last_shape_contiguous();
    }

    /// Pad view with padding.
    /// This function assumes standard padding beginning at last dimension.
    pub fn pad(&mut self, padding: &[isize]) {
        todo!()
        /*assert!(padding.len() > 0);
        assert!(self.shapes[0].len() >= padding.len());
        for (dim, (lp, rp)) in self.shapes[0]
            .iter_mut()
            .rev()
            .zip(padding.into_iter().tuples())
        {
            dim.size = (<usize as TryInto<isize>>::try_into(dim.size).unwrap() + lp + rp)
                .try_into()
                .unwrap();
            dim.shift = (<usize as TryInto<isize>>::try_into(dim.shift).unwrap() + lp) as usize;
        }*/
    }

    /// Expand view into different shape
    pub fn expand(&mut self, shape: &[usize]) {
        #[cfg(feature = "debug1")]
        println!(
            "Expanding shape {:?}, strides {:?}",
            self.shape(),
            self.strides()
        );
        assert!(shape.len() > 0);
        if shape.len() > self.shapes[0].len() {
            let mut sh = self.shape();
            let mut i = sh.len();
            for d in shape.iter().rev() {
                if i == 0 {
                    // Adding dimensions to the front of the shape
                    sh.insert(i, 1);
                } else {
                    i -= 1;
                }
                //println!("Comparing {d}, {}", sh[i]);
                match d.cmp(&sh[i]) {
                    Ordering::Less => {
                        i += 1;
                        sh.insert(i, 1);
                    }
                    Ordering::Greater => {
                        //println!("Here {}", sh[i]);
                        assert_eq!(sh[i], 1, "Cannot expand {:?} into {shape:?}", self.shape());
                    }
                    Ordering::Equal => {}
                }
            }
            //println!("Expand reshape: {sh:?}");
            self.reshape(&sh);
        }
        assert_eq!(self.shapes[0].len(), shape.len());
        for (dim, sh_dim) in self.shapes[0].iter_mut().zip(shape) {
            if dim.size != *sh_dim {
                assert_eq!(dim.size, 1);
                dim.size = *sh_dim;
                dim.stride = 0;
            }
        }
        #[cfg(feature = "debug1")]
        println!("to shape {:?}, strides {:?}", self.shape(), self.strides());
    }

    /// Permute view with axes
    pub fn permute(&mut self, axes: &[usize]) {
        //println!("Permuting to {axes:?}");
        assert_eq!(self.shapes[0].len(), axes.len());
        self.shapes[0] = axes.iter().map(|axis| self.shapes[0][*axis]).collect();
    }

    /// Reshape view into different shape
    pub fn reshape(&mut self, shape: &[usize]) {
        #[cfg(feature = "debug1")]
        println!(
            "Len: {} Reshaping shape {:?}, strides {:?}",
            self.shapes.len(),
            self.shape(),
            self.strides()
        );
        assert!(shape.len() > 0);
        assert_eq!(self.numel(), shape.iter().product());
        if self.shape() == *shape {
            return;
        }
        let mut new_stride = 1;
        let mut new_shape: Vec<Dimension> = shape
            .iter()
            .copied()
            .rev()
            .map(|size| {
                let stride = new_stride;
                new_stride *= size;
                Dimension {
                    size,
                    stride,
                    len: size,
                    shift: size,
                }
            })
            .collect();
        new_shape.reverse();
        if self.is_last_shape_contiguous() {
            // Merge contiguous shape
            self.shapes[0] = new_shape.clone();
            #[cfg(feature = "debug1")]
            println!(
                "merging contiguous to shape {:?} strides {:?}",
                self.shape(),
                self.strides()
            );
            return;
        }

        // TODO perhaps we can merge more shapes with the previous shape

        // TODO perhaps merge reshapes joining dimensions

        // Merge dimension splits (also works for unsqueeze)
        if shape.len() > self.shapes[0].len() {
            let mut merge_possible = true;
            let mut di = self.shapes[0].len() - 1; // index to old shape
            let mut new_shape = Vec::new();
            let mut expansion_stride = 1;
            // Iterate in reverse
            for size in shape.iter().copied().rev() {
                let old_dim = self.shapes[0][di];
                //libc_print::libc_println!("New dim {}, old dim {}", size, old_dim.size);
                if size == old_dim.size {
                    //new_binds.insert(0, self.binds[di]);
                    new_shape.insert(0, old_dim);
                    if di > 0 {
                        di -= 1;
                    }
                    expansion_stride = 1;
                } else if size < old_dim.size {
                    // If this dimension was padded, we probably can't merge
                    //println!("Old len {}, old size {}, old stride {}", old_dim.len, old_dim.size, old_dim.stride);
                    if old_dim.stride != 0
                        && (old_dim.len != old_dim.size || old_dim.shift != old_dim.len)
                    {
                        merge_possible = false;
                        break;
                    }
                    new_shape.insert(
                        0,
                        Dimension {
                            size,
                            stride: old_dim.stride * expansion_stride,
                            len: size,
                            shift: size,
                        },
                    );
                    expansion_stride *= size;
                } else {
                    merge_possible = false;
                    break;
                }
            }
            if merge_possible {
                self.shapes[0] = new_shape;
                #[cfg(feature = "debug1")]
                println!(
                    "merging to shape {:?} strides {:?}",
                    self.shape(),
                    self.strides()
                );
                return;
            }
        }

        // If it could not be merged
        self.shapes.insert(0, new_shape);

        #[cfg(feature = "debug1")]
        println!(
            "adding new shape {:?}, strides {:?}",
            self.shape(),
            self.strides()
        );
    }

    /// Bind index id to dimension, this is used for index names when generating indexes
    /// for buffers in kernels (ir_index function). Overwrites previous binds if any.
    //pub fn bind(&mut self, index: usize, dim: usize) {
    //self.binds[dim] = index;
    //}

    /// This assumes that self is 3d (element wise) or 4d view (reduce).
    /// It calculates optimal local dimensions and reshapes view to 8d (3 global, 3 local
    /// and 2 register dimensions) or 10d (if there is reduce)
    /// Result has shape:
    /// gws0, lws0, gws1, lws1, rws1, gws2, lws2, rws2
    /// or
    /// gws0, lws0, gws1, lws1, rws1, gws2, lws2, rws2, gws3, rws3
    pub(crate) fn optimize_local_mem_size_and_work_per_thread(&mut self, hwinfo: &HWInfo) {
        //println!("Optimize local and wpt {:?}", self.shape());
        assert!(
            self.rank() == 3 || self.rank() == 4,
            "Incorrect view rank for applying optimizations."
        );

        // Optimize work size per thread
        // Over all dimensions excluding first (batch) dimension.
        // Each dimension is divided by the same value d. This can be later optimized
        // for different wpt in each dimension.
        // Current (year 2024) gpus have a bit more than 64 registers.
        // This is set by hwinfo.num_registers.
        // On current devices d will be 8, which means 64 elements per thread in element
        // wise kernels and 512 elements per thread in reduce kernels (with 64 element
        // accumulator)
        let s = &self.shapes[0];
        //let d = Scalar::sqrt(hwinfo.num_registers as i64) as usize;
        let d = 1;
        let mut dims = [s[0].size, 1, s[1].size / d, 1, d, s[2].size / d, 1, d];

        // Optimize local work size
        // Local work will be over second and third dimensions, currently
        // we will not make it over first dimension (usually batch dimension)
        // or reduce dimension. This can be later changed for further optimizations.
        // equally distribute max_local_work_size to dims[4] and dims[5]
        let sqrt = Scalar::sqrt(hwinfo.max_work_group_size as i64) as usize;
        let mut total = 1;
        let mut n = 1;
        while dims[2] % (n * 2) == 0 && n * 2 <= sqrt {
            n *= 2;
        }
        dims[2] /= n;
        dims[3] *= n;
        total *= n;
        // put the rest into third dimension
        let mut n = 1;
        while dims[5] % (n * 2) == 0 && n * 2 * total <= hwinfo.max_work_group_size {
            n *= 2;
        }
        dims[5] /= n;
        dims[6] *= n;
        total *= n;
        // if third dimension was too small, put the rest into second dimension
        let mut n = 1;
        while dims[2] % (n * 2) == 0 && n * 2 * total <= hwinfo.max_work_group_size {
            n *= 2;
        }
        dims[2] /= n;
        dims[3] *= n;

        //println!("Rank optim: {}", self.rank());
        if self.rank() == 4 {
            // if reduce
            let dims = [
                dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7], s[3].size,
                d,
            ];
            //println!("to {:?}", dims);
            self.reshape(&dims);
        } else {
            //println!("to {:?}", dims);
            self.reshape(&dims);
        }
    }

    /// Get index from this view, using bound indices, this function does not work
    /// correctly if some dimensions don't have bound indices.
    /// Binds are indices into this dimension
    pub(crate) fn ir_index(&self, binds: &[usize]) -> Index {
        assert_eq!(
            binds.len(),
            self.rank(),
            "Number of binds ({}) differs from number of dimensions ({}).",
            binds.len(),
            self.rank()
        );
        //println!("{self:?}");
        if self.is_contiguous() {
            return Index::Contiguous {
                dims: self.shapes[0]
                    .iter()
                    .zip(binds)
                    .map(|(dim, b)| (*b, dim.stride))
                    .collect(),
                //padding_condition: String::new(),
            };
        }
        if self.shapes.len() == 1 {
            return Index::Strided {
                dims: self.shapes[0]
                    .iter()
                    .zip(binds)
                    .map(|(dim, b)| (*b, dim.stride))
                    .collect(),
            };
        }
        // + idx / ost % d * st
        let mut reshapes = Vec::with_capacity(self.shapes.len() - 1);
        for shape in self.shapes[1..].iter().rev() {
            let mut ost = 1;
            let mut reshape = Vec::with_capacity(shape.len());
            for dim in shape.iter().rev() {
                reshape.push((ost, dim.size, dim.stride));
                ost *= dim.size;
            }
            reshape.reverse();
            reshapes.push(reshape);
        }
        //println!("Reshapes: {reshapes:?}");
        return Index::Reshaped {
            dims: self.shapes[0]
                .iter()
                .zip(binds)
                .map(|(dim, b)| (*b, dim.stride))
                .collect(),
            reshapes,
            padding_condition: String::new(), // TODO add correct padding condition
        };
    }
}

// With this representation of index, we can find repeating
// multipliers and extract them out into common factors.
// However this would be a bit of micro-optimization, as OpenCL, CUDA, WGPU
// and most other compilers extract them automatically.
// This will be needed if we want to directly generate SPIR or PTX IR.

/// Virtual representation of index into view
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Index {
    /// For variables that only have single element (scalars),
    /// such as most register variables.
    None,
    /// Expanded and/or padded
    /// Pairs of index id and multiplier.
    /// Can use wide loads directly with pointer casts.
    Contiguous {
        /// Dimension and multiplier
        dims: BTreeMap<usize, usize>,
        // When should the padding get applied?
        //padding_condition: String,
    },
    /// Expanded and/or permuted
    /// Pairs of index id and multiplier.
    /// Wide loads are possible only if we can transpose it in the kernel
    Strided {
        /// Dimension and multiplier
        dims: BTreeMap<usize, usize>,
        // When should the padding get applied?
        //padding_condition: String,
    },
    /// Expanded, permuted, reshaped and/or padded
    /// Only if reshape could not be merged.
    Reshaped {
        /// Multiple dimension and multipliers
        dims: BTreeMap<usize, usize>,
        reshapes: Vec<Vec<(usize, usize, usize)>>,
        /// When should the padding get applied?
        padding_condition: String,
    },
}

#[test]
fn test_unsqueeze() {
    use alloc::vec;
    let mut view0 = View::from(&[3, 4, 2, 1]);
    view0.permute(&[0, 3, 1, 2]);
    assert_eq!(view0.shape(), vec![3, 1, 4, 2]);
    view0.expand(&[3, 5, 4, 2]);
    assert_eq!(view0.shape(), vec![3, 5, 4, 2]);
    view0.reshape(&[3, 1, 5, 4, 2]);
    assert_eq!(view0.shapes.len(), 1);
    assert_eq!(
        view0.shapes[0],
        vec![
            Dimension {
                size: 3,
                stride: 8,
                len: 3,
                shift: 3
            },
            Dimension {
                size: 1,
                stride: 8,
                len: 1,
                shift: 1
            },
            Dimension {
                size: 5,
                stride: 0,
                len: 1,
                shift: 1
            },
            Dimension {
                size: 4,
                stride: 2,
                len: 4,
                shift: 4
            },
            Dimension {
                size: 2,
                stride: 1,
                len: 2,
                shift: 2
            },
        ]
    );
}

#[test]
fn test_standard() {
    use alloc::vec;
    let mut view0 = View::from(&[3, 4, 2, 1]);
    view0.permute(&[0, 3, 1, 2]);
    assert_eq!(view0.shape(), vec![3, 1, 4, 2]);
    view0.expand(&[3, 5, 4, 2]);
    assert_eq!(view0.shape(), vec![3, 5, 4, 2]);
    view0.reshape(&[15, 8]);
    assert_eq!(view0.shape(), vec![15, 8]);
}

#[test]
fn test_reshape_merge1() {
    use alloc::vec;
    let mut view0 = View::from(&[3, 4, 20]);
    view0.permute(&[2, 0, 1]);
    view0.reshape(&[5, 4, 3, 4]);
    //libc_println!("{view0:#?}");
    assert_eq!(view0.shapes.len(), 1);
    assert_eq!(
        view0.shapes[0],
        vec![
            Dimension {
                size: 5,
                stride: 4,
                len: 5,
                shift: 5,
            },
            Dimension {
                size: 4,
                stride: 1,
                len: 4,
                shift: 4
            },
            Dimension {
                size: 3,
                stride: 80,
                len: 3,
                shift: 3
            },
            Dimension {
                size: 4,
                stride: 20,
                len: 4,
                shift: 4
            },
        ]
    );
}

#[test]
fn test_ir_index() {
    let view0 = View::from(&[2, 43, 1]);
    assert_eq!(view0.shapes.len(), 1);
    assert_eq!(
        view0.ir_index(&[0, 1, 2]),
        Index::Contiguous {
            dims: BTreeMap::from([(0, 43), (1, 1), (2, 1)])
        }
    );
}

#[test]
#[should_panic]
fn test_expand_2() {
    let mut view0 = View::from(&[2, 2]);
    view0.expand(&[2, 3, 2]);
}

#[test]
fn test_expand_3() {
    let mut view0 = View::from(&[2, 1, 2]);
    view0.expand(&[2, 3, 2]);
    view0.reshape(&[2, 1, 3, 2]);
    view0.reshape(&[2, 1, 1, 1, 1, 3, 1, 1, 2, 1]);
    //println!("{view0:?}");
    assert_eq!(
        Index::Strided {
            dims: BTreeMap::from([
                (0, 2),
                (1, 2),
                (2, 2),
                (3, 2),
                (4, 2),
                (5, 0),
                (6, 0),
                (7, 0),
                (8, 1),
                (9, 1)
            ])
        },
        view0.ir_index(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    );
}

#[test]
fn test_expand_4() {
    let mut view0 = View::from(&[2, 3, 1]);
    view0.expand(&[3, 2, 3, 4]);
    assert_eq!(view0.shape(), [3, 2, 3, 4]);
    assert_eq!(view0.strides(), [0, 3, 1, 0]);
}
