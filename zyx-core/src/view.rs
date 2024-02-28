extern crate alloc;
use crate::scalar::Scalar;
use crate::{axes::Axes, shape::Shape};
use alloc::boxed::Box;
use alloc::string::String;
use alloc::{vec, vec::Vec};

/// View type
pub enum ViewType {
    /// Contiguous
    Contiguous,
    /// Permuted or expanded
    Strided,
    /// Permuted, expanded or reshaped
    Reshaped,
    /// Permuted, expanded, reshaped or padded
    Padded,
}

/// View holds shape of the tensor and allows for arbitrary number of movement ops
/// (reshape, expand, pad, permute) to be executed as noops (without accessing the
/// actual data).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct View {
    // TODO only 2 shape and stride pairs are needed
    views: Vec<InnerView>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct InnerView {
    shape: Shape,
    strides: Shape,
    padding: Box<[(i64, i64)]>,
}

impl InnerView {
    #[must_use]
    fn contiguous(&self) -> bool {
        self.shape.strides() == self.strides && !self.padded()
    }

    #[must_use]
    fn padded(&self) -> bool {
        self.padding.iter().any(|(lp, rp)| *lp != 0 || *rp != 0)
    }
}

/// CPU iterator
pub struct CPUPaddedIter<'a, T> {
    data: &'a [T],
    view: &'a View,
    idx: usize,
    num_iters: usize,
}

impl<'a, T: Scalar> Iterator for CPUPaddedIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx > self.num_iters {
            return None;
        }
        let mut idx = self.idx;
        self.idx += 1;
        for InnerView {
            shape,
            strides,
            padding,
        } in &self.view.views
        {
            let mut res = 0;
            for ((d, st), (lp, rp)) in shape.into_iter().zip(strides).zip(padding.iter()).rev() {
                let mut dim_idx = idx % d;
                if *lp > 0 {
                    let lpu = *lp as usize;
                    if dim_idx < lpu {
                        return Some(T::zero());
                    }
                    dim_idx -= lpu;
                } else if *lp < 0 {
                    dim_idx += (-*lp) as usize;
                }
                if *rp > 0 {
                    if dim_idx > *rp as usize {
                        return Some(T::zero());
                    }
                }
                res += dim_idx * st;
                idx /= d;
            }
            idx = res;
        }
        Some(self.data[idx].clone())
    }
}

/// CPU iterator
pub struct CPUReshapedIter<'a, T> {
    data: &'a [T],
    view: &'a View,
    idx: usize,
    num_iters: usize,
}

impl<'a, T: Scalar> Iterator for CPUReshapedIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx > self.num_iters {
            return None;
        }
        let mut idx = self.idx;
        self.idx += 1;
        for InnerView {
            shape,
            strides,
            padding: _,
        } in &self.view.views
        {
            let mut res = 0;
            for (d, st) in shape.into_iter().zip(strides).rev() {
                let dim_idx = idx % d;
                res += dim_idx * st;
                idx /= d;
            }
            idx = res;
        }
        Some(self.data[idx].clone())
    }
}

/// Strided iterator, only expand and permute
pub struct CPUStridedIter<'a, T> {
    data: &'a [T],
    shape: &'a [usize],
    strides: &'a [usize],
    idx: usize,
    num_iters: usize,
}

impl<'a, T: Scalar> Iterator for CPUStridedIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx > self.num_iters {
            return None;
        }
        let mut idx = self.idx;
        self.idx += 1;
        let mut res = 0;
        for (d, st) in self
            .shape
            .into_iter()
            .copied()
            .zip(self.strides.into_iter().copied())
            .rev()
        {
            res += idx % d * st;
            idx /= d;
        }
        Some(self.data[res].clone())
    }
}

impl View {
    /// Create new view from shape
    #[must_use]
    pub fn new(shape: Shape) -> Self {
        Self {
            views: vec![InnerView {
                strides: shape.strides(),
                padding: core::iter::repeat((0, 0)).take(shape.rank()).collect(),
                shape,
            }],
        }
    }

    /// Is this view contiguous?
    /// i. e. no padding, expands or permutes, only reshapes are allowed
    #[must_use]
    pub fn contiguous(&self) -> bool {
        self.views.iter().all(InnerView::contiguous)
    }

    /// Is this view padded?
    #[must_use]
    pub fn padded(&self) -> bool {
        self.views.iter().any(InnerView::padded)
    }

    /// For cpu backend
    #[must_use]
    pub fn view_type(&self) -> ViewType {
        if self.contiguous() {
            ViewType::Contiguous
        } else if self.padded() {
            ViewType::Padded
        } else if self.views.len() > 1 {
            ViewType::Reshaped
        } else {
            ViewType::Strided
        }
    }

    /// Simple iteration
    #[must_use]
    pub fn iterate_contiguous<'a, T: Scalar>(
        &'a self,
        data: &'a [T],
    ) -> impl Iterator<Item = T> + 'a {
        data.iter().cloned()
    }

    /// Iteration with expands and permutes
    #[must_use]
    pub fn iterate_strided<'a, T: Scalar>(&'a self, data: &'a [T]) -> impl Iterator<Item = T> + 'a {
        let InnerView {
            shape,
            strides,
            padding: _,
        } = self.views.first().unwrap();
        CPUStridedIter {
            data,
            num_iters: shape.numel() - 1,
            shape: shape.as_ref(),
            strides: strides.as_ref(),
            idx: 0,
        }
    }

    /// Iteration with expands, permutes and reshapes, but without padding
    #[must_use]
    pub fn iterate_reshaped<'a, T: Scalar>(
        &'a self,
        data: &'a [T],
    ) -> impl Iterator<Item = T> + 'a {
        CPUReshapedIter {
            data,
            view: self,
            idx: 0,
            num_iters: self.numel() - 1,
        }
    }

    /// Iteration with expands, permutes, reshapes and padding
    #[must_use]
    pub fn iterate_padded<'a, T: Scalar>(&'a self, data: &'a [T]) -> impl Iterator<Item = T> + 'a {
        CPUPaddedIter {
            data,
            view: self,
            idx: 0,
            num_iters: self.numel() - 1,
        }
    }

    /// Access data called name with idx0-idx{rank} converted into self view.
    /// This is used by compiled backends.
    /// Returns padding condition and index.
    /// If padding condition == 0, padding value is applied, if padding condition
    /// is one, value is drawn from data.
    #[must_use]
    pub fn cidx(&self) -> (String, String) {
        // TODO is padding correctly applied?
        // TODO simplify this as much as possible, not for performance (it is cached), just for clarity
        //std::println!("View: {self:?}");
        use alloc::format as f;
        let mut idx = String::new();
        let mut padding_condition = String::new();
        if self.contiguous() {
            for (i, st) in self.views[0].strides.iter().enumerate() {
                if *st == 1 {
                    idx += &f!("+idx{i}");
                } else {
                    idx += &f!("+idx{i}*{st}");
                }
            }
            idx.remove(0);
            return (padding_condition, idx);
        }
        if let Some(InnerView {
            shape,
            strides,
            padding,
        }) = self.views.first()
        {
            for (i, ((d, st), (left_p, right_p))) in shape
                .iter()
                .zip(strides.iter())
                .zip(padding.iter())
                .enumerate()
            {
                //std::println!("i: {i}, d: {d}, st: {st}, lp: {left_p}, rp: {right_p}");
                match *st {
                    0 => idx += "",
                    1 => idx += &f!("idx{i}+"),
                    _ => idx += &f!("idx{i}*{st}+"),
                }
                if *left_p < 0 {
                    idx += &f!("{}+", -left_p);
                } else if *left_p > 0 {
                    padding_condition = f!("{padding_condition} && (idx{i}>{})", left_p - 1);
                }
                if *right_p > 0 {
                    padding_condition =
                        f!("{padding_condition} && (idx{i}<{})", d - *right_p as usize);
                }
                if *left_p > 0 {
                    idx += &f!("-{}+", left_p);
                }
            }
            if idx.is_empty() {
                idx = f!("0+");
            }
        } else {
            return (padding_condition, "0".into());
        }
        idx.remove(idx.len() - 1);
        if self.views.len() == 1 {
            if !padding_condition.is_empty() {
                padding_condition = f!("{}", &padding_condition[4..]);
            }
            return (padding_condition, idx);
        }
        for InnerView {
            shape,
            strides,
            padding,
        } in &self.views[1..]
        {
            let n = shape.numel();
            idx.insert(0, '(');
            idx.push(')');
            let mut res = String::new();
            let mut ost = 1;
            for ((d, st), (left_p, right_p)) in
                shape.into_iter().zip(strides).zip(padding.iter()).rev()
            {
                //println!("d: {d}, st: {st}, lp: {left_p}, rp: {right_p}");
                //res += &f!("{idx}/{ost}%{d}*{st}+");
                //ost *= d;
                let mut temp = f!("{idx}");
                match ost {
                    0 => panic!(),
                    1 => {}
                    _ => temp += &f!("/{ost}"),
                }
                ost *= d;
                match *d {
                    0 => panic!(),
                    1 => temp = f!("0"),
                    _ => {
                        if ost < n {
                            temp += &f!("%{d}");
                        }
                    }
                }
                if *left_p < 0 {
                    temp = f!("{temp}+{}", -left_p);
                } else if *left_p > 0 {
                    padding_condition = f!("{padding_condition} && ({temp}>{})", left_p - 1);
                }
                if *right_p > 0 {
                    padding_condition =
                        f!("{padding_condition} && ({temp}<{})", d - *right_p as usize);
                }
                if *left_p > 0 {
                    temp = f!("({temp}-{left_p})");
                }
                match *st {
                    0 => temp = f!("0"),
                    1 => {}
                    _ => temp += &f!("*{st}"),
                }
                res += &f!("{temp}+");
            }
            idx = res;
            if !idx.is_empty() {
                idx.remove(idx.len() - 1);
            }
        }
        if !padding_condition.is_empty() {
            padding_condition = f!("{}", &padding_condition[4..]);
        }
        (padding_condition, idx)
    }

    /// Number of elements in view with self.shape()
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape().numel()
    }

    /// Last shape of self.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.views.first().unwrap().shape
    }

    /// Original (first) shape of self.
    #[must_use]
    pub fn original_shape(&self) -> &Shape {
        &self.views.last().unwrap().shape
    }

    /// Original number of elements of self.
    #[must_use]
    pub fn original_numel(&self) -> usize {
        let InnerView {
            shape,
            strides,
            padding,
        } = self.views.last().unwrap();
        shape
            .iter()
            .zip(strides.iter())
            .zip(padding.iter())
            .filter_map(|((d, s), (lp, rp))| {
                if *s != 0 {
                    Some((*d as i64 - lp - rp) as usize)
                } else {
                    None
                }
            })
            .product()
    }

    /// Expand self into shape
    #[must_use]
    pub fn expand(&self, shape: &Shape) -> Self {
        let mut views = self.views.clone();
        //std::println!("Expanding {views:?}");
        views[0].strides = views[0]
            .shape
            .expand_strides(shape, views[0].strides.clone());
        views[0].shape = shape.clone();
        let n = shape.rank() - views[0].padding.len();
        views[0].padding = core::iter::repeat((0, 0))
            .take(n)
            .chain(views[0].padding.iter().copied())
            .collect();
        //std::println!("To {views:?}");
        Self { views }
    }

    /// Pad self by padding
    #[must_use]
    pub fn pad(&self, new_padding: &[(i64, i64)]) -> Self {
        //std::println!("{:?}\n{new_padding:?}", self);
        let mut views = self.views.clone();
        if let Some(InnerView {
            shape,
            strides: _,
            padding,
        }) = views.first_mut()
        {
            // Invert padding order
            for (i, d) in shape.iter_mut().rev().enumerate() {
                if let Some((left, right)) = new_padding.get(i) {
                    *d = (*d as i64 + left + right) as usize;
                } else {
                    break;
                }
            }
            let n = padding.len() - new_padding.len();
            *padding = core::iter::repeat(&(0, 0))
                .take(n)
                .chain(new_padding.iter().rev())
                .zip(padding.iter())
                .map(|(x, y)| (x.0 + y.0, x.1 + y.1))
                .collect();
            //std::println!("new_padding: {:?}", padding);
        }
        Self { views }
    }

    /// Reshape self into shape
    #[must_use]
    pub fn reshape(&self, n_shape: &Shape) -> Self {
        //std::println!("Reshaping {self:?} into {n_shape}");
        if n_shape == self.shape() {
            return self.clone();
        }
        debug_assert_eq!(
            n_shape.numel(),
            self.numel(),
            "Can't reshape {} to {}",
            self.shape(),
            n_shape
        );
        let mut views = self.views.clone();
        // If we are reshaping InnerView that is contiguous, we just delete the last reshape
        if views.first().unwrap().contiguous() {
            views[0] = InnerView {
                shape: n_shape.clone(),
                strides: n_shape.strides(),
                padding: core::iter::repeat((0, 0)).take(n_shape.rank()).collect(),
            };
        } else {
            let shape = self.shape();
            if n_shape.rank() > shape.rank() && n_shape
                .iter()
                .filter(|d| **d != 1)
                .zip(shape.iter())
                .all(|(nd, d)| nd == d)
            {
                // If not  contiguous, then merge, this merges if reshape is unsqueeze
                //std::println!("Ok to merge {n_shape} with {}", self.shape());
                if let Some(InnerView {
                    shape,
                    strides,
                    padding,
                }) = views.first_mut()
                {
                    //std::println!("Merging");
                    *shape = n_shape.clone();
                    let mut n_strides: Vec<usize> = strides.clone().into();
                    let mut n_padding = padding.to_vec();
                    for (i, d) in n_shape.iter().rev().enumerate() {
                        if *d == 1 {
                            //std::println!("Inserting");
                            n_strides.insert(
                                n_strides.len() - i,
                                if i == 0 {
                                    1
                                } else {
                                    n_strides[n_strides.len() - i]
                                },
                            );
                            n_padding.insert(n_padding.len() - i, (0, 0));
                        }
                    }
                    //std::println!("n_strides: {n_strides:?}, n_padding: {n_padding:?}");
                    *strides = n_strides.into();
                    *padding = n_padding.into_boxed_slice();
                }
            } else {
                // If there is no merge.
                views.insert(
                    0,
                    InnerView {
                        shape: n_shape.clone(),
                        strides: n_shape.strides(),
                        padding: core::iter::repeat((0, 0)).take(n_shape.rank()).collect(),
                    },
                );
            }
        }
        //std::println!("Merged into: {:?}", views);
        Self { views }
    }

    /// Permute self by axes
    #[must_use]
    pub fn permute(&self, axes: &Axes) -> Self {
        //std::println!("{:?}\n{:?}", self, axes);
        let mut views = self.views.clone();
        views[0].shape = views[0].shape.permute(axes);
        views[0].strides = views[0].strides.permute(axes);
        let padding = &views[0].padding;
        let padding = axes.iter().map(|axis| padding[*axis]).collect();
        views[0].padding = padding;
        Self { views }
    }
}

/*#[test]
fn view() {
    use crate::axes::IntoAxes;

    let s0 = View::new([10, 15].into());
    let s5 = s0.permute(&[1, 0].into_axes(2)).reshape(&[10, 15].into());
    /*let s0 = View::new(Shape::from([4, 5, 2]));
    let s1 = s0.permute(&[2, 0, 1].into_axes(3));
    let s2 = s1.reshape(&[4, 1, 5, 2, 1].into());
    let s3 = s2.expand(&[4, 3, 5, 2, 2].into());
    let s4 = s3.permute(&[3, 0, 4, 2, 1].into_axes(5));
    let s5 = s4.reshape(&[12, 20].into());*/
    for InnerView { shape, strides, padding } in s5.views {
        std::println!("{shape:?}, {strides:?}, {padding:?}");
    }
    panic!();
}*/
