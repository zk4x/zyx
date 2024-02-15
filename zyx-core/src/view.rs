extern crate alloc;
use crate::{axes::Axes, shape::Shape};
use alloc::boxed::Box;
use alloc::string::String;
use alloc::{vec, vec::Vec};

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
    fn contiguous(&self) -> bool {
        self.shape.strides() == self.strides && self.padding.iter().all(|(lp, rp)| *lp == 0 && *rp == 0)
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
        self.views
            .iter()
            .all(InnerView::contiguous)
    }

    /// Convert contiguous idx into idx indexing data with self view
    #[must_use]
    pub fn get_idx(&self, mut idx: usize) -> usize {
        // TODO can this be faster???
        // Preferably like MUCH faster???
        /*if *left_p < 0 {
            idx += &f!("+{}", -left_p);
        } else if *left_p > 0 {
            padding_condition = f!("{padding_condition} && (idx{i}>{})", left_p - 1);
        }
        if *right_p > 0 {
            padding_condition =
                f!("{padding_condition} && (idx{i}<{})", d - *right_p as usize);
        }
        if *left_p > 0 {
            idx += &f!("-{}", left_p);
        }*/
        for InnerView {
            shape,
            strides,
            padding,
        } in &self.views
        {
            let mut res = 0;
            for ((d, st), (lp, rp)) in shape.into_iter().zip(strides).zip(padding.iter()).rev() {
                res += idx % d * st;
                idx /= d;
            }
            idx = res;
        }
        idx
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
        use std::println;
        println!("View: {self:?}");
        use alloc::format as f;
        let mut idx = String::new();
        let mut padding_condition = String::new();
        if self.contiguous() {
            for i in 0..self.shape().rank() {
                idx += &f!("+idx{i}");
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
                //println!("i: {i}, d: {d}, st: {st}, lp: {left_p}, rp: {right_p}");
                match *st {
                    0 => {}
                    1 => {
                        idx += &f!("idx{i}+");
                    }
                    _ => {
                        idx += &f!("idx{i}*{st}+");
                    }
                }
                if *left_p < 0 {
                    idx += &f!("+{}", -left_p);
                } else if *left_p > 0 {
                    padding_condition = f!("{padding_condition} && (idx{i}>{})", left_p - 1);
                }
                if *right_p > 0 {
                    padding_condition =
                        f!("{padding_condition} && (idx{i}<{})", d - *right_p as usize);
                }
                if *left_p > 0 {
                    idx += &f!("-{}", left_p);
                }
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
                    _ => if ost < n { temp += &f!("%{d}"); }
                }
                if *left_p < 0 {
                    temp = f!("{temp}+{}", -left_p);
                } else if *left_p > 0 {
                    padding_condition = f!("{padding_condition} && ({temp}>{})", left_p - 1);
                }
                if *right_p > 0 {
                    padding_condition = f!("{padding_condition} && ({temp}<{})", d - *right_p as usize);
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
        let InnerView { shape, strides, padding } = self.views.last().unwrap();
        shape.iter().zip(strides.iter()).zip(padding.iter()).filter_map(|((d, s), (lp, rp))| if *s != 0 { Some((*d as i64-lp-rp) as usize) } else { None }).product()
    }

    /// Expand self into shape
    #[must_use]
    pub fn expand(&self, shape: &Shape) -> Self {
        // TODO fix padding if needed
        let mut views = self.views.clone();
        //std::println!("Expanding {views:?}");
        views[0].strides = views[0]
            .shape
            .expand_strides(shape, views[0].strides.clone());
        views[0].shape = shape.clone();
        //std::println!("To {views:?}");
        Self { views }
    }

    /// Pad self by padding
    #[must_use]
    pub fn pad(&self, new_padding: &[(i64, i64)]) -> Self {
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
            std::println!("new_padding: {:?}", padding);
        }
        Self { views }
    }

    /// Reshape self into shape
    #[must_use]
    pub fn reshape(&self, shape: &Shape) -> Self {
        if shape == self.shape() {
            return self.clone();
        }
        let mut views = self.views.clone();
        // If we are reshaping InnerView that is contiguous, we just delete the last reshape
        if views.first().unwrap().contiguous() {
            views[0] = InnerView {
                shape: shape.clone(),
                strides: shape.strides(),
                padding: core::iter::repeat((0, 0)).take(shape.rank()).collect(),
            };
        } else {
            views.insert(
                0,
                InnerView {
                    shape: shape.clone(),
                    strides: shape.strides(),
                    padding: core::iter::repeat((0, 0)).take(shape.rank()).collect(),
                },
            );
        }
        Self { views }
    }

    /// Permute self by axes
    #[must_use]
    pub fn permute(&self, axes: &Axes) -> Self {
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

    let s0 = View::new(Shape::from([4, 5, 2]));
    let s1 = s0.permute(&[2, 0, 1].into_axes(3));
    let s2 = s1.reshape(&[4, 1, 5, 2, 1].into());
    let s3 = s2.expand(&[4, 3, 5, 2, 2].into());
    let s4 = s3.permute(&[3, 0, 4, 2, 1].into_axes(5));
    let s5 = s4.reshape(&[12, 20].into());
    for (shape, strides) in s5.views {
        std::println!("{shape:?}, {strides:?}");
    }
    //panic!();
}*/
