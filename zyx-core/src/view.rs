extern crate alloc;
use crate::{
    axes::Axes,
    shape::Shape,
};
use alloc::{vec, vec::Vec};
use alloc::boxed::Box;

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
    padding_value: i32, // TODO make this with other dtypes as well
}

impl View {
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn new(shape: Shape) -> Self {
        Self {
            views: vec![InnerView {
                strides: shape.strides(),
                padding: Box::new([(0, 0)]),
                shape,
                padding_value: 0,
            }],
        }
    }

    pub(crate) fn contiguous(&self) -> bool {
        self.views.iter().all(
            |InnerView {
                 shape,
                 strides,
                 ..
             }| shape.strides() == strides.clone(),
        )
    }

    pub(crate) fn get_idx(&self, mut idx: usize) -> usize {
        // TODO can this be faster???
        for InnerView {
            shape,
            strides,
            padding,
            padding_value,
        } in &self.views
        {
            let mut res = 0;
            for (d, st) in shape.into_iter().zip(strides).rev() {
                res += idx % d * st;
                idx /= d;
            }
            idx = res;
        }
        idx
    }

    pub fn cidx(&self) -> alloc::string::String {
        use alloc::format;
        let mut idx = alloc::string::String::new();
        for (i, st) in self.views.first().unwrap().strides.into_iter().enumerate() {
            match *st {
                0 => {}
                1 => {
                    idx += &format!("idx{i}+");
                }
                _ => {
                    idx += &format!("idx{i}*{st}+");
                }
            }
        }
        if idx.is_empty() {
            return "0".into();
        }
        idx.remove(idx.len() - 1);
        if self.views.len() == 1 {
            return idx;
        }
        for InnerView {
            shape,
            strides,
            padding,
            padding_value,
        } in &self.views[1..]
        {
            idx.insert(0, '(');
            idx.push(')');
            let mut res = alloc::string::String::new();
            let mut ost = 1;
            for (d, st) in shape.into_iter().zip(strides).rev() {
                res += &format!("{idx}/{ost}%{d}*{st}+");
                ost *= d;
                /*let mut temp = format!("{idx}");
                match ost {
                    0 => { ost *= d; continue }
                    1 => {}
                    _ => { temp += &format!("/{ost}"); }
                }
                ost *= d;
                match *d {
                    0 => { continue }
                    1 => { temp = format!("1") }
                    _ => { temp += &format!("%{d}"); }
                }
                match *st {
                    0 => { continue }
                    1 => {}
                    _ => { temp += &format!("*{st}"); }
                }
                res += &format!("{temp}+");*/
            }
            idx = res;
            if !idx.is_empty() {
                idx.remove(idx.len() - 1);
            }
        }
        idx
    }

    pub(crate) fn numel(&self) -> usize {
        self.shape().numel()
    }

    pub(crate) fn shape(&self) -> &Shape {
        &self.views[0].shape
    }

    pub(crate) fn _resize(&self, shape: &Shape) -> Self {
        let mut shapes = self.views.clone();
        shapes[0].padding = Box::new([(0, 0)]);
        shapes[0].shape = shape.clone();
        Self { views: shapes }
    }

    pub(crate) fn expand(&self, shape: &Shape) -> Self {
        let mut shapes = self.views.clone();
        //std::println!("Expanding {shapes:?}");
        shapes[0].strides = shapes[0]
            .shape
            .expand_strides(shape, shapes[0].strides.clone());
        shapes[0].shape = shape.clone();
        //std::println!("To {shapes:?}");
        Self { views: shapes }
    }

    pub(crate) fn reshape(&self, shape: &Shape) -> Self {
        let mut shapes = self.views.clone();
        shapes.insert(
            0,
            InnerView {
                shape: shape.clone(),
                strides: shape.strides(),
                padding: Box::new([(0, 0)]),
                padding_value: 0,
            },
        );
        Self { views: shapes }
    }

    pub(crate) fn permute(&self, axes: &Axes) -> Self {
        let mut shapes = self.views.clone();
        shapes[0].shape = shapes[0].shape.permute(axes);
        shapes[0].strides = shapes[0].strides.permute(axes);
        Self { views: shapes }
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
    for (shape, strides) in s5.shapes {
        std::println!("{shape:?}, {strides:?}");
    }
    //panic!();
}*/
