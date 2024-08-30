use std::rc::Rc;

use zyx::{DType, Scalar};

// Just a very barebones and slow CPU tensor that is slow, but verifiably correct
#[derive(Clone)]
struct CPUTensor {
    view: View,
    data: Data,
}

macro_rules! unary_op {
    ($this: expr, $op: expr) => {{
        CPUTensor {
            view: $this.view.clone(),
            data: match &$this.data {
                Data::F32(data) => Data::F32(unary(data, $op)),
                Data::F64(data) => Data::F64(unary(data, $op)),
                Data::I32(data) => Data::I32(unary(data, $op)),
            }
        }
    }};
}

macro_rules! binary_op {
    ($x: expr, $y: expr, $op: expr) => {{
        CPUTensor {
            view: $x.view.clone(),
            data: match &$x.data {
                Data::F32(xdata) => {
                    let Data::F32(ydata) = &$y.data else { panic!() };
                    Data::F32(binary(&$x.view, xdata, &$y.view, ydata, $op))
                }
                Data::F64(xdata) => {
                    let Data::F64(ydata) = &$y.data else { panic!() };
                    Data::F64(binary(&$x.view, xdata, &$y.view, ydata, $op))
                }
                Data::I32(xdata) => {
                    let Data::I32(ydata) = &$y.data else { panic!() };
                    Data::I32(binary(&$x.view, xdata, &$y.view, ydata, $op))
                }
            },
        }
    }};
}

// Only implements all ops in node.rs
// No backpropagation
impl CPUTensor {
    fn new<T: Scalar>(data: &[T]) -> CPUTensor {
        use std::mem::transmute as t;
        CPUTensor { view: View::new(&[data.len()]), data: match T::dtype() {
            DType::BF16 => todo!(),
            DType::F16 => todo!(),
            DType::F32 => Data::F32(unsafe { t::<_, &[f32]>(data) }.into()),
            DType::F64 => todo!(),
            DType::CF32 => todo!(),
            DType::CF64 => todo!(),
            DType::U8 => todo!(),
            DType::I8 => todo!(),
            DType::I16 => todo!(),
            DType::I32 => todo!(),
            DType::I64 => todo!(),
            DType::Bool => todo!(),
        } }
    }

    fn to_vec<T: Scalar>(&self) -> Vec<T> {
        use std::mem::transmute as t;
        let numel = self.view.numel();
        match &self.data {
            Data::F32(data) => unsafe {
                t(self
                    .view
                    .iterate_padded(data)
                    .take(numel)
                    .collect::<Vec<f32>>())
            },
            Data::F64(data) => todo!(),
            Data::I32(data) => todo!(),
        }
    }

    fn shape(&self) -> Vec<usize> {
        self.view.shape()
    }

    fn cast(&self, dtype: DType) -> CPUTensor {
        let data = match &self.data {
            Data::F32(data) => match dtype {
                DType::F32 => Data::F32(unary(data, Scalar::cast)),
                DType::F64 => Data::F64(unary(data, Scalar::cast)),
                DType::I32 => Data::I32(unary(data, Scalar::cast)),
                _ => todo!(),
            },
            Data::F64(data) => match dtype {
                DType::F32 => Data::F32(unary(data, Scalar::cast)),
                DType::F64 => Data::F64(unary(data, Scalar::cast)),
                DType::I32 => Data::I32(unary(data, Scalar::cast)),
                _ => todo!(),
            },
            Data::I32(data) => match dtype {
                DType::F32 => Data::F32(unary(data, Scalar::cast)),
                DType::F64 => Data::F64(unary(data, Scalar::cast)),
                DType::I32 => Data::I32(unary(data, Scalar::cast)),
                _ => todo!(),
            },
        };
        CPUTensor { view: self.view.clone(), data }
    }

    fn relu(&self) -> CPUTensor {
        unary_op!(self, Scalar::relu)
    }

    fn neg(&self) -> CPUTensor {
        unary_op!(self, Scalar::neg)
    }

    fn exp2(&self) -> CPUTensor {
        unary_op!(self, Scalar::exp2)
    }

    fn log2(&self) -> CPUTensor {
        unary_op!(self, Scalar::log2)
    }

    fn inv(&self) -> CPUTensor {
        unary_op!(self, Scalar::inv)
    }

    fn sqrt(&self) -> CPUTensor {
        unary_op!(self, Scalar::sqrt)
    }

    fn sin(&self) -> CPUTensor {
        unary_op!(self, Scalar::sin)
    }

    fn cos(&self) -> CPUTensor {
        unary_op!(self, Scalar::cos)
    }

    fn not(&self) -> CPUTensor {
        unary_op!(self, Scalar::not)
    }

    fn nonzero(&self) -> CPUTensor {
        unary_op!(self, Scalar::nonzero)
    }

    fn add(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::add)
    }

    fn sub(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::sub)
    }

    fn mul(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::mul)
    }

    fn div(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::div)
    }

    fn pow(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::pow)
    }

    fn cmplt(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::cmplt)
    }

    fn cmpgt(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::cmpgt)
    }

    fn max(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::max)
    }

    fn or(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::or)
    }

    fn reduce_sum(&self, axes: &[usize]) -> CPUTensor {
        todo!()
    }

    fn reduce_max(&self, axes: &[usize]) -> CPUTensor {
        todo!()
    }

    fn pad(&self, padding: &[(isize, isize)]) -> CPUTensor {
        CPUTensor {
            view: self.view.pad(padding),
            data: self.data.clone(), // just rc clone
        }
    }

    fn permute(&self, axes: &[usize]) -> CPUTensor {
        CPUTensor {
            view: self.view.permute(axes),
            data: self.data.clone(), // just rc clone
        }
    }

    fn expand(&self, shape: &[usize]) -> CPUTensor {
        CPUTensor {
            view: self.view.expand(shape),
            data: self.data.clone(), // just rc clone
        }
    }

    fn reshape(&self, shape: &[usize]) -> CPUTensor {
        CPUTensor {
            view: self.view.reshape(shape),
            data: self.data.clone(), // just rc clone
        }
    }
}

#[derive(Debug, Clone)]
enum Data {
    F32(Rc<[f32]>),
    F64(Rc<[f64]>),
    I32(Rc<[i32]>),
}

impl Data {
    unsafe fn as_type<T: Scalar>(&self) -> &[T] {
        use std::mem::transmute as t;
        match self {
            Data::F32(data) => t::<&[f32], &[T]>(data.as_ref()),
            Data::F64(data) => t::<&[f64], &[T]>(data.as_ref()),
            Data::I32(data) => t::<&[i32], &[T]>(data.as_ref()),
        }
    }
}

fn unary<T: Scalar + Sync + Send, T2: Scalar + Send>(
    data: &[T],
    op: impl Fn(T) -> T2 + Sync + Send,
) -> Rc<[T2]> {
    data.iter().cloned().map(op).collect()
}

fn binary<XT: Scalar + Sync + Send, YT: Scalar + Sync + Send, T2: Scalar + Send>(
    xview: &View,
    xdata: &[XT],
    yview: &View,
    ydata: &[YT],
    op: impl Fn(XT, YT) -> T2 + Sync + Send,
) -> Rc<[T2]> {
    xview
        .iterate_padded(xdata)
        .zip(yview.iterate_padded(ydata))
        .map(|(x, y)| op(x, y))
        .collect()
}

/*impl Interpreter {
    fn evaluate(
        &mut self,
        mut rcs: BTreeMap<Id, u32>,
        order: &[Id],
        nodes: &[Node],
    ) -> Result<(), ZyxError> {
        for nid in order.iter().copied() {
            //std::println!("Interpreting {nid}: {:?}", nodes[nid.i()]);
            match &nodes[nid.i()] {
                Node::Leaf(..) => {}
                Node::Sum(x, ax, sh) => {
                    let (view, data) = self.get(x);
                    let data = match data {
                        Data::F32(data) => Data::F32(reduce_op(view, data, ax, sh, true)),
                        Data::F64(data) => Data::F64(reduce_op(view, data, ax, sh, true)),
                        Data::I32(data) => Data::I32(reduce_op(view, data, ax, sh, true)),
                    };
                    self.views.insert(nid, (View::new(sh.clone()), nid));
                    self.buffers.insert(nid, data);
                }
                Node::Max(x, ax, sh) => {
                    let (view, data) = self.get(x);
                    let data = match data {
                        Data::F32(data) => Data::F32(reduce_op(view, data, ax, sh, false)),
                        Data::F64(data) => Data::F64(reduce_op(view, data, ax, sh, false)),
                        Data::I32(data) => Data::I32(reduce_op(view, data, ax, sh, false)),
                    };
                    self.views.insert(nid, (View::new(sh.clone()), nid));
                    self.buffers.insert(nid, data);
                }
            }
            //std::println!("Views {}, buffers {}", self.views.len(), self.buffers.len());
        }
        Ok(())
    }
}*/

fn reduce_op<T: Scalar + Sync + Send>(
    view: &View,
    data: &[T],
    axes: &[usize],
    res_shape: &[usize],
    sum_reduce: bool,
) -> Vec<T> {
    // TODO parallelize this
    use std::boxed::Box;
    // Strides of the input
    let shape = view.shape();
    let strides = shape_to_strides(&shape);
    // indices of dimensions that are not reduced
    let included_dims: Box<[usize]> = (0..shape.len()).filter(|x| !axes.contains(x)).collect();
    // Strides of the result
    let res_strides = shape_to_strides(res_shape);
    let mut res: Vec<T> = if sum_reduce {
        core::iter::repeat(T::zero())
    } else {
        core::iter::repeat(T::min_value())
    }
    .take(res_shape.iter().product())
    .collect();

    // Go over all data and apply sum function to correct values
    // then indices can be added just by making another vector and constantly
    // updating it (adding in case of sum) with new indices as new max/min are found
    for (i, x) in view.iterate_padded(data).enumerate() {
        // calculate index in result
        let mut j = 0;
        for dim in &*included_dims {
            j += ((i / strides[*dim]) % shape[*dim]) * res_strides[*dim];
        }
        // apply reduce function, in this case sum
        if sum_reduce {
            res[j] = Scalar::add(res[j].clone(), x);
        } else {
            res[j] = Scalar::max(res[j].clone(), x);
        }
    }
    res
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct View {
    // TODO only 2 shape and stride pairs are needed
    views: Vec<InnerView>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct InnerView {
    shape: Vec<usize>,
    strides: Vec<usize>,
    padding: Vec<(isize, isize)>,
}

impl InnerView {
    #[must_use]
    fn is_contiguous(&self) -> bool {
        self.shape.strides() == self.strides && !self.is_padded()
    }

    #[must_use]
    fn is_padded(&self) -> bool {
        self.padding.iter().any(|(lp, rp)| *lp != 0 || *rp != 0)
    }
}

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

impl View {
    pub fn new(shape: &[usize]) -> View {
        let shape = shape.into();
        let strides = shape_to_strides(&shape);
        View { views: vec![InnerView { shape, strides, padding: Vec::new() }] }
    }

    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        self.views.iter().all(InnerView::is_contiguous)
    }

    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        self.views.first().unwrap().shape.clone()
    }

    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    #[must_use]
    pub fn iterate_padded<'a, T: Scalar>(&'a self, data: &'a [T]) -> impl Iterator<Item = T> + 'a {
        CPUPaddedIter {
            data,
            view: self,
            idx: 0,
            num_iters: self.numel() - 1,
        }
    }

    #[must_use]
    pub fn expand(&self, shape: &[usize]) -> Self {
        let mut views = self.views.clone();
        //std::println!("Expanding {views:?}");
        views[0].strides = views[0]
            .shape
            .expand_strides(shape, views[0].strides.clone());
        views[0].shape = shape.clone();
        let n = shape.len() - views[0].padding.len();
        views[0].padding = core::iter::repeat((0, 0))
            .take(n)
            .chain(views[0].padding.iter().copied())
            .collect();
        //std::println!("To {views:?}");
        Self { views }
    }

    #[must_use]
    pub fn pad(&self, new_padding: &[(isize, isize)]) -> Self {
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
                    *d = (*d as isize + left + right) as usize;
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

    #[must_use]
    pub fn permute(&self, axes: &[usize]) -> Self {
        //std::println!("{:?}\n{:?}", self, axes);
        let mut views = self.views.clone();
        views[0].shape = views[0].shape.permute(axes);
        views[0].strides = views[0].strides.permute(axes);
        let padding = &views[0].padding;
        let padding = axes.iter().map(|axis| padding[*axis]).collect();
        views[0].padding = padding;
        Self { views }
    }

    #[must_use]
    pub fn reshape(&self, n_shape: &[usize]) -> Self {
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
        if views.first().unwrap().is_contiguous() {
            views[0] = InnerView {
                shape: n_shape.clone(),
                strides: n_shape.strides(),
                padding: core::iter::repeat((0, 0)).take(n_shape.rank()).collect(),
            };
        } else {
            let shape = self.shape();
            if n_shape.rank() > shape.rank()
                && n_shape
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
}

fn shape_to_strides(shape: &[usize]) -> Vec<usize> {
    todo!()
}
