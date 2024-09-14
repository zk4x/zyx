use std::rc::Rc;
use rand::{distributions::Uniform, Rng, SeedableRng};
use zyx::{DType, Scalar, Tensor, ZyxError};

//#[test]
fn fuzzy() -> Result<(), ZyxError> {
    let rand_seed = 21847091824098071;
    let max_tensors = 5;
    let max_numel = 256*256;
    let max_dims = 3;
    let num_nodes = 14;

    let mut rng = rand::rngs::SmallRng::seed_from_u64(rand_seed);
    let num_t = rng.gen_range(0..max_tensors);
    let mut tensors = Vec::new();
    let mut cpu_tensors = Vec::new();

    // Initialize a random number of tensors with random shapes and dtypes
    for _ in 0..num_t {
        let mut shape = Vec::new();
        for i in 0..rng.gen_range(1..max_dims) {
            let n = if i > 1 {
                max_numel / shape.iter().product::<usize>()
            } else {
                max_numel/10
            };
            if n > 1 {
                shape.insert(0, rng.gen_range(1..n));
            } else {
                break;
            }
        }
        //tensors.push(Tensor::randn(&shape, DType::F32));
        let numel = shape.iter().product();
        let r = Uniform::new(-100., 100.);
        let data: Vec<f32> = (0..numel).map(|_| rng.sample(&r)).collect();
        tensors.push(Tensor::from(&data).reshape(&shape));
        cpu_tensors.push(CPUTensor::new(&data).reshape(&shape));
    }

    for _ in 0..num_nodes {
        // Pick binary or unary op
        // pick a random tensor or two tensors for binary op
        // cast if necessary
        // apply that op
        // Assert that CPUTensor and zyx::Tensor give the same result
        let x = rng.gen_range(0..num_t);
        //let y = rng.gen_range(0..num_t);
        match rng.gen_range(0..10) {
            // Unary
            0 => {
                tensors[x] = tensors[x].relu();
                cpu_tensors[x] = cpu_tensors[x].relu();
            }
            1 => {
                tensors[x] = -&tensors[x];
                cpu_tensors[x] = cpu_tensors[x].neg();
            }
            2 => {
                tensors[x] = tensors[x].exp2();
                cpu_tensors[x] = cpu_tensors[x].exp2();
            }
            3 => {
                tensors[x] = tensors[x].log2();
                cpu_tensors[x] = cpu_tensors[x].log2();
            }
            4 => {
                tensors[x] = tensors[x].inv();
                cpu_tensors[x] = cpu_tensors[x].inv();
            }
            5 => {
                tensors[x] = tensors[x].sqrt();
                cpu_tensors[x] = cpu_tensors[x].sqrt();
            }
            6 => {
                tensors[x] = tensors[x].sin();
                cpu_tensors[x] = cpu_tensors[x].sin();
            }
            7 => {
                tensors[x] = tensors[x].cos();
                cpu_tensors[x] = cpu_tensors[x].cos();
            }
            8 => {
                tensors[x] = !&tensors[x];
                cpu_tensors[x] = cpu_tensors[x].not();
            }
            9 => {
                tensors[x] = tensors[x].nonzero();
                cpu_tensors[x] = cpu_tensors[x].nonzero();
            }
            // Binary
            10 => {
                //let t = rng.gen_range(0..num_t);
                //tensors[t] = tensors[t].exp2();
            }
            // Reduce
            //20 => tensors[x] = tensors[x].sum_kd([]),
            //20 => tensors[x] = tensors[x].max_kd([]),
            // Movement
            //20 => tensors[x] = tensors[x].reshape([]),
            _ => panic!(),
        }
    }
    Tensor::plot_graph([], "fuzzy_graph");
    Tensor::realize(&tensors)?;
    for (tensor, cpu_tensor) in tensors.iter().zip(cpu_tensors) {
        let data: Vec<f32> = tensor.clone().try_into()?;
        let cpu_data: Vec<f32> = cpu_tensor.to_vec();
        for (id, (x, y)) in data.iter().zip(cpu_data.iter()).enumerate() {
            assert_eq!(x, y, "Comparing tensor id {id}, x != y at {x} != {y}");
        }
    }

    Ok(())
}

// Just a very barebones and slow CPU tensor that is slow, but verifiably correct
// It's actually absurdly slow, so we may speed it up a bit perhaps
#[derive(Clone)]
pub struct CPUTensor {
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
    pub fn new<T: Scalar>(data: &[T]) -> CPUTensor {
        use std::mem::transmute as t;
        CPUTensor { view: View::new(&[data.len()]), data: match T::dtype() {
            #[cfg(feature = "half")]
            DType::BF16 => todo!(),
            #[cfg(feature = "half")]
            DType::F16 => todo!(),
            DType::F32 => Data::F32(unsafe { t::<_, &[f32]>(data) }.into()),
            DType::F64 => todo!(),
            #[cfg(feature = "complex")]
            DType::CF32 => todo!(),
            #[cfg(feature = "complex")]
            DType::CF64 => todo!(),
            DType::U8 => todo!(),
            DType::I8 => todo!(),
            DType::I16 => todo!(),
            DType::I32 => todo!(),
            DType::I64 => todo!(),
            DType::Bool => todo!(),
        } }
    }

    pub fn to_vec<T: Scalar>(&self) -> Vec<T> {
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
            Data::F64(data) => unsafe {
                t(self
                    .view
                    .iterate_padded(data)
                    .take(numel)
                    .collect::<Vec<f64>>())
            },
            Data::I32(data) => unsafe {
                t(self
                    .view
                    .iterate_padded(data)
                    .take(numel)
                    .collect::<Vec<i32>>())
            },
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.view.shape()
    }

    pub fn cast(&self, dtype: DType) -> CPUTensor {
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

    pub fn relu(&self) -> CPUTensor {
        unary_op!(self, Scalar::relu)
    }

    pub fn neg(&self) -> CPUTensor {
        unary_op!(self, Scalar::neg)
    }

    pub fn exp2(&self) -> CPUTensor {
        unary_op!(self, Scalar::exp2)
    }

    pub fn log2(&self) -> CPUTensor {
        unary_op!(self, Scalar::log2)
    }

    pub fn inv(&self) -> CPUTensor {
        unary_op!(self, Scalar::inv)
    }

    pub fn sqrt(&self) -> CPUTensor {
        unary_op!(self, Scalar::sqrt)
    }

    pub fn sin(&self) -> CPUTensor {
        unary_op!(self, Scalar::sin)
    }

    pub fn cos(&self) -> CPUTensor {
        unary_op!(self, Scalar::cos)
    }

    pub fn not(&self) -> CPUTensor {
        unary_op!(self, Scalar::not)
    }

    pub fn nonzero(&self) -> CPUTensor {
        unary_op!(self, Scalar::nonzero)
    }

    pub fn add(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::add)
    }

    pub fn sub(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::sub)
    }

    pub fn mul(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::mul)
    }

    pub fn div(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::div)
    }

    pub fn pow(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::pow)
    }

    pub fn cmplt(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::cmplt)
    }

    pub fn cmpgt(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::cmpgt)
    }

    pub fn max(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::max)
    }

    pub fn or(&self, other: &CPUTensor) -> CPUTensor {
        binary_op!(self, other, Scalar::or)
    }

    pub fn reduce_sum_kd(&self, axes: &[usize]) -> CPUTensor {
        CPUTensor {
            view: View::new(&self.shape()),
            data: match &self.data {
                Data::F32(data) => Data::F32(reduce_op(&self.view, data, axes, &self.shape().reduce(axes), true)),
                _ => todo!(),
            }
        }
    }

    pub fn reduce_max_kd(&self, axes: &[usize]) -> CPUTensor {
        CPUTensor {
            view: View::new(&self.shape()),
            data: match &self.data {
                Data::F32(data) => Data::F32(reduce_op(&self.view, data, axes, &self.shape().reduce(axes), false)),
                _ => todo!(),
            }
        }
    }

    pub fn pad(&self, padding: &[(isize, isize)]) -> CPUTensor {
        CPUTensor {
            view: self.view.pad(padding),
            data: self.data.clone(), // just rc clone
        }
    }

    pub fn permute(&self, axes: &[usize]) -> CPUTensor {
        CPUTensor {
            view: self.view.permute(axes),
            data: self.data.clone(), // just rc clone
        }
    }

    pub fn expand(&self, shape: &[usize]) -> CPUTensor {
        CPUTensor {
            view: self.view.expand(shape),
            data: self.data.clone(), // just rc clone
        }
    }

    pub fn reshape(&self, shape: &[usize]) -> CPUTensor {
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

/*impl Data {
    unsafe fn as_type<T: Scalar>(&self) -> &[T] {
        use std::mem::transmute as t;
        match self {
            Data::F32(data) => t::<&[f32], &[T]>(data.as_ref()),
            Data::F64(data) => t::<&[f64], &[T]>(data.as_ref()),
            Data::I32(data) => t::<&[i32], &[T]>(data.as_ref()),
        }
    }
}*/

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

fn reduce_op<T: Scalar>(
    view: &View,
    data: &[T],
    axes: &[usize],
    res_shape: &[usize],
    sum_reduce: bool,
) -> Rc<[T]> {
    // TODO parallelize this
    use std::boxed::Box;
    // Strides of the input
    let shape = view.shape();
    let strides = shape.strides();
    // indices of dimensions that are not reduced
    let included_dims: Box<[usize]> = (0..shape.len()).filter(|x| !axes.contains(x)).collect();
    // Strides of the result
    let res_strides = res_shape.strides();
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
    res.into()
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
        let shape: Vec<usize> = shape.into();
        let strides = shape.strides();
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
        views[0].shape = shape.into();
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
            "Can't reshape {:?} to {:?}",
            self.shape(),
            n_shape
        );
        let mut views = self.views.clone();
        // If we are reshaping InnerView that is contiguous, we just delete the last reshape
        if views.first().unwrap().is_contiguous() {
            views[0] = InnerView {
                shape: n_shape.into(),
                strides: n_shape.strides(),
                padding: core::iter::repeat((0, 0)).take(n_shape.len()).collect(),
            };
        } else {
            let shape = self.shape();
            if n_shape.len() > shape.len()
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
                    *shape = n_shape.into();
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
                    *padding = n_padding;
                }
            } else {
                // If there is no merge.
                views.insert(
                    0,
                    InnerView {
                        shape: n_shape.into(),
                        strides: n_shape.strides(),
                        padding: core::iter::repeat((0, 0)).take(n_shape.len()).collect(),
                    },
                );
            }
        }
        //std::println!("Merged into: {:?}", views);
        Self { views }
    }
}

trait Shape {
    fn as_slice(&self) -> &[usize];

    fn strides(&self) -> Vec<usize> {
        let mut stride = 1;
        let shape = self.as_slice();
        let mut res = Vec::with_capacity(shape.len());
        for d in shape.iter().rev() {
            res.push(stride);
            stride *= d;
        }
        res.reverse();
        res
    }

    fn expand_strides(&self, shape: &[usize], mut old_strides: Vec<usize>) -> Vec<usize> {
        let mut vec = self.as_slice().to_vec();
        while vec.len() < shape.len() {
            vec.insert(0, 1);
            old_strides = [0]
                .into_iter()
                .chain(old_strides.iter().copied())
                .collect();
        }
        vec
            .into_iter()
            .zip(shape)
            .zip(&old_strides)
            .map(|((od, nd), st)| if od == *nd { *st } else { 0 })
            .collect()
    }

    fn permute(&self, axes: &[usize]) -> Vec<usize> {
        let shape = self.as_slice();
        axes.iter().map(|axis| shape[*axis]).collect()
    }

    fn numel(&self) -> usize {
        self.as_slice().iter().product()
    }

    fn reduce(&self, axes: &[usize]) -> Vec<usize> {
         let mut shape: Vec<usize> = self.as_slice().into();
        for a in axes.iter() {
            shape[*a] = 1;
        }
        shape
    }
}

impl Shape for Vec<usize> {
    fn as_slice(&self) -> &[usize] {
        self
    }
}

impl Shape for &[usize] {
    fn as_slice(&self) -> &[usize] {
        self
    }
}
