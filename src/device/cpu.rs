extern crate alloc;
use alloc::vec;
use alloc::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    vec::Vec,
};

use crate::{
    axes::Axes,
    graph::Node,
    node_id::NodeId,
    prelude::DType,
    shape::{Shape, Strides},
    OutOfMemoryError,
};

use super::{Dtype, Storage};

trait GetConst {
    fn c(&self, i: NodeId) -> &Storage;
}

impl GetConst for BTreeMap<NodeId, (usize, Node)> {
    fn c(&self, i: NodeId) -> &Storage {
        if let Node::Const(storage) = &self.get(&i).unwrap().1 {
            storage
        } else {
            panic!()
        }
    }
}

#[derive(Debug)]
pub(crate) struct CpuStorage<T> {
    data: Arc<[T]>,
    view: View,
}

impl<T: Copy> CpuStorage<T> {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(super) fn at(&self, idx: usize) -> T {
        self.data[self.view.get_idx(idx)]
    }
}

impl<T> CpuStorage<T> {
    #[allow(clippy::needless_pass_by_value)]
    pub(super) fn new(data: Arc<[T]>, shape: Shape) -> Self {
        CpuStorage {
            data,
            view: View::new(shape),
        }
    }

    pub(super) fn shape(&self) -> &Shape {
        self.view.shape()
    }

    pub(super) fn numel(&self) -> usize {
        self.shape().numel()
    }

    fn expand(&self, shape: &Shape) -> CpuStorage<T> {
        CpuStorage {
            data: self.data.clone(),
            view: self.view.expand(shape),
        }
    }

    fn reshape(&self, shape: &Shape) -> CpuStorage<T> {
        CpuStorage {
            data: self.data.clone(),
            view: self.view.reshape(shape),
        }
    }

    fn permute(&self, axes: &Axes) -> CpuStorage<T> {
        CpuStorage {
            data: self.data.clone(),
            view: self.view.permute(axes),
        }
    }
}

impl<T: Copy + Send + Sync> CpuStorage<T> {
    fn unary_op<T2: Send>(
        &self,
        op: impl Sync + Send + Fn(T) -> T2,
        make_contiguous: bool,
    ) -> CpuStorage<T2> {
        #[cfg(not(feature = "cpu"))]
        {
            if make_contiguous && !self.view.contiguous {
                CpuStorage::new(
                    (0..self.shape().numel())
                        .map(|idx| op(self.at(idx)))
                        .collect::<Arc<[T2]>>(),
                    self.shape().clone(),
                )
            } else {
                CpuStorage {
                    data: self.data.iter().copied().map(op).collect::<Arc<[T2]>>(),
                    view: self.view.clone(),
                }
            }
        }
        #[cfg(feature = "cpu")]
        {
            use rayon::prelude::*;
            if make_contiguous && !self.view.contiguous {
                CpuStorage::new(
                    (0..self.shape().numel())
                        .map(|idx| op(self.at(idx)))
                        .collect::<Arc<[T2]>>(),
                    self.shape().clone(),
                )
            } else {
                CpuStorage {
                    data: self.data.par_iter().copied().map(op).collect::<Arc<[T2]>>(),
                    view: self.view.clone(),
                }
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct CpuDev;

#[derive(Debug, Clone)]
pub(super) struct View {
    shapes: Vec<(Shape, Strides)>,
    contiguous: bool,
}

impl View {
    #[allow(clippy::needless_pass_by_value)]
    fn new(shape: Shape) -> Self {
        Self {
            shapes: vec![(shape.clone(), shape.strides())],
            contiguous: true,
        }
    }

    fn get_idx(&self, mut idx: usize) -> usize {
        // TODO can this be faster???
        //std::println!("Idx {idx}, view: {:?}", self.shapes);
        for (shape, strides) in &self.shapes {
            let mut res = 0;
            for (d, st) in shape.into_iter().zip(strides).rev() {
                res += idx % d * st;
                idx /= d;
            }
            idx = res;
            //std::println!("{idx}");
        }
        idx
    }

    fn shape(&self) -> &Shape {
        &self.shapes[0].0
    }

    fn expand(&self, shape: &Shape) -> Self {
        let mut shapes = self.shapes.clone();
        //std::println!("Expanding {shapes:?}");
        shapes[0].1 = shapes[0].0.expand_strides(shape, shapes[0].1.clone());
        shapes[0].0 = shape.clone();
        //std::println!("To {shapes:?}");
        // TODO
        Self {
            shapes,
            contiguous: false,
        }
    }

    fn reshape(&self, shape: &Shape) -> Self {
        let mut shapes = self.shapes.clone();
        shapes.insert(0, (shape.clone(), shape.strides()));
        Self {
            shapes,
            contiguous: false,
        }
    }

    fn permute(&self, axes: &Axes) -> Self {
        let mut shapes = self.shapes.clone();
        shapes[0].0 = shapes[0].0.permute(axes);
        shapes[0].1 = shapes[0].1.permute(axes);
        Self {
            shapes,
            contiguous: false,
        }
    }
}

impl CpuDev {
    pub(crate) fn new() -> Self {
        Self
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::unnecessary_wraps)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_precision_loss)]
    pub(super) fn realize(
        &mut self,
        graph: &mut BTreeMap<NodeId, (usize, Node)>, // id, refcount and Node
        order: &[NodeId],                            // recommended realization order
        _nodes: &BTreeSet<NodeId>,                   // which nodes need to be realized
    ) -> Result<(), OutOfMemoryError> {
        'a: for node_id in order {
            let node = &graph.get(node_id).unwrap().1;
            match node {
                Node::None | Node::Leaf | Node::Const(..) => continue 'a,
                _ => {}
            }
            let res = match node {
                Node::None | Node::Leaf | Node::Const(..) => panic!(),
                Node::StoreF32(data, shape) => {
                    Storage::CPUF32(CpuStorage::new(data.clone().into(), shape.clone()))
                }
                Node::StoreI32(data, shape) => {
                    Storage::CPUI32(CpuStorage::new(data.clone().into(), shape.clone()))
                }
                Node::Add(x, y) => binary_op(Shape::default(), graph.c(*x), graph.c(*y), "+"),
                Node::Sub(x, y) => binary_op(Shape::default(), graph.c(*x), graph.c(*y), "-"),
                Node::Mul(x, y) => binary_op(Shape::default(), graph.c(*x), graph.c(*y), "*"),
                Node::Div(x, y) => binary_op(Shape::default(), graph.c(*x), graph.c(*y), "/"),
                Node::Cmplt(x, y) => binary_op(Shape::default(), graph.c(*x), graph.c(*y), "<"),
                Node::Pow(x, y) => binary_op(Shape::default(), graph.c(*x), graph.c(*y), "pow"),
                Node::TDot(x, y, shape) => {
                    binary_op(shape.clone(), graph.c(*x), graph.c(*y), "tdot")
                }
                Node::Neg(x) => unary_op(graph.c(*x), "neg"),
                Node::ReLU(x) => unary_op(graph.c(*x), "relu"),
                Node::Exp(x) => unary_op(graph.c(*x), "exp"),
                Node::Ln(x) => unary_op(graph.c(*x), "ln"),
                Node::Sin(x) => unary_op(graph.c(*x), "sin"),
                Node::Cos(x) => unary_op(graph.c(*x), "cos"),
                Node::Sqrt(x) => unary_op(graph.c(*x), "sqrt"),
                Node::Tanh(x) => unary_op(graph.c(*x), "tanh"),
                Node::Dropout(x, seed, prob) => match graph.c(*x) {
                    Storage::CPUF32(data) => Storage::CPUF32(dropout_op_t(data, *seed, *prob)),
                    Storage::CPUI32(data) => Storage::CPUI32(dropout_op_t(data, *seed, *prob)),
                    #[cfg(any(feature = "opencl", feature = "torch"))]
                    _ => panic!(),
                },
                Node::Cast(x, dtype) => match graph.c(*x) {
                    Storage::CPUF32(data) => match dtype {
                        DType::F32 => Storage::CPUF32(data.unary_op(|x| x, false)),
                        DType::I32 => Storage::CPUI32(data.unary_op(|x| x as i32, false)),
                    },
                    Storage::CPUI32(data) => match dtype {
                        DType::F32 => Storage::CPUF32(data.unary_op(|x| x as f32, false)),
                        DType::I32 => Storage::CPUI32(data.unary_op(|x| x, false)),
                    },
                    #[cfg(any(feature = "opencl", feature = "torch"))]
                    _ => todo!(),
                },
                Node::Expand(x, eshape) => match graph.c(*x) {
                    Storage::CPUF32(data) => Storage::CPUF32(data.expand(eshape)),
                    Storage::CPUI32(data) => Storage::CPUI32(data.expand(eshape)),
                    #[cfg(any(feature = "opencl", feature = "torch"))]
                    _ => panic!(),
                },
                Node::Reshape(x, shape) => match graph.c(*x) {
                    Storage::CPUF32(data) => Storage::CPUF32(data.reshape(shape)),
                    Storage::CPUI32(data) => Storage::CPUI32(data.reshape(shape)),
                    #[cfg(any(feature = "opencl", feature = "torch"))]
                    _ => panic!(),
                },
                Node::Permute(x, axes, _) => match graph.c(*x) {
                    Storage::CPUF32(data) => Storage::CPUF32(data.permute(axes)),
                    Storage::CPUI32(data) => Storage::CPUI32(data.permute(axes)),
                    #[cfg(any(feature = "opencl", feature = "torch"))]
                    _ => panic!(),
                },
                Node::Sum(x, axes, shape) => match graph.c(*x) {
                    Storage::CPUF32(data) => {
                        Storage::CPUF32(reduce_op_t(data, axes, shape, |x, y| x + y))
                    }
                    Storage::CPUI32(data) => {
                        Storage::CPUI32(reduce_op_t(data, axes, shape, |x, y| x + y))
                    }
                    #[cfg(any(feature = "opencl", feature = "torch"))]
                    _ => panic!(),
                },
                Node::Max(x, axes, shape) => match graph.c(*x) {
                    Storage::CPUF32(data) => {
                        Storage::CPUF32(reduce_op_t(data, axes, shape, f32::max))
                    }
                    Storage::CPUI32(data) => {
                        Storage::CPUI32(reduce_op_t(data, axes, shape, core::cmp::Ord::max))
                    }
                    #[cfg(any(feature = "opencl", feature = "torch"))]
                    _ => panic!(),
                },
            };
            let parameters = node.parameters();
            graph.get_mut(node_id).unwrap().1 = Node::Const(res);
            for parameter in &*parameters {
                let val = graph.get_mut(parameter).unwrap();
                val.0 -= 1;
                if val.0 == 0 {
                    val.1 = Node::None;
                }
            }
        }
        Ok(())
    }
}

// Simple and very slow cpu kernels
fn unary_op(data: &Storage, op: &str) -> Storage {
    match data {
        Storage::CPUF32(data) => match op {
            "neg" => Storage::CPUF32(data.unary_op(|x| -x, false)),
            "relu" => Storage::CPUF32(data.unary_op(|x| x.max(0.), false)),
            "drelu" => Storage::CPUF32(data.unary_op(|x| f32::from(x > 0.), false)),
            "exp" => Storage::CPUF32(data.unary_op(libm::expf, false)),
            "ln" => Storage::CPUF32(data.unary_op(libm::logf, false)),
            "sin" => Storage::CPUF32(data.unary_op(libm::sinf, false)),
            "cos" => Storage::CPUF32(data.unary_op(libm::cosf, false)),
            "sqrt" => Storage::CPUF32(data.unary_op(libm::sqrtf, false)),
            "tanh" => Storage::CPUF32(data.unary_op(libm::tanhf, false)),
            _ => panic!(),
        },
        Storage::CPUI32(data) => match op {
            "neg" => Storage::CPUI32(data.unary_op(|x| -x, false)),
            "relu" => Storage::CPUI32(data.unary_op(|x| x.max(0), false)),
            "drelu" => Storage::CPUI32(data.unary_op(|x| i32::from(x > 0), false)),
            _ => panic!("Impossible op {op} on i32"),
        },
        #[cfg(any(feature = "opencl", feature = "torch"))]
        _ => panic!(),
    }
}

#[allow(clippy::match_wildcard_for_single_variants)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::needless_pass_by_value)]
fn binary_op(shape: Shape, data_x: &Storage, data_y: &Storage, op: &str) -> Storage {
    match data_x {
        Storage::CPUF32(data_x) => {
            if let Storage::CPUF32(data_y) = data_y {
                match op {
                    "+" => Storage::CPUF32(binary_op_t(data_x, data_y, |(x, y)| x + y)),
                    "-" => Storage::CPUF32(binary_op_t(data_x, data_y, |(x, y)| x - y)),
                    "*" => Storage::CPUF32(binary_op_t(data_x, data_y, |(x, y)| x * y)),
                    "/" => Storage::CPUF32(binary_op_t(data_x, data_y, |(x, y)| x / y)),
                    "<" => Storage::CPUF32(binary_op_t(data_x, data_y, |(x, y)| {
                        i8::from(x < y).into()
                    })),
                    "pow" => {
                        Storage::CPUF32(binary_op_t(data_x, data_y, |(x, y)| libm::powf(x, y)))
                    }
                    "tdot" => {
                        // k, m @ k, n -> m, n
                        #[cfg(not(feature = "cpu"))]
                        {
                            Storage::CPUF32(tdot_op_t(&shape, data_x, data_y))
                        }
                        #[cfg(feature = "cpu")]
                        {
                            // TODO make use of strides in data.view and contiguous storage
                            //use crate::axes::IntoAxes;
                            //let data_x = data_x.unary_op(|x| x, true);
                            //let data_y = data_y.unary_op(|x| x, true);
                            let m: usize = data_x.shape()[-1];
                            let k = if data_x.shape().rank() > 1 {
                                data_x.shape()[-2]
                            } else {
                                1
                            };
                            let n: usize = data_y.shape()[-1];
                            let xr = data_x.shape().rank();
                            let yr = data_y.shape().rank();
                            //std::println!("{:?}", data_x);
                            //std::println!("{:?}", data_y);
                            //let mut data: Vec<f32> = (0..shape.numel()).map(|_| 0.).collect();
                            let data: Arc<[f32]> = (0..shape.numel()).map(|_| 0.).collect();
                            let mut i = 0;
                            while i < shape.numel() / (m * k) {
                                unsafe {
                                    gemm::gemm(
                                        m,
                                        n,
                                        k,
                                        //data.as_mut_ptr().add(i * m * n),
                                        data.as_ptr().add(i * m * n).cast_mut(),
                                        1,
                                        n.try_into().unwrap(),
                                        false,
                                        data_x.data.as_ptr().add(i * k * m),
                                        data_x.view.shapes[0].1[xr - 1] as isize,
                                        data_x.view.shapes[0].1[xr - 2] as isize,
                                        data_y.data.as_ptr().add(i * k * n),
                                        data_y.view.shapes[0].1[yr - 1] as isize,
                                        data_y.view.shapes[0].1[yr - 2] as isize,
                                        1.,
                                        1.,
                                        false,
                                        false,
                                        false,
                                        gemm::Parallelism::Rayon(rayon::current_num_threads()),
                                    );
                                }
                                i += 1;
                            }
                            Storage::CPUF32(CpuStorage::new(data.into(), shape))
                        }
                    }
                    _ => panic!(),
                }
            } else {
                panic!()
            }
        }
        Storage::CPUI32(data_x) => {
            if let Storage::CPUI32(data_y) = data_y {
                match op {
                    "+" => Storage::CPUI32(binary_op_t(data_x, data_y, |(x, y)| x + y)),
                    "-" => Storage::CPUI32(binary_op_t(data_x, data_y, |(x, y)| x - y)),
                    "*" => Storage::CPUI32(binary_op_t(data_x, data_y, |(x, y)| x * y)),
                    "/" => Storage::CPUI32(binary_op_t(data_x, data_y, |(x, y)| x / y)),
                    "<" => Storage::CPUI32(binary_op_t(data_x, data_y, |(x, y)| i32::from(x < y))),
                    "pow" => Storage::CPUI32(binary_op_t(data_x, data_y, |(x, y)| x.pow(y as u32))),
                    "tdot" => Storage::CPUI32(tdot_op_t(&shape, data_x, data_y)),
                    _ => panic!(),
                }
            } else {
                panic!()
            }
        }
        #[cfg(any(feature = "opencl", feature = "torch"))]
        _ => panic!(),
    }
}

fn binary_op_t<T: Copy + Sync + Send>(
    data_x: &CpuStorage<T>,
    data_y: &CpuStorage<T>,
    op: impl Fn((T, T)) -> T + Sync + Send,
) -> CpuStorage<T> {
    #[cfg(not(feature = "cpu"))]
    let data = {
        match (data_x.view.contiguous, data_y.view.contiguous) {
            (true, true) => (0..data_x.numel()).map(|idx| ( data_x.data[idx], data_y.data[idx])).map(op).collect(),
            (true, false) => (0..data_x.numel()).map(|idx| ( data_x.data[idx], data_y.at(idx))).map(op).collect(),
            (false, true) => (0..data_x.numel()).map(|idx| ( data_x.at(idx), data_y.data[idx])).map(op).collect(),
            (false, false) => (0..data_x.numel()).map(|idx| ( data_x.at(idx), data_y.at(idx))).map(op).collect(),
        }
    };
    #[cfg(feature = "cpu")]
    let data = {
        use rayon::prelude::*;
        match (data_x.view.contiguous, data_y.view.contiguous) {
            (true, true) => (0..data_x.numel()).into_par_iter().map(|idx| ( data_x.data[idx], data_y.data[idx])).map(op).collect(),
            (true, false) => (0..data_x.numel()).into_par_iter().map(|idx| ( data_x.data[idx], data_y.at(idx))).map(op).collect(),
            (false, true) => (0..data_x.numel()).into_par_iter().map(|idx| ( data_x.at(idx), data_y.data[idx])).map(op).collect(),
            (false, false) => (0..data_x.numel()).into_par_iter().map(|idx| ( data_x.at(idx), data_y.at(idx))).map(op).collect(),
        }
    };
    CpuStorage::new(
        data,
        data_x.shape().clone(),
    )
}

fn tdot_op_t<T: Dtype + Copy>(
    shape: &Shape,
    data_x: &CpuStorage<T>,
    data_y: &CpuStorage<T>,
) -> CpuStorage<T> {
    // TODO this is super slow, because it does not use tiling for memory caching,
    // but its simple and works.
    // k, m @ k, n -> m, n
    const WIDTH: usize = 16;
    let m = data_x.shape()[-1];
    let k = if data_x.shape().rank() > 1 {
        data_x.shape()[-2]
    } else {
        1
    };
    let n = data_y.shape()[-1];
    let data_x = data_x.unary_op(|x| x, true).data;
    let data_y = data_y.unary_op(|x| x, true).data;
    let transpose = |data: &[T], last_dim, n| {
        (0..last_dim)
            .flat_map(|j| (j..n).step_by(last_dim).map(|i| data[i]))
            .collect::<Vec<T>>()
    };
    CpuStorage::new(
        data_y
            .chunks(k * n)
            .zip(data_x.chunks(k * m))
            .flat_map(|(y_chunk, x_chunk)| {
                transpose(
                    &{
                        let x_chunk = transpose(x_chunk, m, k * m);
                        transpose(y_chunk, n, k * n)
                            .chunks(k)
                            .flat_map(|y_row| {
                                x_chunk.chunks(k).map(|x| {
                                    x.chunks(WIDTH)
                                        .zip(y_row.chunks(WIDTH))
                                        .map(|(a, b)| {
                                            a.iter().zip(b.iter()).map(|(a, b)| *a * *b).sum::<T>()
                                        })
                                        .sum()
                                })
                            })
                            .collect::<Vec<T>>()
                    },
                    m,
                    n * m,
                )
            })
            .collect(),
        shape.clone(),
    )
}

#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
fn dropout_op_t<T: Dtype + Copy>(data: &CpuStorage<T>, seed: u64, prob: f32) -> CpuStorage<T> {
    // TODO parallelize
    let (xr, yr) = (seed as u32, (seed >> 32) as u32);
    CpuStorage::new(
        (0..data.shape().numel())
            .map(|i| {
                let seed = xr + i as u32;
                let t = seed ^ (seed << 11);
                let r = yr ^ (yr >> 19) ^ (t ^ (t >> 8));
                if r > (u32::MAX as f32 * prob) as u32 {
                    T::zero()
                } else {
                    data.at(i)
                }
            })
            .collect(),
        data.shape().clone(),
    )
}

fn reduce_op_t<T: Dtype + Copy>(
    data: &CpuStorage<T>,
    axes: &Axes,
    res_shape: &Shape,
    op: impl Fn(T, T) -> T,
) -> CpuStorage<T> {
    use alloc::boxed::Box;
    // Strides of the input
    let shape = data.shape();
    let strides = shape.strides();
    // indices of dimensions that are not reduced
    let included_dims: Box<[usize]> = (0..shape.rank()).filter(|x| !axes.contains(*x)).collect();
    // Strides of the result
    let res_strides = res_shape.strides();
    let mut res: Vec<T> = core::iter::repeat(T::zero())
        .take(res_shape.numel())
        .collect();

    // Go over all data and apply sum function to correct values
    // then indices can be added just by making another vector and constantly
    // updating it (adding in case of sum) with new indices as new max/min are found
    if data.view.contiguous {
        for i in 0..data.shape().numel() {
            // calculate index in result
            let mut j = 0;
            for dim in &*included_dims {
                j += ((i / strides[*dim]) % shape[*dim]) * res_strides[*dim]; // TODO this is quite a lot of calculations, do this with just adding and subtracting
            }
            // apply reduce function, in this case sum
            res[j] = op(res[j], data.data[i]);
        }
    } else {
        for i in 0..data.shape().numel() {
            // calculate index in result
            let mut j = 0;
            for dim in &*included_dims {
                j += ((i / strides[*dim]) % shape[*dim]) * res_strides[*dim]; // TODO this is quite a lot of calculations, do this with just adding and subtracting
            }
            // apply reduce function, in this case sum
            res[j] = op(res[j], data.at(i));
        }
    }
    CpuStorage::new(res.into(), res_shape.clone())
}

/*#[test]
fn ras() {
    use crate::context::Context;
    let ctx = Context::new();
    //let ctx = Context::opencl().unwrap();
    let m = 2048;
    let x = ctx.randn((m, m));
    let y = ctx.randn((m, m));
    for _ in 0..100 {
        let mut z = x.t_dot(&y);
        z.realize().unwrap();
    }
}*/
