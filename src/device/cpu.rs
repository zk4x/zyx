extern crate alloc;
use alloc::{collections::{BTreeMap, BTreeSet}, sync::Arc, vec::Vec};
use alloc::vec;

use crate::{shape::{Strides, Shape}, OutOfMemoryError, node_id::NodeId, graph::Node, prelude::DType, axes::Axes};

use super::{Storage, Dtype};

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

#[derive(Debug)]
pub(super) struct CpuStorageIter<'a, T> {
    data: &'a [T],
    view: &'a View,
    idx: usize,
}

impl<T> CpuStorage<T> {
    pub(super) fn new(data: Arc<[T]>, shape: Shape) -> Self {
        CpuStorage {
            data: data.into(),
            view: View::new(shape),
        }
    }

    pub(super) fn shape(&self) -> &Shape {
        self.view.shape()
    }

    pub(super) fn numel(&self) -> usize {
        self.shape().numel()
    }

    pub(super) fn iter(&self) -> CpuStorageIter<'_, T> {
        CpuStorageIter {
            data: self.data.as_ref(),
            view: &self.view,
            idx: 0,
        }
    }

    fn expand(&self, shape: &Shape) -> CpuStorage<T> {
        CpuStorage {
            data: self.data.clone(),
            view: self.view.expand(shape)
        }
    }

    fn reshape(&self, shape: &Shape) -> CpuStorage<T> {
        CpuStorage {
            data: self.data.clone(),
            view: self.view.reshape(shape)
        }
    }

    fn permute(&self, axes: &Axes) -> CpuStorage<T> {
        CpuStorage {
            data: self.data.clone(),
            view: self.view.permute(axes)
        }
    }
}

impl<T: Copy + Send + Sync> CpuStorage<T> {
    fn unary_op(&self, op: impl Sync + Send + Fn(&T) -> T) -> CpuStorage<T> {
        #[cfg(not(feature = "cpu"))]
        {
            self.iter().map(op).collect()
        }
        #[cfg(feature = "cpu")]
        {
            use rayon::prelude::*;
            CpuStorage {
                data: self.data.par_iter().map(op).collect::<Arc<[T]>>(),
                view: self.view.clone(),
            }
        }
    }
}

impl<T: Clone> Iterator for CpuStorageIter<'_, T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == self.view.shape().numel() {
            None
        } else {
            let res = self.data[self.view.get_idx(self.idx)].clone();
            self.idx += 1;
            Some(res)
        }
    }
}

#[derive(Debug)]
pub(crate) struct CpuDev {}

#[derive(Debug, Clone)]
pub(super) struct View {
    shapes: Vec<(Shape, Strides)>,
    contiguous: bool,
}

impl View {
    fn new(shape: Shape) -> Self {
        Self {
            shapes: vec![(shape.clone(), shape.strides())],
            contiguous: true,
        }
    }

    fn get_idx(&self, mut idx: usize) -> usize {
        //std::println!("Idx {idx}, view: {:?}", self.shapes);
        if self.contiguous {
            return idx
        }
        for (shape, strides) in &self.shapes {
            let mut res = 0;
            for (d, st) in shape.into_iter().zip(strides).rev() {
                res += (idx%d)*st;
                idx /= d;
            }
            idx = res;
            //std::println!("{idx}");
        }
        idx
    }

    fn shape(&self) -> &Shape  {
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
        Self {}
    }

    pub(super) fn realize(
        &mut self,
        graph: &mut BTreeMap<NodeId, (usize, Node)>, // id, refcount and Node
        order: &[NodeId],                            // recommended realization order
        _nodes: &BTreeSet<NodeId>,                    // which nodes need to be realized
        ) -> Result<(), OutOfMemoryError> {
    'a: for node_id in order {
        let node = &graph.get(node_id).unwrap().1;
        match node {
            Node::None |
            Node::Leaf |
            Node::Const(..) => continue 'a,
            _ => {}
        }
        let res = match node {
            Node::None |
            Node::Leaf |
            Node::Const(..) => panic!(),
            Node::StoreF32(data, shape) => Storage::CPUF32(CpuStorage::new(data.clone().into(), shape.clone())),
            Node::StoreI32(data, shape) => Storage::CPUI32(CpuStorage::new(data.clone().into(), shape.clone())),
            Node::Add(x, y) => binary_op(Shape::default(), graph.c(*x), graph.c(*y), "+"),
            Node::Sub(x, y) => binary_op(Shape::default(), graph.c(*x), graph.c(*y), "-"),
            Node::Mul(x, y) => binary_op(Shape::default(), graph.c(*x), graph.c(*y), "*"),
            Node::Div(x, y) => binary_op(Shape::default(), graph.c(*x), graph.c(*y), "/"),
            Node::Cmplt(x, y) => binary_op(Shape::default(), graph.c(*x), graph.c(*y), "<"),
            Node::Pow(x, y) => binary_op(Shape::default(), graph.c(*x), graph.c(*y), "pow"),
            Node::TDot(x, y, shape) => binary_op(shape.clone(), graph.c(*x), graph.c(*y), "tdot"),
            Node::Neg(x) => unary_op(graph.c(*x), "neg"),
            Node::ReLU(x) => unary_op(graph.c(*x), "relu"),
            Node::DReLU(x) => unary_op(graph.c(*x), "drelu"),
            Node::Exp(x) => unary_op(graph.c(*x), "exp"),
            Node::Ln(x) => unary_op(graph.c(*x), "ln"),
            Node::Sin(x) => unary_op(graph.c(*x), "sin"),
            Node::Cos(x) => unary_op(graph.c(*x), "cos"),
            Node::Sqrt(x) => unary_op(graph.c(*x), "sqrt"),
            Node::Tanh(x) => unary_op(graph.c(*x), "tanh"),
            Node::Dropout(x, seed, prob) => match graph.c(*x) {
                Storage::CPUF32(data) => {
                    Storage::CPUF32(dropout_op_t(&data, *seed, *prob))
                }
                Storage::CPUI32(data) => {
                    Storage::CPUI32(dropout_op_t(&data, *seed, *prob))
                }
                _ => panic!(),
            },
            Node::Cast(x, dtype) => match graph.c(*x) {
                Storage::CPUF32(data) => match dtype {
                    DType::F32 => Storage::CPUF32(CpuStorage::new(data.iter().collect(), data.shape().clone())),
                    DType::I32 => {
                        Storage::CPUI32(CpuStorage::new(data.iter().map(|x| x as i32).collect(), data.shape().clone()))
                    }
                },
                Storage::CPUI32(data) => match dtype {
                    DType::F32 => {
                        Storage::CPUF32(CpuStorage::new(data.iter().map(|x| x as f32).collect(), data.shape().clone()))
                    }
                    DType::I32 => Storage::CPUI32(CpuStorage::new(data.iter().collect(), data.shape().clone())),
                },
                _ => todo!(),
            },
            Node::Expand(x, eshape) => match graph.c(*x) {
                Storage::CPUF32(data) => Storage::CPUF32(data.expand(eshape)),
                Storage::CPUI32(data) => Storage::CPUI32(data.expand(eshape)),
                _ => panic!(),
            },
            Node::Reshape(x, shape) => match graph.c(*x) {
                Storage::CPUF32(data) => Storage::CPUF32(data.reshape(shape)),
                Storage::CPUI32(data) => Storage::CPUI32(data.reshape(shape)),
                _ => panic!(),
            },
            Node::Permute(x, axes, _) => match graph.c(*x) {
                Storage::CPUF32(data) => Storage::CPUF32(data.permute(axes)),
                Storage::CPUI32(data) => Storage::CPUI32(data.permute(axes)),
                _ => panic!(),
            },
            Node::Sum(x, axes, shape) => {
                match graph.c(*x) {
                    Storage::CPUF32(data) => Storage::CPUF32(reduce_op_t(data, axes, shape, |x, y| x + y)),
                    Storage::CPUI32(data) => Storage::CPUI32(reduce_op_t(data, axes, shape, |x, y| x + y)),
                    _ => panic!(),
                }
            }
            Node::Max(x, axes, shape) => {
                match graph.c(*x) {
                    Storage::CPUF32(data) => Storage::CPUF32(reduce_op_t(data, axes, shape, |x, y| x.max(y))),
                    Storage::CPUI32(data) => Storage::CPUI32(reduce_op_t(data, axes, shape, |x, y| x.max(y))),
                    _ => panic!(),
                }
            }
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
#[allow(clippy::match_wildcard_for_single_variants)]
fn unary_op(data: &Storage, op: &str) -> Storage {
    match data {
        Storage::CPUF32(data) => match op {
            "neg" => Storage::CPUF32(data.unary_op(|x| -x)),
            "relu" => Storage::CPUF32(data.unary_op(|x| x.max(0.))),
            "drelu" => Storage::CPUF32(data.unary_op(|x| f32::from(*x > 0.))),
            "exp" => Storage::CPUF32(data.unary_op(|x| libm::expf(*x))),
            "ln" => Storage::CPUF32(data.unary_op(|x| libm::logf(*x))),
            "sin" => Storage::CPUF32(data.unary_op(|x| libm::sinf(*x))),
            "cos" => Storage::CPUF32(data.unary_op(|x| libm::cosf(*x))),
            "sqrt" => Storage::CPUF32(data.unary_op(|x| libm::sqrtf(*x))),
            "tanh" => Storage::CPUF32(data.unary_op(|x| libm::tanhf(*x))),
            _ => panic!(),
        },
        Storage::CPUI32(data) => match op {
            "neg" => Storage::CPUI32(data.unary_op(|x| -x)),
            "relu" => Storage::CPUI32(data.unary_op(|x| (*x).max(0))),
            "drelu" => Storage::CPUI32(data.unary_op(|x| i32::from(*x > 0))),
            _ => panic!("Impossible op {op} on i32"),
        },
        _ => panic!(),
    }
}

#[allow(clippy::match_wildcard_for_single_variants)]
#[allow(clippy::cast_sign_loss)]
fn binary_op(shape: Shape, data_x: &Storage, data_y: &Storage, op: &str) -> Storage {
    match data_x {
        Storage::CPUF32(data_x) => {
            if let Storage::CPUF32(data_y) = data_y {
                match op {
                    "+" => Storage::CPUF32(
                        binary_op_t(data_x, data_y, |(x, y)| x + y),
                    ),
                    "-" => Storage::CPUF32(
                        binary_op_t(data_x, data_y, |(x, y)| x - y),
                    ),
                    "*" => Storage::CPUF32(
                        binary_op_t(data_x, data_y, |(x, y)| x * y),
                    ),
                    "/" => Storage::CPUF32(
                        binary_op_t(data_x, data_y, |(x, y)| x / y),
                    ),
                    "pow" => Storage::CPUF32(
                        binary_op_t(data_x, data_y, |(x, y)| libm::powf(x, y)),
                    ),
                    "tdot" => {
                        #[cfg(not(feature = "cpu"))]
                        {
                            Storage::CPUF32(tdot_op_t(data_x, data_y, shape))
                        }
                        #[cfg(feature = "cpu")]
                        {
                            // TODO fix strides
                            let data_x = data_x.unary_op(|x| *x);
                            let data_y = data_y.unary_op(|x| *x);
                            let m: usize = data_x.shape()[-1];
                            let k = if data_x.shape().rank() > 1 { data_x.shape()[-2] } else { 1 };
                            let n: usize = data_y.shape()[-1];
                            let data: Arc<[f32]> = (0..shape.numel()).map(|_| 0.).collect();
                            let mut i = 0;
                            while i < shape.numel() / (m * k) {
                                unsafe {
                                    matrixmultiply::sgemm(
                                        m,
                                        k,
                                        n,
                                        1.,
                                        data_x.data.as_ptr().add(i * k * m),
                                        1,
                                        m.try_into().unwrap(),
                                        data_y.data.as_ptr().add(i * k * n),
                                        n.try_into().unwrap(),
                                        1,
                                        0.,
                                        data.as_ptr().add(i * m * n) as *mut f32, // Hopefully this is OK
                                        n.try_into().unwrap(),
                                        1,
                                    );
                                }
                                i += 1;
                            }
                            Storage::CPUF32(CpuStorage::new(data, shape))
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
                    "+" => Storage::CPUI32(
                        binary_op_t(data_x, data_y, |(x, y)| x + y),
                    ),
                    "-" => Storage::CPUI32(
                        binary_op_t(data_x, data_y, |(x, y)| x - y),
                    ),
                    "*" => Storage::CPUI32(
                        binary_op_t(data_x, data_y, |(x, y)| x * y),
                    ),
                    "/" => Storage::CPUI32(
                        binary_op_t(data_x, data_y, |(x, y)| x / y),
                    ),
                    "pow" => Storage::CPUI32(
                        binary_op_t(data_x, data_y, |(x, y)| x.pow(y as u32)),
                    ),
                    "tdot" => {
                        Storage::CPUI32(tdot_op_t(&shape, data_x, data_y))
                    }
                    _ => panic!(),
                }
            } else {
                panic!()
            }
        }
        _ => panic!(),
    }
}

fn binary_op_t<T: Copy + Sync + Send>(
    data_x: &CpuStorage<T>,
    data_y: &CpuStorage<T>,
    op: impl Fn((T, T)) -> T + Sync + Send,
) -> CpuStorage<T> {
    #[cfg(not(feature = "cpu"))]
    {
        data_x.iter().zip(data_y.iter()).map(op).collect()
    }
    #[cfg(feature = "cpu")]
    {
        use rayon::prelude::*;
        CpuStorage::new((0..data_x.numel()).into_par_iter().map(|idx| (data_x.data[data_x.view.get_idx(idx)], data_y.data[data_y.view.get_idx(idx)])).map(op).collect::<Arc<[T]>>(), data_x.shape().clone())
    }
}

fn tdot_op_t<T: Dtype + Copy>(shape: &Shape, data_x: &CpuStorage<T>, data_y: &CpuStorage<T>) -> CpuStorage<T> {
    // TODO this is super slow, because it does not use tiling for memory caching,
    // but its simple and works.
    // k, m @ k, n -> m, n
    const WIDTH: usize = 16;
    let m = data_x.shape()[-1];
    let k = if data_x.shape().rank() > 1 { data_x.shape()[-2] } else { 1 };
    let n = data_y.shape()[-1];
    // TODO parallel iter
    let data_x: Vec<T> = data_x.iter().collect();
    let data_y: Vec<T> = data_y.iter().collect();
    let transpose = |data: &[T], last_dim, n| {
        (0..last_dim).map(|j| (j..n).step_by(last_dim).map(|i| data[i].clone())).flatten().collect::<Vec<T>>()
    };
    CpuStorage::new(data_y
        .chunks(k * n)
        .zip(data_x.chunks(k * m))
        .flat_map(|(y_chunk, x_chunk)| {
            transpose(
                &{
                let x_chunk = transpose(x_chunk, m, k*m);
                transpose(y_chunk, n, k*n)
                    .chunks(k)
                    .flat_map(|y_row| {
                        x_chunk.chunks(k).map(|x| {
                            x.chunks(WIDTH)
                                .zip(y_row.chunks(WIDTH))
                                .map(|(a, b)| {
                                    a.iter()
                                        .zip(b.iter())
                                        .map(|(a, b)| a.clone() * b.clone())
                                        .sum::<T>()
                                })
                                .sum()
                        })
                    })
                    .collect::<Vec<T>>()
                    }, m, n*m)
        })
        .collect(), shape.clone())
}

#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
fn dropout_op_t<T: Dtype>(data: &CpuStorage<T>, seed: u64, prob: f32) -> CpuStorage<T> {
    // TODO parallelize
    let (xr, yr) = (seed as u32, (seed >> 32) as u32);
    CpuStorage::new(data.iter().enumerate().map(|(i, x)| {
        let seed = xr + i as u32;
        let t = seed ^ (seed << 11);
        let r = yr ^ (yr >> 19) ^ (t ^ (t >> 8));
        if r > (u32::MAX as f32 * prob) as u32 {
            T::zero()
        } else {
            x.clone()
        }
    }).collect(), data.shape().clone())
}

fn reduce_op_t<T: Dtype>(
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
    for (i, x) in data.iter().enumerate() {
        // calculate index in result
        let mut j = 0;
        for dim in &*included_dims {
            j += ((i / strides[*dim]) % shape[*dim]) * res_strides[*dim]; // TODO this is quite a lot of calculations, do this with just adding and subtracting
        }
        // apply reduce function, in this case sum
        res[j] = op(res[j].clone(), x.clone());
    }
    CpuStorage::new(res.into(), res_shape.clone())
}
