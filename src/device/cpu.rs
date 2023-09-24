// This is very unoptimized but simple CPU version
// Every backend needs to consits of two parts - work manager inside realize function
// and kenrnels enqueued by this manager.

extern crate alloc;
use crate::{
    device::Storage,
    node_id::NodeId,
    axes::Axes,
    dtype::DType,
    graph::Node,
    libm::{expf, logf, powf, tanhf},
    shape::{Shape, Strides},
    OutOfMemoryError,
};
use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};

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

#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
pub(super) fn realize(
    graph: &mut BTreeMap<NodeId, (usize, Node)>, // id, refcount and Node
    order: &[NodeId],
    _nodes: &BTreeSet<NodeId>,
) -> Result<(), OutOfMemoryError> {
    for node_id in order {
        let node = &graph.get(node_id).unwrap().1;
        let res = match node {
            Node::None |
            Node::Leaf |
            Node::Const(..) => Storage::None,
            Node::StoreF32(data, shape) => Storage::CPUF32(data.clone(), shape.clone()),
            Node::StoreI32(data, shape) => Storage::CPUI32(data.clone(), shape.clone()),
            Node::Add(x, y) => binary_op(Shape::default(), graph.c(*x), graph.c(*y), "+"),
            Node::Sub(x, y) => binary_op(Shape::default(), graph.c(*x), graph.c(*y), "-"),
            Node::Mul(x, y) => binary_op(Shape::default(), graph.c(*x), graph.c(*y), "*"),
            Node::Div(x, y) => binary_op(Shape::default(), graph.c(*x), graph.c(*y), "/"),
            Node::Pow(x, y) => binary_op(Shape::default(), graph.c(*x), graph.c(*y), "pow"),
            Node::TDot(x, y, shape) => binary_op(shape.clone(), graph.c(*x), graph.c(*y), "tdot"),
            Node::Neg(x) => unary_op(graph.c(*x), "neg"),
            Node::ReLU(x) => unary_op(graph.c(*x), "relu"),
            Node::DReLU(x) => unary_op(graph.c(*x), "drelu"),
            Node::Exp(x) => unary_op(graph.c(*x), "exp"),
            Node::Ln(x) => unary_op(graph.c(*x), "ln"),
            Node::Tanh(x) => unary_op(graph.c(*x), "tanh"),
            Node::Cast(x, dtype) => match graph.c(*x) {
                Storage::CPUF32(data, shape) => match dtype {
                    DType::F32 => Storage::CPUF32(data.clone(), shape.clone()),
                    DType::I32 => {
                        Storage::CPUI32(data.iter().map(|x| *x as i32).collect(), shape.clone())
                    }
                },
                Storage::CPUI32(data, shape) => match dtype {
                    DType::F32 => {
                        Storage::CPUF32(data.iter().map(|x| *x as f32).collect(), shape.clone())
                    }
                    DType::I32 => Storage::CPUI32(data.clone(), shape.clone()),
                },
                _ => todo!(),
            },
            Node::Expand(x, eshape) => match graph.c(*x) {
                Storage::CPUF32(data, shape) => {
                    Storage::CPUF32(expand_op_t(data, shape, eshape), eshape.clone())
                }
                Storage::CPUI32(data, shape) => {
                    Storage::CPUI32(expand_op_t(data, shape, eshape), eshape.clone())
                }
                _ => panic!(),
            },
            Node::Reshape(x, shape) => match graph.c(*x) {
                Storage::CPUF32(data, _) => Storage::CPUF32(data.clone(), shape.clone()),
                Storage::CPUI32(data, _) => Storage::CPUI32(data.clone(), shape.clone()),
                _ => panic!(),
            },
            Node::Permute(x, axes, shape) => match graph.c(*x) {
                Storage::CPUF32(data, xshape) => {
                    Storage::CPUF32(permute_op_t(xshape, data, axes), shape.clone())
                }
                Storage::CPUI32(data, xshape) => {
                    Storage::CPUI32(permute_op_t(xshape, data, axes), shape.clone())
                }
                _ => panic!(),
            },
            Node::Max(x, axes, _) => axes_op(graph.c(*x), "max", axes),
            Node::Sum(x, axes, _) => axes_op(graph.c(*x), "sum", axes),
        };
        let parameters = node.parameters();
        if !matches!(res, Storage::None) {
            graph.get_mut(node_id).unwrap().1 = Node::Const(res);
        }
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

// Simple and very slow cpu kernels
fn unary_op(data: &Storage, op: &str) -> Storage {
    match data {
        Storage::CPUF32(data, shape) => match op {
            "" => Storage::CPUF32(data.clone(), shape.clone()),
            "neg" => Storage::CPUF32(unary_op_t(data, |x| -x), shape.clone()),
            "relu" => Storage::CPUF32(unary_op_t(data, |x| x.max(0.)), shape.clone()),
            "drelu" => Storage::CPUF32(unary_op_t(data, |x| if *x > 0. { 1. } else { 0. }), shape.clone()),
            "exp" => Storage::CPUF32(unary_op_t(data, |x| expf(*x)), shape.clone()),
            "ln" => Storage::CPUF32(unary_op_t(data, |x| logf(*x)), shape.clone()),
            "tanh" => Storage::CPUF32(unary_op_t(data, |x| tanhf(*x)), shape.clone()),
            _ => panic!(),
        },
        Storage::CPUI32(data, shape) => match op {
            "" => Storage::CPUI32(data.clone(), shape.clone()),
            "neg" => Storage::CPUI32(unary_op_t(data, |x| -x), shape.clone()),
            "relu" => Storage::CPUI32(unary_op_t(data, |x| (*x).max(0)), shape.clone()),
            "drelu" => Storage::CPUI32(unary_op_t(data, |x| if *x > 0 { 1 } else { 0 }), shape.clone()),
            _ => panic!("Impossible op {op} on i32"),
        },
        _ => panic!(),
    }
}

fn unary_op_t<T: Sync + Send>(data: &[T], op: impl Sync + Send + Fn(&T) -> T) -> Box<[T]> {
    #[cfg(not(feature = "cpu"))]
    {
        data.iter().map(op).collect()
    }
    #[cfg(feature = "cpu")]
    {
        use rayon::prelude::*;
        data.par_iter().map(op).collect::<Vec<T>>().into()
    }
}

#[allow(clippy::cast_sign_loss)]
fn binary_op(shape: Shape, data_x: &Storage, data_y: &Storage, op: &str) -> Storage {
    match data_x {
        Storage::CPUF32(data_x, shape_x) => {
            if let Storage::CPUF32(data_y, shape_y) = data_y {
                match op {
                    "+" => Storage::CPUF32(
                        binary_op_t(data_x, shape_x, data_y, shape_y, |(x, y)| x + y),
                        shape_x.clone(),
                    ),
                    "-" => Storage::CPUF32(
                        binary_op_t(data_x, shape_x, data_y, shape_y, |(x, y)| x - y),
                        shape_x.clone(),
                    ),
                    "*" => Storage::CPUF32(
                        binary_op_t(data_x, shape_x, data_y, shape_y, |(x, y)| x * y),
                        shape_x.clone(),
                    ),
                    "/" => Storage::CPUF32(
                        binary_op_t(data_x, shape_x, data_y, shape_y, |(x, y)| x / y),
                        shape_x.clone(),
                    ),
                    "pow" => Storage::CPUF32(
                        binary_op_t(data_x, shape_x, data_y, shape_y, |(x, y)| powf(*x, *y)),
                        shape_x.clone(),
                    ),
                    "tdot" => {
                        #[cfg(not(feature = "cpu"))]
                        {
                            Storage::CPUF32(tdot_op_t(data_x, shape_x, data_y, shape_y), shape)
                        }
                        #[cfg(feature = "cpu")]
                        {
                            // TODO fix strides
                            let m: usize = shape_x[-1];
                            let k = if shape_x.rank() > 1 { shape_x[-2] } else { 1 };
                            let n: usize = shape_y[-1];
                            let mut data: Box<[f32]> = (0..shape.numel()).map(|_| 0.).collect();
                            let mut i = 0;
                            while i < shape.numel() / (m * k) {
                                unsafe {
                                    matrixmultiply::sgemm(
                                        m,
                                        k,
                                        n,
                                        1.,
                                        data_x.as_ptr().offset((i * k * m) as isize),
                                        1,
                                        m.try_into().unwrap(),
                                        data_y.as_ptr().offset((i * k * n) as isize),
                                        n.try_into().unwrap(),
                                        1,
                                        0.,
                                        data.as_mut_ptr().offset((i * m * n).try_into().unwrap()),
                                        n.try_into().unwrap(),
                                        1,
                                    );
                                }
                                i += 1;
                            }
                            Storage::CPUF32(data, shape)
                        }
                    }
                    _ => panic!(),
                }
            } else {
                panic!()
            }
        }
        Storage::CPUI32(data_x, shape_x) => {
            if let Storage::CPUI32(data_y, shape_y) = data_y {
                match op {
                    "+" => Storage::CPUI32(
                        binary_op_t(data_x, shape_x, data_y, shape_y, |(x, y)| x + y),
                        shape_x.clone(),
                    ),
                    "-" => Storage::CPUI32(
                        binary_op_t(data_x, shape_x, data_y, shape_y, |(x, y)| x - y),
                        shape_x.clone(),
                    ),
                    "*" => Storage::CPUI32(
                        binary_op_t(data_x, shape_x, data_y, shape_y, |(x, y)| x * y),
                        shape_x.clone(),
                    ),
                    "/" => Storage::CPUI32(
                        binary_op_t(data_x, shape_x, data_y, shape_y, |(x, y)| x / y),
                        shape_x.clone(),
                    ),
                    "pow" => Storage::CPUI32(
                        binary_op_t(data_x, shape_x, data_y, shape_y, |(x, y)| x.pow(*y as u32)),
                        shape_x.clone(),
                    ),
                    "tdot" => Storage::CPUI32(tdot_op_t(data_x, shape_x, data_y, shape_y), shape),
                    _ => panic!(),
                }
            } else {
                panic!()
            }
        }
        _ => panic!(),
    }
}

fn binary_op_t<T: Sync + Send>(
    data_x: &[T],
    shape_x: &Shape,
    data_y: &[T],
    shape_y: &Shape,
    op: impl Fn((&T, &T)) -> T + Sync + Send,
) -> Box<[T]> {
    _ = shape_x;
    _ = shape_y;
    #[cfg(not(feature = "cpu"))]
    {
        data_x.iter().zip(data_y.iter()).map(op).collect()
    }
    #[cfg(feature = "cpu")]
    {
        use rayon::prelude::*;
        data_x.par_iter().zip(data_y.par_iter()).map(op).collect::<Vec<T>>().into()
    }
}

fn permute_op_t<T: Clone>(shape: &Shape, data: &[T], axes: &Axes) -> Box<[T]> {
    let rank = shape.rank();
    let strides = shape.strides();
    let pstrides = strides.permute(axes);
    //println!("{strides:?}, {pstrides:?}");
    let mut a = 1;
    let acc = Strides(
        shape
            .into_iter()
            .rev()
            .map(|d| {
                a *= d;
                a
            })
            .collect::<Box<[usize]>>()
            .iter()
            .copied()
            .rev()
            .collect(),
    )
    .permute(axes);
    let mut temp: Box<[(usize, usize)]> = (0..rank).map(|_| (0, 0)).collect();
    let mut clock: Box<[usize]> = (0..rank).map(|_| 0).collect();
    for k in 0..rank {
        temp[rank - k - 1] = (pstrides[k], acc[k]);
    }
    // clock is array of indices over each of dimensions. They are slowly increased by strides until it reaches dimension size stored in acc
    // then we increase index in higher dimension and we go over lower dimension again (clockwork)
    let mut i = 0;
    (0..shape.numel())
        .map(|_| {
            let res = data[i].clone();
            for (j, (st, acc)) in temp.iter().enumerate() {
                clock[j] += st;
                i += st;
                if clock[j] < *acc {
                    break;
                }
                i -= clock[j];
                clock[j] = 0;
            }
            res
        })
        .collect()
}

fn expand_op_t<T: Clone>(data: &[T], shape: &Shape, eshape: &Shape) -> Box<[T]> {
    let erank = eshape.rank();
    // Add ones before shape if it is shorter
    let shape: Shape = (0..erank - shape.rank())
        .map(|_| 1)
        .chain(shape.into_iter().copied())
        .collect::<Box<[usize]>>()
        .into();
    let eaxes: Box<[usize]> = eshape
        .into_iter()
        .zip(&shape)
        .enumerate()
        .filter_map(|(a, (x, y))| if x == y { None } else { Some(a) })
        .collect();
    let mut strides = shape.strides();
    let estrides = eshape.strides();
    for a in 0..erank {
        if eaxes.contains(&a) {
            strides.0[a] = 0;
        }
    }
    //println!("{shape:?}, {eaxes:?}, {strides:?}, {estrides:?}");
    (0..eshape.numel())
        .map(|i| {
            let mut rem = i;
            let mut idx = 0;
            for a in 0..erank {
                idx += rem / estrides[a] * strides[a];
                rem %= estrides[a];
            }
            //println!("{i} -> {idx}");
            data[idx].clone()
        })
        .collect()
}

fn tdot_op_t<T: Dtype>(data_x: &[T], shape_x: &Shape, data_y: &[T], shape_y: &Shape) -> Box<[T]> {
    // TODO this is super slow, because it does not use tiling for memory caching,
    // but its simple and works.
    // k, m @ k, n -> m, n
    let m = shape_x[-1];
    let k = if shape_x.rank() > 1 { shape_x[-2] } else { 1 };
    let n = shape_y[-1];
    #[cfg(feature = "debug1")]
    std::println!("{m}, {k}, {n}");
    const NUM: usize = 16;
    let transpose = |data: &[T], last_dim, n| {
        let mut res = Vec::with_capacity(n);
        let mut j = 0;
        while j < last_dim {
            let mut i = j;
            while i < n {
                res.push(data[i].clone());
                i += last_dim;
            }
            j += 1;
        }
        res
    };
    data_y
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
                            x.chunks(NUM)
                                .zip(y_row.chunks(NUM))
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
        .collect()
}

fn axes_op(data: &Storage, op: &str, axes: &Axes) -> Storage {
    match data {
        Storage::CPUF32(data, shape) => match op {
            "sum" => Storage::CPUF32(
                reduce_op_t(shape, data, axes, |x, y| x + y),
                shape.clone().reduce(axes),
            ),
            "max" => Storage::CPUF32(
                reduce_op_t(shape, data, axes, f32::max),
                shape.clone().reduce(axes),
            ),
            _ => panic!(),
        },
        Storage::CPUI32(data, shape) => match op {
            "sum" => Storage::CPUI32(
                reduce_op_t(shape, data, axes, |x, y| x + y),
                shape.clone().reduce(axes),
            ),
            "max" => Storage::CPUI32(
                reduce_op_t(shape, data, axes, core::cmp::Ord::max),
                shape.clone().reduce(axes),
            ),
            _ => panic!(),
        },
        _ => panic!(),
    }
}

fn reduce_op_t<T: Dtype>(
    shape: &Shape,
    data: &[T],
    axes: &Axes,
    op: impl Fn(T, T) -> T,
) -> Box<[T]> {
    // Strides of the input
    let strides = shape.strides();
    // indices of dimensions that are not reduced
    let included_dims: Box<[usize]> = (0..shape.rank()).filter(|x| !axes.contains(*x)).collect();
    // final resulting buffer
    let res_shape = shape.clone().reduce(axes);
    // Strides of the result
    let res_strides = res_shape.strides();
    let mut res: Box<[T]> = core::iter::repeat(T::zero())
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
    res
}

trait Dtype:
    Clone
    + core::fmt::Debug
    + core::fmt::Display
    + core::ops::Add<Output = Self>
    + core::ops::Mul<Output = Self>
    + Sync
    + Send
    + core::iter::Sum
{
    fn dtype() -> DType;
    fn zero() -> Self;
    //fn exp(self) -> Self;
}

impl Dtype for f32 {
    fn dtype() -> DType {
        DType::F32
    }

    fn zero() -> Self {
        0.
    }
}

impl Dtype for i32 {
    fn dtype() -> DType {
        DType::I32
    }

    fn zero() -> Self {
        0
    }
}
