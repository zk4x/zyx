use alloc::{collections::{BTreeSet, BTreeMap, btree_map::Entry}, vec::Vec};
use zyx_core::{
    error::ZyxError, node::Node, runtime::RuntimeBackend, scalar::Scalar, tensor::Id, view::View, shape::Shape, axes::Axes
};
#[cfg(feature = "std")]
use rayon::prelude::*;
use zyx_core::dtype::DType;
use zyx_core::view::ViewType;

macro_rules! unary_op {
    ($ctx: expr, $x: expr, $nid: expr, $op: expr) => {
        {
            let (view, data) = $ctx.get($x);
            let data = match data {
                Data::F32(data) => Data::F32(unary(data, $op)),
                Data::F64(data) => Data::F64(unary(data, $op)),
                Data::I32(data) => Data::I32(unary(data, $op)),
            };
            $ctx.views.insert($nid, (view.clone(), $nid));
            $ctx.buffers.insert($nid, data);
        }
    }
}

fn unary<T: Scalar + Sync + Send, T2: Scalar + Send>(data: &[T], op: impl Fn(T) -> T2 + Sync + Send) -> Vec<T2> {
    #[cfg(not(feature = "std"))]
    {
        data.iter().cloned().map(op).collect()
    }
    #[cfg(feature = "std")]
    {
        data.par_iter().cloned().map(op).collect()
    }
}

macro_rules! binary_op {
    ($ctx: expr, $x: expr, $y: expr, $nid: expr, $op: expr) => {
        {
            let (xview, xdata) = $ctx.get($x);
            let (yview, ydata) = $ctx.get($y);
            let data = match xdata {
                Data::F32(xdata) => {
                    let Data::F32(ydata) = ydata else { panic!() };
                    Data::F32(binary(xview, xdata, yview, ydata, $op))
                }
                Data::F64(xdata) => {
                    let Data::F64(ydata) = ydata else { panic!() };
                    Data::F64(binary(xview, xdata, yview, ydata, $op))
                }
                Data::I32(xdata) => {
                    let Data::I32(ydata) = ydata else { panic!() };
                    Data::I32(binary(xview, xdata, yview, ydata, $op))
                }
            };
            $ctx.views.insert($nid, (View::new(xview.shape().clone()), $nid));
            $ctx.buffers.insert($nid, data);
        }
    }
}

fn binary<XT: Scalar + Sync + Send, YT: Scalar + Sync + Send, T2: Scalar + Send>(xview: &View, xdata: &[XT], yview: &View, ydata: &[YT], op: impl Fn((XT, YT)) -> T2 + Sync + Send) -> Vec<T2> {
    /*Ok(match view.view_type() {
        ViewType::Contiguous => view.iterate_contiguous(data).collect(),
        ViewType::Strided => view.iterate_strided(data).collect(),
        ViewType::Reshaped => view.iterate_reshaped(data).collect(),
        ViewType::Padded => view.iterate_padded(data).collect(),
    })*/
    //#[cfg(not(feature = "std"))]
    //{
    // TODO parallel iterator and match on view_type()
    xview.iterate_padded(xdata).zip(yview.iterate_padded(ydata)).map(op).collect()
    //}
    //#[cfg(feature = "std")]
    //{
        //xview.iterate_padded(xdata).zip(yview.iterate_padded(ydata)).into_par_iter().map(op).collect()
    //}
}

fn terciary<XT: Scalar + Sync + Send, YT: Scalar + Sync + Send, ZT: Scalar + Sync + Send, T2: Scalar + Send>(xview: &View, xdata: &[XT], yview: &View, ydata: &[YT], zview: &View, zdata: &[ZT], op: impl Fn((XT, YT, ZT)) -> T2 + Sync + Send) -> Vec<T2> {
    // TODO parallel iterator and match on view_type()
    //#[cfg(not(feature = "std"))]
    //{
        //xview.iterate_padded(xdata).zip(yview.iterate_padded(ydata)).zip(zview.iterate_padded(zdata)).map(|((x, y), z)| (x, y, z)).map(op).collect()
    //}
    //#[cfg(feature = "std")]
    //{
    xview.iterate_padded(xdata).zip(yview.iterate_padded(ydata)).zip(zview.iterate_padded(zdata)).map(|((x, y), z)| (x, y, z)).map(op).collect()
    //}
}

#[derive(Debug)]
enum Data {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
}

impl Data {
    unsafe fn as_type<T: Scalar>(&self) -> &[T] {
        match self {
            Data::F32(data) => core::mem::transmute(data.as_slice()),
            Data::F64(data) => core::mem::transmute(data.as_slice()),
            Data::I32(data) => core::mem::transmute(data.as_slice()),
        }
    }
}

pub struct Interpreter {
    buffers: BTreeMap<Id, Data>,
    views: BTreeMap<Id, (View, Id)>,
}

impl Interpreter {
    pub(crate) fn new() -> Self {
        Self {
            buffers: BTreeMap::new(),
            views: BTreeMap::new(),
        }
    }

    fn get(&self, x: &Id) -> (&View, &Data) {
        let (view, id) = &self.views[x];
        (view, &self.buffers[id])
    }
}

impl RuntimeBackend for Interpreter {
    fn is_evaluated(&self, x: Id) -> bool {
        self.views.contains_key(&x)
    }

    fn remove(&mut self, x: Id) -> Result<(), ZyxError> {
        self.views.remove(&x);
        if !self.views.values().any(|(_, id)| *id == x) {
            self.buffers.remove(&x);
        }
        Ok(())
    }

    fn load<T: Scalar>(&mut self, x: Id, numel: usize) -> Result<Vec<T>, ZyxError> {
        let (view, id) = &self.views[&x];
        let data = unsafe { self.buffers[&id].as_type::<T>() };
        Ok(match view.view_type() {
            ViewType::Contiguous => view.iterate_contiguous(data).take(numel).collect(),
            ViewType::Strided => view.iterate_strided(data).take(numel).collect(),
            ViewType::Reshaped => view.iterate_reshaped(data).take(numel).collect(),
            ViewType::Padded => view.iterate_padded(data).take(numel).collect(),
        })
    }

    fn store<T: Scalar, IT>(&mut self, x: Id, iter: IT) -> Result<(), ZyxError>
    where
        IT: IntoIterator<Item=T>,
        IT::IntoIter: ExactSizeIterator,
    {
        let iter = iter.into_iter();
        self.views.insert(x, (View::new(iter.len().into()), x));
        self.buffers.insert(x, match T::dtype() {
            DType::F32 => Data::F32(iter.map(|x| x.into_f32()).collect()),
            DType::F64 => Data::F64(iter.map(|x| x.into_f64()).collect()),
            DType::I32 => Data::I32(iter.map(|x| x.into_i32()).collect()),
        });
        Ok(())
    }

    fn evaluate(
        &mut self,
        _to_eval: BTreeSet<Id>,
        mut rcs: BTreeMap<Id, u32>,
        order: &[Id],
        nodes: &[Node],
    ) -> Result<(), ZyxError> {
        for nid in order.iter().copied() {
            match &nodes[nid.i()] {
                Node::Leaf(..) => {}
                Node::Uniform(..) => todo!(),
                Node::Cast(x, dtype) => {
                    let (view, data) = self.get(x);
                    let data = match data {
                        Data::F32(data) => match dtype {
                            DType::F32 => Data::F32(unary(data, Scalar::into_f32)),
                            DType::F64 => Data::F64(unary(data, Scalar::into_f64)),
                            DType::I32 => Data::I32(unary(data, Scalar::into_i32)),
                        }
                        Data::F64(data) => match dtype {
                            DType::F32 => Data::F32(unary(data, Scalar::into_f32)),
                            DType::F64 => Data::F64(unary(data, Scalar::into_f64)),
                            DType::I32 => Data::I32(unary(data, Scalar::into_i32)),
                        }
                        Data::I32(data) => match dtype {
                            DType::F32 => Data::F32(unary(data, Scalar::into_f32)),
                            DType::F64 => Data::F64(unary(data, Scalar::into_f64)),
                            DType::I32 => Data::I32(unary(data, Scalar::into_i32)),
                        }
                    };
                    self.views.insert(nid, (view.clone(), nid));
                    self.buffers.insert(nid, data);
                }
                Node::Detach(x) => unary_op!(self, x, nid, core::convert::identity),
                Node::Neg(x) => unary_op!(self, x, nid, Scalar::neg),
                Node::ReLU(x) => unary_op!(self, x, nid, Scalar::relu),
                Node::Sin(x) => unary_op!(self, x, nid, Scalar::sin),
                Node::Cos(x) => unary_op!(self, x, nid, Scalar::cos),
                Node::Ln(x) => unary_op!(self, x, nid, Scalar::ln),
                Node::Exp(x) => unary_op!(self, x, nid, Scalar::exp),
                Node::Tanh(x) => unary_op!(self, x, nid, Scalar::tanh),
                Node::Sqrt(x) => unary_op!(self, x, nid, Scalar::sqrt),
                Node::Add(x, y) => binary_op!(self, x, y, nid, |(x, y)| Scalar::add(x, y)),
                Node::Sub(x, y) => binary_op!(self, x, y, nid, |(x, y)| Scalar::sub(x, y)),
                Node::Mul(x, y) => binary_op!(self, x, y, nid, |(x, y)| Scalar::mul(x, y)),
                Node::Div(x, y) => binary_op!(self, x, y, nid, |(x, y)| Scalar::div(x, y)),
                Node::Pow(x, y) => binary_op!(self, x, y, nid, |(x, y)| Scalar::pow(x, y)),
                Node::Cmplt(x, y) => binary_op!(self, x, y, nid, |(x, y)| Scalar::cmplt(x, y)),
                Node::Where(x, y, z) => {
                    let (xview, xdata) = self.get(x);
                    let (yview, ydata) = self.get(y);
                    let (zview, zdata) = self.get(z);
                    let data = match xdata {
                        Data::F32(xdata) => {
                            let Data::F32(ydata) = ydata else { panic!() };
                            let Data::F32(zdata) = zdata else { panic!() };
                            Data::F32(terciary(xview, xdata, yview, ydata, zview, zdata, |(x, y, z)| if x != 0.0 { y } else { z }))
                        }
                        Data::F64(xdata) => {
                            let Data::F64(ydata) = ydata else { panic!() };
                            let Data::F64(zdata) = zdata else { panic!() };
                            Data::F64(terciary(xview, xdata, yview, ydata, zview, zdata, |(x, y, z)| if x != 0.0 { y } else { z }))
                        }
                        Data::I32(xdata) => {
                            let Data::I32(ydata) = ydata else { panic!() };
                            let Data::I32(zdata) = zdata else { panic!() };
                            Data::I32(terciary(xview, xdata, yview, ydata, zview, zdata, |(x, y, z)| if x != 0 { y } else { z }))
                        }
                    };
                    self.views.insert(nid, (View::new(xview.shape().clone()), nid));
                    self.buffers.insert(nid, data);
                }
                Node::Reshape(x, sh) => {
                    let (view, id) = &self.views[x];
                    self.views.insert(nid, (view.reshape(sh), *id));
                }
                Node::Expand(x, sh) => {
                    let (view, id) = &self.views[x];
                    self.views.insert(nid, (view.expand(sh), *id));
                }
                Node::Permute(x, ax, ..) => {
                    let (view, id) = &self.views[x];
                    self.views.insert(nid, (view.permute(ax), *id));
                }
                Node::Pad(x, padding, ..) => {
                    let (view, id) = &self.views[x];
                    self.views.insert(nid, (view.pad(padding), *id));
                }
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
            for p in nodes[nid.i()].parameters() {
                if let Entry::Occupied(e) = rcs.entry(p).and_modify(|rc| *rc -= 1) {
                    if *e.get() == 0 {
                        self.remove(p)?;
                    }
                }
            }
        }
        Ok(())
    }
}

fn reduce_op<T: Scalar + Sync + Send>(
    view: &View,
    data: &[T],
    axes: &Axes,
    res_shape: &Shape,
    sum_reduce: bool
) -> Vec<T> {
    // TODO parallelize this
    use alloc::boxed::Box;
    // Strides of the input
    let shape = view.shape();
    let strides = shape.strides();
    // indices of dimensions that are not reduced
    let included_dims: Box<[usize]> = (0..shape.rank()).filter(|x| !axes.contains(*x)).collect();
    // Strides of the result
    let res_strides = res_shape.strides();
    let mut res: Vec<T> = if sum_reduce {
        core::iter::repeat(T::zero())
    } else {
        core::iter::repeat(T::min_value())
    }.take(res_shape.numel()).collect();

    // Go over all data and apply sum function to correct values
    // then indices can be added just by making another vector and constantly
    // updating it (adding in case of sum) with new indices as new max/min are found
    if view.contiguous() {
        for i in 0..view.shape().numel() {
            // calculate index in result
            let mut j = 0;
            for dim in &*included_dims {
                j += ((i / strides[*dim]) % shape[*dim]) * res_strides[*dim]; // TODO this is quite a lot of calculations, do this with just adding and subtracting
            }
            // apply reduce function, in this case sum
            if sum_reduce {
                res[j] = Scalar::add(res[j].clone(), data[i].clone());
            } else {
                res[j] = Scalar::max(res[j].clone(), data[i].clone());
            }
        }
    } else {
        for (i, x) in view.iterate_padded(data).enumerate() {
            // calculate index in result
            let mut j = 0;
            for dim in &*included_dims {
                j += ((i / strides[*dim]) % shape[*dim]) * res_strides[*dim]; // TODO this is quite a lot of calculations, do this with just adding and subtracting
            }
            // apply reduce function, in this case sum
            if sum_reduce {
                res[j] = Scalar::add(res[j].clone(), x);
            } else {
                res[j] = Scalar::max(res[j].clone(), x);
            }
        }
    }
    res
}
