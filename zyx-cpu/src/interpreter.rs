use alloc::{collections::{BTreeSet, BTreeMap, btree_map::Entry}, vec::Vec};
use zyx_core::{
    error::ZyxError, node::Node, runtime::RuntimeBackend, scalar::Scalar, tensor::Id, view::View, shape::Shape, axes::Axes
};
#[cfg(feature = "std")]
use rayon::prelude::*;
//use std::println;

macro_rules! unary_op {
    ($ctx: expr, $x: expr, $nid: expr, $op: expr) => {
        {
            let (view, data) = $ctx.get($x);
            let data = match data {
                Data::F32(data) => Data::F32(unary(data, $op)),
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
    match (xview.contiguous(), yview.contiguous()) {
        (true, true) => {
            #[cfg(not(feature = "std"))]
            {
                xdata.iter().cloned().zip(ydata.iter().cloned()).map(op).collect()
            }
            #[cfg(feature = "std")]
            {
                xdata.par_iter().cloned().zip(ydata.par_iter().cloned()).map(op).collect()
            }
        }
        (true, false) => {
            #[cfg(not(feature = "std"))]
            {
                (0..xview.numel()).map(|i| op((xdata[i].clone(), ydata[xview.get_idx(i)].clone()))).collect()
            }
            #[cfg(feature = "std")]
            {
                (0..xview.numel()).into_par_iter().map(|i| op((xdata[i].clone(), ydata[xview.get_idx(i)].clone()))).collect()
            }
        }
        (false, true) => {
            #[cfg(not(feature = "std"))]
            {
                (0..xview.numel()).map(|i| op((xdata[xview.get_idx(i)].clone(), ydata[i].clone()))).collect()
            }
            #[cfg(feature = "std")]
            {
                (0..xview.numel()).into_par_iter().map(|i| op((xdata[xview.get_idx(i)].clone(), ydata[i].clone()))).collect()
            }
        }
        (false, false) => {
            #[cfg(not(feature = "std"))]
            {
                (0..xview.numel()).map(|i| op((xdata[xview.get_idx(i)].clone(), ydata[yview.get_idx(i)].clone()))).collect()
            }
            #[cfg(feature = "std")]
            {
                (0..xview.numel()).into_par_iter().map(|i| op((xdata[xview.get_idx(i)].clone(), ydata[yview.get_idx(i)].clone()))).collect()
            }
        }
    }
}

#[derive(Debug)]
enum Data {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

impl Data {
    unsafe fn as_type<T: Scalar>(&self) -> &[T] {
        match self {
            Data::F32(data) => core::mem::transmute(data.as_slice()),
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
        Ok(if view.contiguous() {
            data.to_vec()
        } else {
            (0..numel).map(|i| data[view.get_idx(i)].clone()).collect()
        })
    }

    fn evaluate(
        &mut self,
        _to_eval: BTreeSet<Id>,
        mut rcs: BTreeMap<Id, u8>,
        order: &[Id],
        nodes: &mut [Node],
    ) -> Result<(), ZyxError> {
        //println!("\nrcs: {rcs:?}");
        //println!("views: {:?}\n", self.views);
        //std::println!("Evaluating: {_to_eval:?}, rcs: {rcs:?}");
        //std::println!("Order: {order:?}");
        for nid in order.iter().copied() {
            match &mut nodes[nid.i()] {
                Node::LeafF32(..)
                | Node::LeafI32(..) => {}
                Node::IterF32(_, shape) => {
                    let mut new_node = Node::LeafF32(shape.clone());
                    self.views.insert(nid, (View::new(shape.clone()), nid));
                    core::mem::swap(&mut nodes[nid.i()], &mut new_node);
                    if let Node::IterF32(iter, _) = new_node {
                        self.buffers.insert(nid, Data::F32(iter.collect()));
                    }
                }
                Node::IterI32(_, shape) => {
                    let mut new_node = Node::LeafI32(shape.clone());
                    self.views.insert(nid, (View::new(shape.clone()), nid));
                    core::mem::swap(&mut nodes[nid.i()], &mut new_node);
                    if let Node::IterI32(iter, _) = new_node {
                        self.buffers.insert(nid, Data::I32(iter.collect()));
                    }
                }
                Node::UniformF32(..) => todo!(),
                Node::CastF32(x) => {
                    let (view, data) = self.get(x);
                    let data = match data {
                        Data::F32(data) => Data::F32(unary(data, Scalar::into_f32)),
                        Data::I32(data) => Data::F32(unary(data, Scalar::into_f32)),
                    };
                    self.views.insert(nid, (view.clone(), nid));
                    self.buffers.insert(nid, data);
                }
                Node::CastI32(x) => {
                    let (view, data) = self.get(x);
                    let data = match data {
                        Data::F32(data) => Data::I32(unary(data, Scalar::into_i32)),
                        Data::I32(data) => Data::I32(unary(data, Scalar::into_i32)),
                    };
                    self.views.insert(nid, (view.clone(), nid));
                    self.buffers.insert(nid, data);
                }
                Node::Neg(x) => unary_op!(self, x, nid, Scalar::neg),
                Node::ReLU(x) => unary_op!(self, x, nid, Scalar::relu),
                Node::Sin(x) => unary_op!(self, x, nid, Scalar::sin),
                Node::Cos(x) => unary_op!(self, x, nid, Scalar::cos),
                Node::Ln(x) => unary_op!(self, x, nid, Scalar::ln),
                Node::Exp(x) => unary_op!(self, x, nid, Scalar::exp),
                Node::Tanh(x) => unary_op!(self, x, nid, Scalar::tanh),
                Node::Sqrt(x) => unary_op!(self, x, nid, Scalar::sqrt),
                Node::Add(x, y) => binary_op!(self, x, y, nid, |(x, y)| x.add(y)),
                Node::Sub(x, y) => binary_op!(self, x, y, nid, |(x, y)| x.sub(y)),
                Node::Mul(x, y) => binary_op!(self, x, y, nid, |(x, y)| x.mul(y)),
                Node::Div(x, y) => binary_op!(self, x, y, nid, |(x, y)| x.div(y)),
                Node::Pow(x, y) => binary_op!(self, x, y, nid, |(x, y)| Scalar::pow(x, y)),
                Node::Cmplt(x, y) => binary_op!(self, x, y, nid, |(x, y)| x.cmplt(y)),
                Node::Reshape(x, sh) => {
                    let (view, id) = &self.views[x];
                    self.views.insert(nid, (view.reshape(sh), *id));
                    //println!("views: {:?}, buffers: {:?}", self.views, self.buffers);
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
                        Data::F32(data) => Data::F32(reduce_op(view, data, ax, sh, |(x, y)| x.add(y))),
                        Data::I32(data) => Data::I32(reduce_op(view, data, ax, sh, |(x, y)| x.add(y))),
                    };
                    self.views.insert(nid, (View::new(sh.clone()), nid));
                    self.buffers.insert(nid, data);
                }
                Node::Max(x, ax, sh) => {
                    let (view, data) = self.get(x);
                    let data = match data {
                        Data::F32(data) => Data::F32(reduce_op(view, data, ax, sh, |(x, y)| Scalar::max(x, y))),
                        Data::I32(data) => Data::I32(reduce_op(view, data, ax, sh, |(x, y)| Scalar::max(x, y))),
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
    op: impl Fn((T, T)) -> T,
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
    let mut res: Vec<T> = core::iter::repeat(T::zero())
        .take(res_shape.numel())
        .collect();

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
            res[j] = op((res[j].clone(), data[i].clone()));
        }
    } else {
        for i in 0..view.shape().numel() {
            // calculate index in result
            let mut j = 0;
            for dim in &*included_dims {
                j += ((i / strides[*dim]) % shape[*dim]) * res_strides[*dim]; // TODO this is quite a lot of calculations, do this with just adding and subtracting
            }
            // apply reduce function, in this case sum
            res[j] = op((res[j].clone(), data[view.get_idx(i)].clone()));
        }
    }
    res
}
