use alloc::{
    collections::BTreeMap,
    vec::Vec,
};
use std::collections::BTreeSet;
use half::f16;
#[cfg(feature = "std")]
use rayon::prelude::*;
use zyx_core::dtype::DType;
use zyx_core::{
    axes::Axes, error::ZyxError, node::Node, runtime::RuntimeBackend, scalar::Scalar, shape::Shape,
    tensor::Id, view::View,
};

macro_rules! unary_op {
    ($ctx: expr, $x: expr, $nid: expr, $op: expr) => {{
        let (view, data) = $ctx.get($x);
        let data = match data {
            Data::F32(data) => Data::F32(unary(data, $op)),
            Data::F64(data) => Data::F64(unary(data, $op)),
            Data::I32(data) => Data::I32(unary(data, $op)),
        };
        $ctx.views.insert($nid, (view.clone(), $nid));
        $ctx.buffers.insert($nid, data);
    }};
}

fn unary<T: Scalar + Sync + Send, T2: Scalar + Send>(
    data: &[T],
    op: impl Fn(T) -> T2 + Sync + Send,
) -> Vec<T2> {
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
    ($ctx: expr, $x: expr, $y: expr, $nid: expr, $op: expr) => {{
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
        $ctx.views
            .insert($nid, (View::new(xview.shape().clone()), $nid));
        $ctx.buffers.insert($nid, data);
    }};
}

fn binary<XT: Scalar + Sync + Send, YT: Scalar + Sync + Send, T2: Scalar + Send>(
    xview: &View,
    xdata: &[XT],
    yview: &View,
    ydata: &[YT],
    op: impl Fn((XT, YT)) -> T2 + Sync + Send,
) -> Vec<T2> {
    /*Ok(match view.view_type() {
        ViewType::Contiguous => view.iterate_contiguous(data).collect(),
        ViewType::Strided => view.iterate_strided(data).collect(),
        ViewType::Reshaped => view.iterate_reshaped(data).collect(),
        ViewType::Padded => view.iterate_padded(data).collect(),
    })*/
    //#[cfg(not(feature = "std"))]
    //{
    // TODO parallel iterator and match on view_type()
    /*xview
        .iterate_padded(xdata)
        .zip(yview.iterate_padded(ydata))
        .map(op)
        .collect()*/
    //}
    //#[cfg(feature = "std")]
    //{
    //xview.iterate_padded(xdata).zip(yview.iterate_padded(ydata)).into_par_iter().map(op).collect()
    //}
    todo!()
}

fn terciary<
    XT: Scalar + Sync + Send,
    YT: Scalar + Sync + Send,
    ZT: Scalar + Sync + Send,
    T2: Scalar + Send,
>(
    xview: &View,
    xdata: &[XT],
    yview: &View,
    ydata: &[YT],
    zview: &View,
    zdata: &[ZT],
    op: impl Fn((XT, YT, ZT)) -> T2 + Sync + Send,
) -> Vec<T2> {
    // TODO parallel iterator and match on view_type()
    //#[cfg(not(feature = "std"))]
    //{
    //xview.iterate_padded(xdata).zip(yview.iterate_padded(ydata)).zip(zview.iterate_padded(zdata)).map(|((x, y), z)| (x, y, z)).map(op).collect()
    //}
    //#[cfg(feature = "std")]
    //{
    /*xview
        .iterate_padded(xdata)
        .zip(yview.iterate_padded(ydata))
        .zip(zview.iterate_padded(zdata))
        .map(|((x, y), z)| (x, y, z))
        .map(op)
        .collect()*/
    //}
    todo!()
}

#[derive(Debug)]
enum Data {
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
}

impl Data {
    unsafe fn as_type<T: Scalar>(&self) -> &[T] {
        match self {
            Data::F16(data) => core::mem::transmute(data.as_slice()),
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
    type CompiledGraph = ();

    fn evaluated_nodes(&self) -> BTreeSet<Id> {
        self.buffers.keys().copied().collect()
    }

    fn is_empty(&self, x: Id) -> bool {
        !(self.buffers.contains_key(&x) || self.views.contains_key(&x))
    }

    fn remove(&mut self, x: Id) -> Result<(), ZyxError> {
        //std::println!("Removing {x}");
        if let Some((_, id)) = self.views.remove(&x) {
            if !self.views.values().any(|(_, x)| *x == id) {
                self.buffers.remove(&id);
            }
        }
        //else {
        //let temp: Vec<_> = self.views.iter().map(|(id, x)| (id, x.1)).collect();
        //std::println!("Not removing {x}, because {temp:?}");
        //}
        //std::println!("Num buffers: {}", self.buffers.len());
        Ok(())
    }

    fn load<T: Scalar>(&mut self, x: Id, numel: usize) -> Result<Vec<T>, ZyxError> {
        /*let (view, id) = &self.views[&x];
        let data = unsafe { self.buffers[&id].as_type::<T>() };
        Ok(match view.view_type() {
            ViewType::Contiguous => view.iterate_contiguous(data).take(numel).collect(),
            ViewType::Strided => view.iterate_strided(data).take(numel).collect(),
            ViewType::Reshaped => view.iterate_reshaped(data).take(numel).collect(),
            ViewType::Padded => view.iterate_padded(data).take(numel).collect(),
        })*/
        todo!()
    }

    fn store<T: Scalar, IT>(&mut self, x: Id, iter: IT) -> Result<(), ZyxError>
    where
        IT: IntoIterator<Item = T>,
        IT::IntoIter: ExactSizeIterator,
    {
        //std::println!("CPU Storing {x}");
        /*let iter = iter.into_iter();
        self.views.insert(x, (View::new(iter.len().into()), x));
        self.buffers.insert(
            x,
            match T::dtype() {
                DType::F16 => Data::F16(iter.map(|x| x.into_f16()).collect()),
                DType::F32 => Data::F32(iter.map(|x| x.into_f32()).collect()),
                DType::F64 => Data::F64(iter.map(|x| x.into_f64()).collect()),
                DType::I32 => Data::I32(iter.map(|x| x.into_i32()).collect()),
            },
        );*/
        Ok(())
    }

    fn compile_graph(&mut self, rcs: &[u32], nodes: &[Node], to_eval: &BTreeSet<Id>) -> Result<Self::CompiledGraph, ZyxError> {
        todo!()
    }

    fn launch_graph(&mut self, graph: &Self::CompiledGraph) -> Result<(), ZyxError> {
        todo!()
    }
}

fn reduce_op<T: Scalar + Sync + Send>(
    view: &View,
    data: &[T],
    axes: &Axes,
    res_shape: &Shape,
    sum_reduce: bool,
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
    }
    .take(res_shape.numel())
    .collect();

    // Go over all data and apply sum function to correct values
    // then indices can be added just by making another vector and constantly
    // updating it (adding in case of sum) with new indices as new max/min are found
    /*if view.is_contiguous() {
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
    }*/
    res
}
