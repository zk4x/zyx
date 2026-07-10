// ----- ASYNC EVENT RULES -----
//
// host_to_pool is async: the host-side source buffer must stay valid
// until the returned event is synced via sync_events.
//
// Kernel launch is async: ALL buffers used by the kernel (both loads
// and stores) must stay valid until the kernel's event is consumed.
//
// Events are tracked in self.events: Map<BTreeSet<BufferId>, Event>.
// The key set must include every buffer the kernel touches, so future
// operations can find and wait on the event before reusing the buffer.
//
// When extracting a pending event for a buffer: iterate events.keys(),
// find the set containing the buffer, remove the event, and add it to
// the wait list passed to the next launch. A removed event is consumed.
// -----------------------------

use std::{
    collections::BTreeSet,
    env,
    hash::BuildHasherDefault,
    path::{Path, PathBuf},
};

use nanoserde::DeJson;

use crate::{
    DType, DebugMask, Map, Scalar, Set, ZyxError,
    backend::{AutotuneConfig, BufferId, Config, Device, DeviceInfo, DeviceProgramId, Event, MemoryPool, OpCapability, PoolId},
    dtype::Constant,
    kernel::{BOp, DeviceId, Kernel, MoveOp, Op, OpId, UOp, autotune::OptSeq},
    rng::Rng,
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    tensor::TensorId,
    view::View,
};

#[derive(Debug, Copy, Clone, Hash, PartialEq, PartialOrd, Eq, Ord)]
struct ShapeId(u16);

impl From<usize> for ShapeId {
    fn from(value: usize) -> Self {
        ShapeId(value as u16)
    }
}

impl From<ShapeId> for usize {
    fn from(value: ShapeId) -> Self {
        value.0 as usize
    }
}

impl SlabId for ShapeId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u16::MAX);
    fn inc(&mut self) {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub(crate) struct DeviceInfoId(u32);

impl From<usize> for DeviceInfoId {
    fn from(value: usize) -> Self {
        DeviceInfoId(value as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub(crate) struct KernelId(u32);

impl From<usize> for KernelId {
    fn from(value: usize) -> Self {
        KernelId(value as u32)
    }
}

impl From<KernelId> for usize {
    fn from(value: KernelId) -> Self {
        value.0 as usize
    }
}

impl SlabId for KernelId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug)]
struct TensorData {
    shape_id: ShapeId,
    dtype: DType,
    kernel_id: KernelId,
    op_id: OpId,
    pending_store: bool,
}

#[derive(Debug)]
struct KernelData {
    /// Tensor reference count. Each entry is a tensor this kernel must produce.
    /// When a tensor is consumed as input to a new op within the same kernel,
    /// it is removed from outputs (since the kernel produces the new op's result instead).
    outputs: Vec<TensorId>,
    loads: Vec<TensorId>,
    stores: Vec<TensorId>,
    kernel: Kernel,
}

pub struct Runtime {
    shape_map: Map<Vec<Dim>, ShapeId>,
    shapes: Slab<ShapeId, Vec<Dim>>,
    tensors: Slab<TensorId, TensorData>,
    kernels: Slab<KernelId, KernelData>,
    kernel_map: Map<Kernel, KernelId>,
    optimizations: Map<(KernelId, DeviceInfoId), OptSeq>,
    device_infos: Map<DeviceInfo, DeviceInfoId>,
    programs: Map<KernelId, DeviceProgramId>,
    pub devices: Slab<DeviceId, Device>,
    // Pool 0 is always host, pool 1 is disk if disk is present
    pools: Slab<PoolId, MemoryPool>,
    config_dir: Option<PathBuf>,
    buffer_map: Map<TensorId, BufferId>,
    events: Map<BTreeSet<BufferId>, Event>,
    pub rng: Rng,
    autotune_config: AutotuneConfig,
    pub implicit_casts: bool,
    pub training: bool,
    pub debug: DebugMask,
}

impl Runtime {
    pub const fn new() -> Self {
        Runtime {
            shape_map: Map::with_hasher(BuildHasherDefault::new()),
            shapes: Slab::new(),
            tensors: Slab::new(),
            kernels: Slab::new(),
            kernel_map: Map::with_hasher(BuildHasherDefault::new()),
            device_infos: Map::with_hasher(BuildHasherDefault::new()),
            devices: Slab::new(),
            pools: Slab::new(),
            programs: Map::with_hasher(BuildHasherDefault::new()),
            config_dir: None,
            optimizations: Map::with_hasher(BuildHasherDefault::new()),
            buffer_map: Map::with_hasher(BuildHasherDefault::new()),
            events: Map::with_hasher(BuildHasherDefault::new()),
            rng: Rng::seed_from_u64(42069),
            autotune_config: AutotuneConfig::new(),
            implicit_casts: true,
            training: false,
            debug: DebugMask::new(0),
        }
    }

    pub fn shape(&self, x: TensorId) -> &[Dim] {
        &self.shapes[self.tensors[x].shape_id]
    }

    pub fn dtype(&self, x: TensorId) -> DType {
        self.tensors[x].dtype
    }

    pub fn is_realized(&self, x: TensorId) -> bool {
        self.buffer_map.contains_key(&x)
    }

    /// Returns operation capabilities for a dtype across all devices.
    pub fn supports_dtype(&mut self, dtype: DType) -> OpCapability {
        if self.initialize_devices().is_err() {
            return OpCapability::none();
        }
        let mut caps = OpCapability::none();
        for (_id, dev) in self.devices.iter() {
            caps.0 |= dev.info().supports_dtype(dtype).0;
        }
        caps
    }

    pub fn retain(&mut self, x: TensorId) {
        eprintln!("Retain tensor x={x}");
        let kid = self.tensors[x].kernel_id;
        self.kernels[kid].outputs.push(x);
    }

    pub fn release(&mut self, x: TensorId) {
        let kid = self.tensors[x].kernel_id;
        let kd = &mut self.kernels[kid];
        kd.outputs.iter().position(|e| *e == x).map(|i| kd.outputs.remove(i));
        if !kd.outputs.contains(&x) && !self.buffer_map.contains_key(&x) && !self.tensors[x].pending_store {
            self.tensors.remove(x);
        }
        if kd.outputs.is_empty() {
            if !kd.kernel.contains_stores() {
                eprintln!("A: kernels.remove({kid:?})");
                self.kernels.remove(kid);
            } else {
                self.materialize_kernel(kid).unwrap();
            }
        }
    }

    fn push_shape(&mut self, shape: Vec<Dim>) -> ShapeId {
        if let Some(&shape_id) = self.shape_map.get(&shape) {
            shape_id
        } else {
            let shape_id = self.shapes.push(shape.clone());
            self.shape_map.insert(shape, shape_id);
            shape_id
        }
    }

    fn new_kernel(&mut self, op: Op, shape: Vec<Dim>, dtype: DType) -> TensorId {
        let shape_id = self.push_shape(shape);
        let mut kernel = Kernel::new(DeviceId::AUTO);
        let op_id = kernel.push_back(op);
        let kernel_id = self.kernels.push(KernelData { outputs: Vec::new(), loads: Vec::new(), stores: Vec::new(), kernel });
        let tid = self.tensors.push(TensorData { shape_id, dtype, kernel_id, op_id, pending_store: false });
        self.kernels[kernel_id].outputs.push(tid);
        tid
    }

    pub fn new_constant_tensor(&mut self, value: Constant) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::new_constant_tensor(value={value:?})");
        let op = Op::ConstView(Box::new((value, View::contiguous(&[1]))));
        let result = self.new_kernel(op, [1].into(), value.dtype());
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={result}, {:?}", self.tensors[result]);
        result
    }

    pub fn new_full(&mut self, shape: Vec<Dim>, value: Constant) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::new_full(shape={shape:?}, value={value:?})");
        let dtype = value.dtype();
        let op = Op::ConstView(Box::new((value, View::contiguous(&[1]))));
        let x = self.new_kernel(op, [1].into(), dtype);
        let expanded = self.expand(x, shape).unwrap();
        self.release(x);
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={expanded}, {:?}", self.tensors[expanded]);
        expanded
    }

    // Creates new tensor in host memory
    pub fn new_host_tensor<T: Scalar>(&mut self, shape: Vec<Dim>, data: Box<[T]>) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::new_host_tensor(shape={shape:?})");
        let dtype = T::dtype();

        self.initialize_devices()?;
        debug_assert_eq!(shape.iter().product::<Dim>(), data.len() as Dim);
        let bytes = (data.len() * dtype.bit_size() as usize / 8) as Dim;
        debug_assert_eq!(data.len() * std::mem::size_of::<T>(), bytes as usize);

        // Convert to Box<[u8]>
        let ptr = (Box::into_raw(data) as *mut T) as *mut u8;
        let slice = std::ptr::slice_from_raw_parts_mut(ptr, bytes as usize);
        let data = unsafe { Box::from_raw(slice) };

        // Store to Host memory
        let MemoryPool::Host(ref mut pool) = self.pools[PoolId::HOST] else {
            unreachable!("Host must exist.")
        };
        let buffer_id = BufferId { pool: PoolId::HOST, buffer: pool.insert(data) };

        // Create kerenl for it
        let op = Op::LoadView(Box::new((T::dtype(), View::contiguous(&shape))));
        let tid = self.new_kernel(op, shape, dtype);
        self.kernels[self.tensors[tid].kernel_id].loads.push(tid);

        self.buffer_map.insert(tid, buffer_id);

        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, {:?}", self.tensors[tid]);
        Ok(tid)
    }

    // Creates new tensor in disk
    pub fn new_disk_tensor(
        &mut self,
        shape: Vec<Dim>,
        dtype: DType,
        path: &Path,
        offset_bytes: u64,
    ) -> Result<TensorId, ZyxError> {
        self.initialize_devices()?;
        todo!()
    }

    pub fn cast(&mut self, x: TensorId, dtype: DType) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::cast(x={x}, dtype={dtype:?})");
        let shape_id = self.tensors[x].shape_id;
        let kernel_id = self.tensors[x].kernel_id;
        let op_id = self.tensors[x].op_id;
        let op_id = self.kernels[kernel_id].kernel.cast(op_id, dtype);
        let tid = self.tensors.push(TensorData { shape_id, dtype, kernel_id, op_id, pending_store: false });
        self.kernels[kernel_id].outputs.push(tid);
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
        tid
    }

    pub fn bitcast(&mut self, x: TensorId, dtype: DType) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::bitcast(x={x}, dtype={dtype:?})");
        let shape_id = self.tensors[x].shape_id;
        let kernel_id = self.tensors[x].kernel_id;
        let op_id = self.tensors[x].op_id;
        let op_id = self.kernels[kernel_id].kernel.bitcast(op_id, dtype);
        let tid = self.tensors.push(TensorData { shape_id, dtype, kernel_id, op_id, pending_store: false });
        self.kernels[kernel_id].outputs.push(tid);
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
        tid
    }

    pub fn unary(&mut self, x: TensorId, uop: UOp) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::unary(x={x}, uop={uop:?})");
        let shape_id = self.tensors[x].shape_id;
        let dtype = self.tensors[x].dtype;
        let kernel_id = self.tensors[x].kernel_id;
        let op_id = self.kernels[kernel_id].kernel.unary(self.tensors[x].op_id, uop);
        let tid = self.tensors.push(TensorData { shape_id, dtype, kernel_id, op_id, pending_store: false });
        self.kernels[kernel_id].outputs.push(tid);
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
        tid
    }

    pub fn binary(&mut self, x: TensorId, y: TensorId, bop: BOp) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::binary(x={x}, y={y}, bop={bop:?})");
        let shape_id = self.tensors[x].shape_id;
        let dtype = if bop.returns_bool() {
            DType::Bool
        } else {
            self.tensors[x].dtype
        };
        let kid_x = self.tensors[x].kernel_id;
        let op_id_x = self.tensors[x].op_id;
        let kid_y = self.tensors[y].kernel_id;
        let op_id_y = self.tensors[y].op_id;
        //println!("Binary input kernels: {kid_x:?} and {kid_y:?}");

        let (kernel_id, op_id) = if kid_x == kid_y {
            let op_id = self.kernels[kid_x].kernel.binary(op_id_x, op_id_y, bop);
            (kid_x, op_id)
        } else {
            let x_stores = !self.kernels[kid_x].stores.is_empty();
            let y_stores = !self.kernels[kid_y].stores.is_empty();
            if x_stores || y_stores {
                todo!("binary with stores not yet handled (kernelize.rs materializes input via add_store before merge)");
            }

            let swap = self.kernels[kid_y].kernel.is_reduce() && !self.kernels[kid_x].kernel.is_reduce();
            let (keep_kid, merge_kid, keep_op, merge_op) = if swap {
                (kid_y, kid_x, op_id_y, op_id_x)
            } else {
                (kid_x, kid_y, op_id_x, op_id_y)
            };

            //println!("Remove kernel {merge_kid:?}");
            let KernelData { outputs: merge_outputs, loads: merge_loads, stores: merge_stores, kernel } = unsafe {
                eprintln!("C: kernels.remove_and_return({merge_kid:?})");
                self.kernels.remove_and_return(merge_kid)
            };
            let Kernel { ops: merge_ops, head: merge_head, .. } = kernel;

            let mut op_map: Map<OpId, OpId> = Map::with_hasher(BuildHasherDefault::new());
            let mut i = merge_head;
            while !i.is_null() {
                let mut op = merge_ops[i].op.clone();
                for param in op.parameters_mut() {
                    if let Some(&new_param) = op_map.get(param) {
                        *param = new_param;
                    }
                }
                let new_op_id = self.kernels[keep_kid].kernel.push_back(op);
                op_map.insert(i, new_op_id);
                i = merge_ops[i].next;
            }

            for (_tid, t_data) in self.tensors.iter_mut() {
                let kid = &mut t_data.kernel_id;
                let op_id = &mut t_data.op_id;
                if *kid == merge_kid {
                    *kid = keep_kid;
                    if let Some(&new_op_id) = op_map.get(op_id) {
                        *op_id = new_op_id;
                    }
                }
            }

            eprintln!("D: kernel_data.remove({merge_kid:?})");
            let keep_data = &mut self.kernels[keep_kid];
            keep_data.outputs.extend(merge_outputs);
            keep_data.loads.extend(merge_loads);
            keep_data.stores.extend(merge_stores);

            let op_id = if swap {
                self.kernels[keep_kid].kernel.binary(op_map[&merge_op], keep_op, bop)
            } else {
                self.kernels[keep_kid].kernel.binary(keep_op, op_map[&merge_op], bop)
            };
            (keep_kid, op_id)
        };

        let tid = self.tensors.push(TensorData { shape_id, dtype, kernel_id, op_id, pending_store: false });
        self.kernels[kernel_id].outputs.push(tid);

        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
        Ok(tid)
    }

    pub fn to_device(&mut self, x: TensorId, device_id: DeviceId) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::to_device(x={x}, device_id={device_id:?})");
        todo!()
    }

    pub fn reduce(&mut self, x: TensorId, axes: Vec<UAxis>, rop: BOp) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::reduce(x={x}, axes={axes:?}, rop={rop:?})");

        let (kid, mut op_id) = self.duplicate_or_store(x)?;

        // Permute axes so reduce axes are last
        let shape = self.shape(x).to_vec();
        let n = shape.len();
        let max_axis = *axes.last().unwrap() as usize;
        let mut ai = 0;
        let mut permute_axes = Vec::with_capacity(n);
        for i in 0..=max_axis {
            if axes[ai] as usize == i {
                ai += 1;
            } else {
                permute_axes.push(i as UAxis);
            }
        }
        permute_axes.extend((max_axis + 1..n).map(|i| i as UAxis));
        permute_axes.extend_from_slice(&axes);

        if !permute_axes.iter().copied().eq(0..permute_axes.len() as UAxis) {
            op_id = self.kernels[kid].kernel.permute(op_id, &permute_axes);
        }

        op_id = self.kernels[kid].kernel.push_back(Op::Reduce { x: op_id, rop, n_axes: axes.len() });

        if shape.len() == axes.len() {
            op_id = self.kernels[kid].kernel.reshape(op_id, &[1]);
        }

        let reduce_shape = crate::shape::reduce(&shape, &axes);
        let shape_id = self.push_shape(reduce_shape);
        let dtype = self.tensors[x].dtype;
        let tid = self.tensors.push(TensorData { shape_id, dtype, kernel_id: kid, op_id, pending_store: false });

        debug_assert_eq!(self.kernels[kid].outputs.len(), 0, "input into reduce must have empty outputs");
        self.kernels[kid].outputs.push(tid);

        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kid:?}, op_id={op_id:?}");
        Ok(tid)
    }

    pub(super) fn reshape(&mut self, x: TensorId, shape: Vec<Dim>) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::reshape(x={x}, shape={shape:?})");
        let sh = self.shape(x);
        debug_assert_eq!(shape.iter().product::<Dim>(), sh.iter().product::<Dim>());
        if shape == sh {
            self.retain(x);
            return x;
        }
        // If x is realized, create a load kernel with the target shape.
        // The result shares x's buffer (set in buffer_map), so add_store
        // won't add a StoreView for it. This avoids copying data for a
        // view-only reshape.
        if let Some(&buf_id) = self.buffer_map.get(&x) {
            let dtype = self.tensors[x].dtype;
            let mut kernel = Kernel::new(DeviceId::AUTO);
            let op_id = kernel.load_contiguous(dtype, &shape);
            let shape_id = self.push_shape(shape);
            let kernel_id = self.kernels.push(KernelData { outputs: Vec::new(), loads: vec![x], stores: Vec::new(), kernel });
            let tid = self.tensors.push(TensorData { shape_id, dtype, kernel_id, op_id, pending_store: false });
            self.kernels[kernel_id].outputs.push(tid);
            self.buffer_map.insert(tid, buf_id);
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid} (load kernel, shares buffer with x={x})");
            return tid;
        }

        let (kernel_id, op_id) = self.duplicate_or_store(x).unwrap();
        let shape_id = self.push_shape(shape);
        let dtype = self.tensors[x].dtype;

        let op_id = self.kernels[kernel_id].kernel.reshape(op_id, &shape);
        let tid = self.tensors.push(TensorData { shape_id, dtype, kernel_id, op_id, pending_store: false });

        debug_assert_eq!(self.kernels[kernel_id].outputs.len(), 0, "input into reshape must have empty outputs");
        self.kernels[kernel_id].outputs.push(tid);

        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
        tid
    }

    pub fn expand(&mut self, mut x: TensorId, shape: Vec<Dim>) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::expand(x={x}, shape={shape:?})");
        let sh = self.shape(x);
        // View::expand cannot increase rank — unsqueeze leading dims via reshape first.
        if shape.len() > sh.len() {
            let new_shape: Vec<Dim> = std::iter::repeat_n(1, shape.len() - sh.len()).chain(sh.iter().copied()).collect();
            x = self.reshape(x, new_shape);
        }
        if shape == self.shape(x) {
            self.retain(x);
            return Ok(x);
        }

        let (kernel_id, op_id) = self.duplicate_or_store(x)?;
        let shape_id = self.push_shape(shape.clone());
        let dtype = self.tensors[x].dtype;

        let op_id = self.kernels[kernel_id].kernel.expand(op_id, &shape);
        let tid = self.tensors.push(TensorData { shape_id, dtype, kernel_id, op_id, pending_store: false });

        debug_assert_eq!(self.kernels[kernel_id].outputs.len(), 0, "input into expand must have empty outputs");
        self.kernels[kernel_id].outputs.push(tid);

        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
        Ok(tid)
    }

    pub fn permute(&mut self, x: TensorId, axes: Vec<UAxis>) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::permute(x={x}, axes={axes:?})");
        let sh = self.shape(x);
        if axes.iter().copied().eq(0..sh.len() as UAxis) {
            self.retain(x);
            return x;
        }

        let (kernel_id, op_id) = self.duplicate_or_store(x).unwrap();
        let new_shape = crate::shape::permute(self.shape(x), &axes);
        let shape_id = self.push_shape(new_shape.clone());
        let dtype = self.tensors[x].dtype;

        let tid = self.tensors.push(TensorData { shape_id, dtype, kernel_id, op_id, pending_store: false });
        let op_id = self.kernels[kernel_id].kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Permute { axes: axes.into(), shape: new_shape }) });

        debug_assert_eq!(self.kernels[kernel_id].outputs.len(), 0, "input into permute must have empty outputs");
        self.kernels[kernel_id].outputs.push(tid);

        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
        tid
    }

    pub fn pad_zeros(&mut self, x: TensorId, padding: Vec<(i64, i64)>) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::pad_zeros(x={x}, padding={padding:?})");

        let (kernel_id, op_id) = self.duplicate_or_store(x).unwrap();
        let mut new_shape = self.shape(x).to_vec();
        crate::shape::pad(&mut new_shape, &padding);
        let shape_id = self.push_shape(new_shape.clone());
        let dtype = self.tensors[x].dtype;

        let tid = self.tensors.push(TensorData { shape_id, dtype, kernel_id, op_id, pending_store: false });
        let op_id = self.kernels[kernel_id].kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Pad { padding: padding.into(), shape: new_shape }) });

        debug_assert_eq!(self.kernels[kernel_id].outputs.len(), 0, "input into pad must have empty outputs");
        self.kernels[kernel_id].outputs.push(tid);

        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
        tid
    }

    pub fn load<T: Scalar>(&mut self, x: TensorId, data: &mut [T]) -> Result<(), ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::load(x={x})");
        let dt = self.tensors[x].dtype;
        if dt != T::dtype() {
            return Err(ZyxError::DTypeError(format!("loading dtype {}, but the data has dtype {dt}", T::dtype()).into()));
        }

        // Fast path: already realized
        if let Some(&buffer_id) = self.buffer_map.get(&x) {
            let n: usize = self.shape(x).iter().product::<Dim>() as usize;
            let bytes = n * (T::bit_size() / 8) as usize;
            let byte_slice = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), bytes) };
            for buffers in self.events.keys() {
                if buffers.contains(&buffer_id) {
                    let buffers = buffers.clone();
                    let event = self.events.remove(&buffers).unwrap();
                    self.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, vec![event])?;
                    #[cfg(feature = "debug_tensor_op")]
                    println!("  -> x={x}, {:?}", self.tensors[x]);
                    return Ok(());
                }
            }
            self.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, Vec::new())?;
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> x={x}, {:?}", self.tensors[x]);
            return Ok(());
        }

        // Slow path: add store for each output, last one triggers materialize
        self.initialize_devices()?;

        let kid = self.tensors[x].kernel_id;

        // Deduplicate: add_store removes ALL occurrences at once and creates a load kernel,
        // so we must process each unique tid only once
        let seen: Set<TensorId> = self.kernels[kid].outputs.iter().copied().collect();
        for tid in seen {
            self.add_store(tid)?;
        }

        // Copy result to host
        let n: usize = self.shape(x).iter().product::<Dim>() as usize;
        let bytes = n * (T::bit_size() / 8) as usize;
        let byte_slice = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), bytes) };
        let buffer_id = self.buffer_map[&x];
        for buffers in self.events.keys() {
            if buffers.contains(&buffer_id) {
                let buffers = buffers.clone();
                let event = self.events.remove(&buffers).unwrap();
                self.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, vec![event])?;
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> x={x}, {:?}", self.tensors[x]);
                return Ok(());
            }
        }
        self.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, Vec::new())?;
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> x={x}, {:?}", self.tensors[x]);
        Ok(())
    }

    // Initializes all available devices, creating a device for each compute
    // device and a memory pool for each physical memory.
    // Does nothing if devices were already initialized.
    // Returns error if all devices failed to initialize
    // DeviceParameters allows to disable some devices if requested
    pub fn initialize_devices(&mut self) -> Result<(), ZyxError> {
        if !self.devices.is_empty() {
            return Ok(());
        }

        // Set env vars
        if let Ok(x) = env::var("ZYX_DEBUG")
            && let Ok(x) = x.parse::<u32>()
        {
            self.debug = DebugMask(x);
        }

        // Search through config directory and find zyx/backend_config.json
        // If not found or failed to parse, use defaults.

        let config_file = env::var_os("XDG_CONFIG_HOME")
            .and_then(|path| {
                let path = PathBuf::from(path);
                if path.is_absolute() { Some(path) } else { None }
            })
            .or_else(|| env::home_dir().map(|home| home.join(".config")))
            .map(|path| path.join("zyx/config.json"))
            .and_then(|mut path| {
                if let Ok(file) = std::fs::read_to_string(&path) {
                    path.pop();
                    self.config_dir = Some(path);
                    Some(file)
                } else {
                    None
                }
            });

        let config = config_file
            .and_then(|file| {
                DeJson::deserialize_json(&file)
                    .map_err(|e| {
                        if self.debug.dev() {
                            println!("Failed to parse config.json, {e}");
                        }
                    })
                    .ok()
            })
            .inspect(|_| {
                if self.debug.dev() {
                    println!("Device config successfully read and parsed.");
                }
            })
            .unwrap_or_else(|| {
                if self.debug.dev() {
                    println!("Failed to get device config, using defaults.");
                }
                Config::default()
            });

        // Load optimizer cache from disk if it exists
        /*if let Some(mut path) = self.config_dir.clone() {
            path.push("cached_kernels");
            if let Ok(mut file) = std::fs::File::open(path) {
                use std::io::Read;
                let mut buf = Vec::new();
                file.read_to_end(&mut buf).unwrap();
                if let Ok(cache) = nanoserde::DeBin::deserialize_bin(&buf) {
                    self.kernel_cache = cache;
                }
            }
        }*/

        crate::backend::initialize_backends(&config, &mut self.pools, &mut self.devices, self.debug.dev())?;

        self.autotune_config = config.autotune;
        //println!("INIT runtime");
        Ok(())
    }

    /// This function deinitializes the whole runtime, deallocates all allocated memory and deallocates all caches
    /// It does not reset the rng and it does not change debug, search, training and `config_dir` fields
    #[allow(unused)]
    pub fn deinitialize(&mut self) {
        #[cfg(feature = "time")]
        {
            let lock = crate::ET.lock();
            let mut timings: Vec<_> = lock.iter().map(|(name, &(total_us, count))| (name.clone(), total_us, count)).collect();
            timings.sort_by_key(|a| std::cmp::Reverse(a.1));
            println!("\n=== Timing Info (sorted by total time, descending) ===");
            for (name, total_us, count) in timings {
                let per_call = total_us.checked_div(count).unwrap_or(0);
                println!("{name}: {total_us}us total, {per_call}us/call ({count} calls)");
            }
        }
        //println!("DEINIT runtime");
        self.shape_map = Map::default();
        self.shapes = Slab::new();
        self.tensors = Slab::new();
        self.kernels = Slab::new();
    }

    pub const fn manual_seed(&mut self, seed: u64) {
        self.rng = Rng::seed_from_u64(seed);
    }
}

#[allow(clippy::similar_names)]
pub fn get_perf(flop: u64, bytes_read: u64, bytes_written: u64, nanos: u64) -> String {
    const fn value_unit(x: u64) -> (u64, &'static str) {
        match x {
            0..1000 => (x * 100, ""),
            1_000..1_000_000 => (x / 10, "k"),
            1_000_000..1_000_000_000 => (x / 10_000, "M"),
            1_000_000_000..1_000_000_000_000 => (x / 10_000_000, "G"),
            1_000_000_000_000..1_000_000_000_000_000 => (x / 10_000_000_000, "T"),
            1_000_000_000_000_000..1_000_000_000_000_000_000 => (x / 10_000_000_000_000, "P"),
            1_000_000_000_000_000_000.. => (x / 10_000_000_000_000_000, "E"),
        }
    }

    if nanos == u64::MAX {
        return "INF time taken".to_string();
    }

    let (t, t_u) = match nanos {
        0..1_000 => (nanos * 10, "ns"),
        1_000..1_000_000 => (nanos / 100, "μs"),
        1_000_000..1_000_000_000 => (nanos / 100_000, "ms"),
        1_000_000_000..1_000_000_000_000 => (nanos / 100_000_000, "s"),
        1_000_000_000_000.. => (nanos / 6_000_000_000, "min"),
    };

    let (fs, f_us) = value_unit(flop * 1_000_000 / nanos * 1000);
    let (brs, br_us) = value_unit(bytes_read * 1_000_000_000 / nanos);
    let (bws, bw_us) = value_unit(bytes_written * 1_000_000_000 / nanos);

    format!(
        "{}.{} {t_u} ~ {}.{:02} {f_us}FLOP/s, {}.{:02} {br_us}B/s r, {}.{:02} {bw_us}B/s w",
        t / 10,
        t % 10,
        fs / 100,
        fs % 100,
        brs / 100,
        brs % 100,
        bws / 100,
        bws % 100,
    )
}

impl Runtime {
    fn duplicate_or_store(&mut self, x: TensorId) -> Result<(KernelId, OpId), ZyxError> {
        let (mut kid, mut op_id) = self.visited[&x];

        let contains_stores = self.kernels[kid].contains_stores();
        let reduce_dims_big = self.kernels[kid].is_preceded_by_reduce(op_id);
        if contains_stores | reduce_dims_big {
            eprintln!("STORE PATH");
            self.add_store(x)?;
            (kid, op_id) = self.visited[&x];
        } else {
            eprintln!("DUPLICATE PATH");
            kid = {
                let this = &mut *self;
                let x = x;
                let orig_loads = this.kernels[kid].loads.clone();
                let mut kernel = this.kernels[kid].kernel.clone();
                let new_output = vec![this.tensors[x].op_id];
                let new_loads = kernel.drop_unused_ops_by_params(new_output, &orig_loads);

                let old_outputs: Vec<TensorId> = this.kernels[kid].outputs.clone();
                let old_params: Vec<OpId> = old_outputs.iter().map(|&tid| this.tensors[tid].op_id).collect();
                let old_loads = this.kernels[kid].kernel.drop_unused_ops_by_params(old_params, &orig_loads);
                this.kernels[kid].loads = old_loads;

                let op_id = this.tensors[&x].op_id;
                let count = old_outputs.iter().filter(|&&tid| tid == x).count();
                // No, this iw WRONG assumption, fix it
                //
                // Remove from old kernel (phantom reference would never be released
                // since visited[x] now points to the new kernel)
                //
                this.kernels[kid].outputs.position(|&e| e != x);
                let stores = kd.stores.clone();
                let new_kid = this.kernels.push(KernelData { outputs: vec![x; count], loads: new_loads, stores, kernel });
                this.tensors[x].kernel_id = new_kid;
                new_kid
            };
        }

        self.kernel_data.get_mut(&kid).unwrap().outputs = Vec::new();

        Ok((kid, op_id))
    }

    fn add_store(&mut self, x: TensorId) -> Result<(), ZyxError> {
        let kid = self.tensors[x].kernel_id;
        let op_id = self.tensors[x].op_id;

        let (outputs_empty, count) = {
            // Remove ALL occurrences of x (handles reference counting from retain/clone)
            let prev_len = self.kernels[kid].outputs.len();
            self.kernels[kid].outputs.retain(|&e| e != x);
            let count = prev_len - self.kernels[kid].outputs.len();
            debug_assert!(count > 0, "add_store called for tid not in outputs");

            // Only add StoreView if x isn't already realized
            if !self.buffer_map.contains_key(&x) && !self.tensors[x].pending_store {
                // Invariant: a kernel must never both load and store the same tensor
                debug_assert!(!self.kernels[kid].loads.contains(&x), "kernel {kid:?} both loads and stores tid {x}");

                let dtype = self.tensors[x].dtype;
                self.kernels[kid].kernel.store_contiguous(op_id, dtype);
                self.kernels[kid].stores.push(x);

                self.tensors[x].pending_store = true;
            }

            (self.kernels[kid].outputs.is_empty(), count)
        };

        // Create load kernel so the tensor remains usable (visited must point to a live kernel)
        let dtype = self.tensors[x].dtype;
        let mut kernel = Kernel::new(DeviceId::AUTO);
        let shape = self.shape(x);
        let load_op_id = kernel.load_contiguous(dtype, &shape);
        let load_kid = self.kernels.push(KernelData { outputs: vec![x; count], loads: vec![x], stores: Vec::new(), kernel });
        self.tensors[x].kernel_id = load_kid;
        self.tensors[x].op_id = load_op_id;

        if outputs_empty {
            self.materialize_kernel(kid)?;
        }
        Ok(())
    }

    pub fn get_or_autotune(
        &mut self,
        mut kernel: Kernel,
        device_id: DeviceId,
        pool_id: PoolId,
        flop: u64,
        read: u64,
        write: u64,
    ) -> Result<(DeviceProgramId, u64), ZyxError> {
        let dev_info = self.devices[device_id].info().clone();
        let dev_info_id = self.get_or_add_dev_info(&dev_info);

        kernel.sort_global_defines();

        let kernel_id = if let Some(&cached_kid) = self.kernel_map.get(&kernel) {
            if let Some(&program_id) = self.programs.get(&cached_kid) {
                return Ok((program_id, 0));
            }
            if let Some(opt_seq) = self.optimizations.get(&(cached_kid, dev_info_id)) {
                opt_seq.apply(&mut kernel, &dev_info);
                let program_id = {
                    let device = &mut self.devices[device_id];
                    device.compile(&kernel, self.debug.asm())?
                };
                self.programs.insert(cached_kid, program_id);
                return Ok((program_id, 0));
            }
            cached_kid
        } else {
            let kernel_id =
                KernelId::from(self.kernel_map.values().copied().max().map_or(0, |id| usize::from(id).checked_add(1).unwrap()));
            let newly_inserted = self.kernel_map.insert(kernel.clone(), kernel_id).is_none();
            assert!(newly_inserted);
            kernel_id
        };

        kernel.unfold_movement_ops();
        {
            let device = &mut self.devices[device_id];
            let global_indices = kernel.get_global_indices();
            let max_global_dims = device.info().max_global_work_dims.len();
            if global_indices.len() > max_global_dims {
                let n = global_indices.len() + 1 - max_global_dims;
                let loops: Vec<OpId> = global_indices.values().copied().take(n).collect();
                kernel.merge_indices(&loops);
            }
            kernel.renumber_indices();
            kernel.verify();
        }
        let (program_id, opts, timing) = kernel.autotune_(
            &mut self.devices[device_id],
            &mut self.pools[pool_id],
            &self.autotune_config,
            flop,
            read,
            write,
            self.debug,
        )?;
        self.programs.insert(kernel_id, program_id);
        self.optimizations.insert((kernel_id, dev_info_id), opts);

        Ok((program_id, timing))
    }

    /// Materializes a kernel by adding store ops for all its outputs, compiling,
    /// launching, then creating load kernels for each output so the tensors remain
    /// usable in further graph construction. The kernel is consumed (removed from
    /// the slab) and cached in `kernel_map`/`programs` for reuse.
    ///
    /// # Invariant
    /// A kernel must never both load and store the same tensor (prevents aliasing).
    /// The debug_assert in the recursive materialization loop enforces this.
    fn materialize_kernel(&mut self, kid: KernelId) -> Result<(), ZyxError> {
        eprintln!("E: Materialize kernel and kernel_data.remove({kid:?})");
        let KernelData { outputs, loads, stores, kernel } = unsafe { self.kernels.remove_and_return(kid) };

        debug_assert!(outputs.is_empty(), "all outputs must be stored before materialize");

        debug_assert!(
            loads.iter().all(|&tid| {
                self.buffer_map.contains_key(&tid)
                    || outputs.contains(&tid)
                    || self.kernels.values().any(|kd| kd.outputs.contains(&tid) || kd.stores.contains(&tid))
            }),
            "load tid must be realized or in kernel's own outputs, or in some other kernel's outputs/stores"
        );

        // Debug: ensure each store tid is in exactly one kernel's outputs
        #[cfg(debug_assertions)]
        {
            for &tid in &stores {
                let count = self.kernels.values().filter(|kd| kd.outputs.contains(&tid)).count();
                debug_assert_eq!(count, 1, "store tid={tid} is in {count} kernels' outputs");
            }
        }

        // Recursive materialization
        for &load in &loads {
            if self.tensors[load].pending_store {
                let kid = self.tensors[load].kernel_id;
                let outputs: Set<TensorId> = self.kernels[kid].outputs.iter().copied().collect();
                for output in outputs {
                    self.add_store(output)?;
                }
            }
        }

        debug_assert!(
            loads.iter().all(|&tid| self.buffer_map.contains_key(&tid)),
            "all loads must be realized after recursive materialization"
        );

        // Pick device and pool
        self.initialize_devices()?;
        let mut dev_ids: Vec<DeviceId> = self.devices.ids().collect();
        dev_ids.sort_unstable_by_key(|&dev_id| self.devices[dev_id].free_compute());
        dev_ids.reverse();
        let dev_id = *dev_ids.first().ok_or_else(|| ZyxError::AllocationError("no available device".into()))?;
        let pool_id = self.devices[dev_id].memory_pool_id();

        // Ensure loads are in target pool
        let mut event_wait_list = Vec::new();
        for &tid in &loads {
            let buf_id = self.buffer_map[&tid];
            if buf_id.pool != pool_id {
                let src = buf_id.buffer;
                let bytes = self.shape(tid).iter().product::<Dim>() as usize * (self.dtype(tid).bit_size() / 8) as usize;
                let mut byte_slice = vec![0u8; bytes];

                let mut ev = Vec::new();
                for buffers in self.events.keys() {
                    if buffers.contains(&buf_id) {
                        let buffers = buffers.clone();
                        let event = self.events.remove(&buffers).unwrap();
                        ev.push(event);
                        break;
                    }
                }
                self.pools[buf_id.pool].pool_to_host(src, &mut byte_slice, ev)?;
                self.buffer_map.remove(&tid);
                // Deallocate old buffer if no other mapping uses it
                if !self.buffer_map.values().any(|b| b.buffer == src) {
                    self.pools[buf_id.pool].deallocate(src, vec![]);
                }

                let (dst, event) = self.pools[pool_id].allocate(bytes as Dim)?;
                let dst_global = BufferId { pool: pool_id, buffer: dst };
                let event = self.pools[pool_id].host_to_pool(&byte_slice, dst, vec![event])?;
                self.pools[pool_id].sync_events(vec![event])?;
                self.buffer_map.insert(tid, dst_global);
            } else {
                for buffers in self.events.keys() {
                    if buffers.contains(&buf_id) {
                        let buffers = buffers.clone();
                        let event = self.events.remove(&buffers).unwrap();
                        event_wait_list.push(event);
                        break;
                    }
                }
            }
        }

        // Allocate store buffers (one per unique tid)
        let mut kernel_buffers = BTreeSet::new();
        // All kernel buffers (loads + stores) must be tracked in kernel_buffers
        // so future operations can find and wait on the kernel's event before
        // reusing any of these buffers.
        for &tid in &loads {
            kernel_buffers.insert(self.buffer_map[&tid]);
        }
        for &tid in &stores {
            let bytes = self.shape(tid).iter().product::<Dim>() as usize * (self.dtype(tid).bit_size() / 8) as usize;
            let alloc_bytes = bytes as Dim + Dim::from(self.dtype(tid).bit_size() / 8);
            let (buf, event) = self.pools[pool_id].allocate(alloc_bytes)?;
            let global_id = BufferId { pool: pool_id, buffer: buf };
            self.buffer_map.insert(tid, global_id);
            self.tensors[tid].pending_store = false;
            kernel_buffers.insert(global_id);
            event_wait_list.push(event);
        }

        // Build args: load buffers first, then store buffers
        let mut args = Vec::new();
        for &tid in &loads {
            args.push(self.buffer_map[&tid].buffer);
        }
        for &tid in &stores {
            args.push(self.buffer_map[&tid].buffer);
        }

        // Compile and launch (caches in kernel_map / programs)
        let debug = self.debug;
        if debug.sched() {
            eprintln!("tensors: {:?}", self.tensors.ids().collect::<Vec<TensorId>>());
            eprintln!("loads (tids): {loads:?}");
            eprintln!("stores (tids): {stores:?}");
            for (info_kid, info_kd) in self.kernels.iter() {
                eprintln!(
                    "  kernel {info_kid:?}: outputs={:?}, loads={:?}, stores={:?}",
                    info_kd.outputs, info_kd.loads, info_kd.stores
                );
            }
            kernel.debug();
        }
        let (flop, read, write) = kernel.flop_mem_rw();
        let (dev_prog, _timing) = self.get_or_autotune(kernel, dev_id, pool_id, flop, read, write)?;

        let event = self.devices[dev_id].launch(dev_prog, &mut self.pools[pool_id], &args, event_wait_list)?;
        self.events.insert(kernel_buffers, event);

        Ok(())
    }

    fn get_or_add_dev_info(&mut self, device_info: &DeviceInfo) -> DeviceInfoId {
        if let Some(&dev_info_id) = self.device_infos.get(device_info) {
            dev_info_id
        } else {
            let dev_info_id =
                DeviceInfoId(self.device_infos.values().copied().max().map_or(0, |id| id.0.checked_add(1).unwrap()));
            let newly_inserted = self.device_infos.insert(device_info.clone(), dev_info_id).is_none();
            assert!(newly_inserted);
            dev_info_id
        }
    }
}
