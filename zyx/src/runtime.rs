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
    kernel::{BOp, DeviceId, Kernel, Op, OpId, UOp, autotune::OptSeq},
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

struct TensorData {
    shape_id: ShapeId,
    dtype: DType,
}

struct KernelData {
    outputs: Vec<TensorId>,
    loads: Vec<TensorId>,
    stores: Vec<TensorId>,
}

pub struct Runtime {
    tensors: Slab<TensorId, TensorData>,
    shape_map: Map<Vec<Dim>, ShapeId>,
    shapes: Slab<ShapeId, Vec<Dim>>,
    visited: Map<TensorId, (KernelId, OpId)>,
    kernel_data: Map<KernelId, KernelData>,
    kernels: Slab<KernelId, Kernel>,
    kernel_map: Map<Kernel, KernelId>,
    device_infos: Map<DeviceInfo, DeviceInfoId>,
    pub devices: Slab<DeviceId, Device>,
    // Pool 0 is always host, pool 1 is disk if disk is present
    pools: Slab<PoolId, MemoryPool>,
    programs: Map<KernelId, DeviceProgramId>,
    config_dir: Option<PathBuf>,
    optimizations: Map<(KernelId, DeviceInfoId), OptSeq>,
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
            tensors: Slab::new(),
            shape_map: Map::with_hasher(BuildHasherDefault::new()),
            shapes: Slab::new(),
            visited: Map::with_hasher(BuildHasherDefault::new()),
            kernel_data: Map::with_hasher(BuildHasherDefault::new()),
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
        let kid = self.visited.get_mut(&x).unwrap().0;
        self.kernel_data.get_mut(&kid).unwrap().outputs.push(x);
    }

    pub fn release(&mut self, x: TensorId) {
        let Some(&(kid, _)) = self.visited.get(&x) else {
            panic!("Kernel must exist");
        };
        let Some(kd) = self.kernel_data.get_mut(&kid) else {
            panic!("Kernel must exist");
        };
        kd.outputs.iter().position(|e| *e == x).map(|i| kd.outputs.remove(i));
        if !kd.outputs.contains(&x) && !self.buffer_map.contains_key(&x) {
            self.tensors.remove(x);
        }
        if kd.outputs.is_empty() {
            if !self.kernels[kid].contains_stores() {
                eprintln!("A: kernels.remove({kid:?})");
                self.kernels.remove(kid);
                eprintln!("B: kernel_data.remove({kid:?})");
                self.kernel_data.remove(&kid);
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
        let tid = self.tensors.push(TensorData { shape_id, dtype });
        let mut kernel = Kernel::new(DeviceId::AUTO);
        let op_id = kernel.push_back(op);
        let kid = self.kernels.push(kernel);
        self.kernel_data.insert(
            kid,
            KernelData {
                outputs: vec![tid],
                loads: Vec::new(),
                stores: Vec::new(),
            },
        );
        self.visited.insert(tid, (kid, op_id));
        tid
    }

    pub fn new_constant_tensor(&mut self, value: Constant) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::new_constant_tensor(value={value:?})");
        let op = Op::ConstView(Box::new((value, View::contiguous(&[1]))));
        let result = self.new_kernel(op, [1].into(), value.dtype());
        #[cfg(feature = "debug_tensor_op")]
        println!(
            "  -> tid={result}, kid={:?}, op_id={:?}",
            self.visited[&result].0, self.visited[&result].1
        );
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
        println!(
            "  -> tid={expanded}, kid={:?}, op_id={:?}",
            self.visited[&expanded].0, self.visited[&expanded].1
        );
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
        let buffer_id = BufferId {
            pool: PoolId::HOST,
            buffer: pool.insert(data),
        };

        // Create kerenl for it
        let op = Op::LoadView(Box::new((T::dtype(), View::contiguous(&shape))));
        let tid = self.new_kernel(op, shape, dtype);
        self.kernel_data.get_mut(&self.visited[&tid].0).unwrap().loads.push(tid);

        self.buffer_map.insert(tid, buffer_id);

        #[cfg(feature = "debug_tensor_op")]
        println!(
            "  -> tid={tid}, kid={:?}, op_id={:?}",
            self.visited[&tid].0, self.visited[&tid].1
        );
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
        let tid = self.tensors.push(TensorData { shape_id, dtype });
        let (kid, op_id) = self.visited[&x];
        let op_id = self.kernels[kid].cast(op_id, dtype);
        self.kernel_data.get_mut(&kid).unwrap().outputs.push(tid);
        self.visited.insert(tid, (kid, op_id));
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kid:?}, op_id={op_id:?}");
        tid
    }

    pub fn bitcast(&mut self, x: TensorId, dtype: DType) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::bitcast(x={x}, dtype={dtype:?})");
        let shape_id = self.tensors[x].shape_id;
        let tid = self.tensors.push(TensorData { shape_id, dtype });
        let (kid, op_id) = self.visited[&x];
        let op_id = self.kernels[kid].bitcast(op_id, dtype);
        self.kernel_data.get_mut(&kid).unwrap().outputs.push(tid);
        self.visited.insert(tid, (kid, op_id));
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kid:?}, op_id={op_id:?}");
        tid
    }

    pub fn unary(&mut self, x: TensorId, uop: UOp) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::unary(x={x}, uop={uop:?})");
        let shape_id = self.tensors[x].shape_id;
        let dtype = self.tensors[x].dtype;
        let tid = self.tensors.push(TensorData { shape_id, dtype });
        let (kid, op_id) = self.visited[&x];
        let op_id = self.kernels[kid].unary(op_id, uop);
        self.kernel_data.get_mut(&kid).unwrap().outputs.push(tid);
        self.visited.insert(tid, (kid, op_id));
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kid:?}, op_id={op_id:?}");
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
        let tid = self.tensors.push(TensorData { shape_id, dtype });
        let (kid_x, op_id_x) = self.visited[&x];
        let (kid_y, op_id_y) = self.visited[&y];
        //println!("Binary input kernels: {kid_x:?} and {kid_y:?}");

        let (kid, op_id) = if kid_x == kid_y {
            let op_id = self.kernels[kid_x].binary(op_id_x, op_id_y, bop);
            (kid_x, op_id)
        } else {
            let x_stores = !self.kernel_data.get(&kid_x).unwrap().stores.is_empty();
            let y_stores = !self.kernel_data.get(&kid_y).unwrap().stores.is_empty();
            if x_stores || y_stores {
                todo!("binary with stores not yet handled (kernelize.rs materializes input via add_store before merge)");
            }

            let swap = self.kernels[kid_y].is_reduce() && !self.kernels[kid_x].is_reduce();
            let (keep_kid, merge_kid, keep_op, merge_op) = if swap {
                (kid_y, kid_x, op_id_y, op_id_x)
            } else {
                (kid_x, kid_y, op_id_x, op_id_y)
            };

            //println!("Remove kernel {merge_kid:?}");
            let Kernel {
                ops: merge_ops,
                head: merge_head,
                custom_kernel_id,
                ..
            } = unsafe {
                eprintln!("C: kernels.remove_and_return({merge_kid:?})");
                self.kernels.remove_and_return(merge_kid)
            };
            debug_assert!(custom_kernel_id.is_none());

            let mut op_map: Map<OpId, OpId> = Map::with_hasher(BuildHasherDefault::new());
            let mut i = merge_head;
            while !i.is_null() {
                let mut op = merge_ops[i].op.clone();
                for param in op.parameters_mut() {
                    if let Some(&new_param) = op_map.get(param) {
                        *param = new_param;
                    }
                }
                let new_op_id = self.kernels[keep_kid].push_back(op);
                op_map.insert(i, new_op_id);
                i = merge_ops[i].next;
            }

            for (_tid, (kid, op_id)) in self.visited.iter_mut() {
                if *kid == merge_kid {
                    *kid = keep_kid;
                    if let Some(&new_op_id) = op_map.get(op_id) {
                        *op_id = new_op_id;
                    }
                }
            }

            eprintln!("D: kernel_data.remove({merge_kid:?})");
            let merge_data = self.kernel_data.remove(&merge_kid).unwrap();
            let keep_data = self.kernel_data.get_mut(&keep_kid).unwrap();
            keep_data.outputs.extend(merge_data.outputs);
            keep_data.loads.extend(merge_data.loads);
            keep_data.stores.extend(merge_data.stores);

            let op_id = if swap {
                self.kernels[keep_kid].binary(op_map[&merge_op], keep_op, bop)
            } else {
                self.kernels[keep_kid].binary(keep_op, op_map[&merge_op], bop)
            };
            (keep_kid, op_id)
        };

        self.kernel_data.get_mut(&kid).unwrap().outputs.push(tid);
        self.visited.insert(tid, (kid, op_id));

        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kid:?}, op_id={op_id:?}");
        Ok(tid)
    }

    pub fn reduce(&mut self, x: TensorId, axes: Vec<UAxis>, rop: BOp) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::reduce(x={x}, axes={axes:?}, rop={rop:?})");
        let (mut kid, mut op_id) = self.visited[&x];

        if self.kernels[kid].contains_stores() | self.kernels[kid].is_preceded_by_reduce(op_id) {
            self.add_store(x)?;
            (kid, op_id) = self.visited[&x];
        }

        if self.kernel_data[&kid].outputs.len() > 1 {
            let reduce_dims_big = self.kernels[kid].is_preceded_by_reduce(op_id);
            if reduce_dims_big {
                self.add_store(x)?;
                (kid, op_id) = self.visited[&x];
            } else {
                kid = self.duplicate_kernel(x, kid);
            }
        }

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
            op_id = self.kernels[kid].permute(op_id, &permute_axes);
        }

        op_id = self.kernels[kid].push_back(Op::Reduce {
            x: op_id,
            rop,
            n_axes: axes.len(),
        });

        if shape.len() == axes.len() {
            op_id = self.kernels[kid].reshape(op_id, &[1]);
        }

        let reduce_shape = crate::shape::reduce(&shape, &axes);
        let shape_id = self.push_shape(reduce_shape);
        let dtype = self.tensors[x].dtype;
        let tid = self.tensors.push(TensorData { shape_id, dtype });

        self.kernel_data.get_mut(&kid).unwrap().outputs.push(tid);
        self.visited.insert(tid, (kid, op_id));

        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kid:?}, op_id={op_id:?}");
        Ok(tid)
    }

    pub fn to_device(&mut self, x: TensorId, device_id: DeviceId) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::to_device(x={x}, device_id={device_id:?})");
        todo!()
    }

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

    /// Adds a StoreView for x to its kernel, removes x from visited and outputs.
    /// Unlike `load`, this does not launch the kernel — it only records the store in the IR.
    ///
    /// # Invariant
    /// A kernel must never both load and store the same tensor (prevents aliasing).
    /// `add_store` asserts that x is not already in the kernel's load list.
    fn add_store(&mut self, x: TensorId) -> Result<(), ZyxError> {
        let &(kid, op_id) = self.visited.get(&x).unwrap();
        let (outputs_empty, count) = {
            let kd = self.kernel_data.get_mut(&kid).unwrap();

            // Remove ALL occurrences of x (handles reference counting from retain/clone)
            let prev_len = kd.outputs.len();
            kd.outputs.retain(|&e| e != x);
            let count = prev_len - kd.outputs.len();
            debug_assert!(count > 0, "add_store called for tid not in outputs");

            // Only add StoreView if x isn't already realized
            if !self.buffer_map.contains_key(&x) && !kd.stores.contains(&x) {
                // Invariant: a kernel must never both load and store the same tensor
                debug_assert!(!kd.loads.contains(&x), "kernel {kid:?} both loads and stores tid {x}");

                let dtype = self.tensors[x].dtype;
                self.kernels[kid].store_contiguous(op_id, dtype);
                kd.stores.push(x);
            }

            (kd.outputs.is_empty(), count)
        };

        // Create load kernel so the tensor remains usable (visited must point to a live kernel)
        let shape = self.shape(x).to_vec();
        let dtype = self.tensors[x].dtype;
        let mut kernel = Kernel::new(DeviceId::AUTO);
        let load_op_id = kernel.load_contiguous(dtype, &shape);
        let load_kid = self.kernels.push(kernel);
        self.kernel_data.insert(
            load_kid,
            KernelData {
                outputs: vec![x; count],
                loads: vec![x],
                stores: Vec::new(),
            },
        );
        self.visited.insert(x, (load_kid, load_op_id));

        if outputs_empty {
            self.materialize_kernel(kid)?;
        }
        Ok(())
    }

    /// Creates a new kernel with a LoadView for x (used to reload after a store).
    /// Splits x's op chain into its own kernel. Both the original and new kernel
    /// keep only their needed ops via `drop_unused_ops_by_params`.
    ///
    /// # Avoiding phantom references
    /// `duplicate_kernel` moves `x` from the old kernel's outputs to the new
    /// kernel's outputs. This is critical: if `x` stays in the old kernel's
    /// outputs, it becomes a phantom reference — `visited[x]` now points to the
    /// new kernel, so `release(x)` will only hit the new kernel, never the old
    /// one. The old kernel keeps a stale entry that inflates `self.tensors`
    /// and may keep the old kernel alive indefinitely.
    ///
    /// # Debug printing
    /// When alive-tensor counts are out of sync with `self.tensors`, add prints at
    /// materialization time dumping `self.tensors.ids()` and all kernel_data
    /// entries. Any tensor id that appears in `self.tensors` but not in any
    /// kernel's `outputs` or `buffer_map` is a phantom — released by the user
    /// but never removed from `self.tensors` because the owning kernel had
    /// other outputs that kept it alive.
    fn duplicate_kernel(&mut self, x: TensorId, kid: KernelId) -> KernelId {
        let orig_loads = self.kernel_data[&kid].loads.clone();
        let mut kernel = self.kernels[kid].clone();
        let new_params = vec![self.visited[&x].1];
        let new_loads = kernel.drop_unused_ops_by_params(new_params, &orig_loads);

        let kd = self.kernel_data.get_mut(&kid).unwrap();
        let old_outputs: Vec<TensorId> = kd.outputs.clone();
        let old_params: Vec<OpId> = old_outputs.iter().map(|tid| self.visited[tid].1).collect();
        let old_loads = self.kernels[kid].drop_unused_ops_by_params(old_params, &orig_loads);
        kd.loads = old_loads;

        let old_op = self.visited[&x].1;
        let count = old_outputs.iter().filter(|&&tid| tid == x).count();
        // Remove from old kernel (phantom reference would never be released
        // since visited[x] now points to the new kernel)
        kd.outputs.retain(|&e| e != x);
        let stores = kd.stores.clone();
        let new_kid = self.kernels.push(kernel);
        self.kernel_data.insert(
            new_kid,
            KernelData {
                outputs: vec![x; count],
                loads: new_loads,
                stores,
            },
        );
        self.visited.insert(x, (new_kid, old_op));
        new_kid
    }

    /// Ensures x is in a kernel where it's safe to append a movement op.
    /// Handles four cases:
    /// 1. Kernel has stores → materialize x, create new load kernel
    /// 2. x not in outputs (already replaced by prior movement op) → materialize
    /// 3. Multi-output + preceded by reduce → materialize, load back
    /// 4. Multi-output only → duplicate kernel
    fn duplicate_or_store(&mut self, x: TensorId) -> Result<(KernelId, OpId), ZyxError> {
        let (mut kid, mut op_id) = self.visited[&x];

        if self.kernels[kid].contains_stores() {
            self.add_store(x)?;
            (kid, op_id) = self.visited[&x];
        }

        // If values inside reduction need to be used elsewhere, we have to duplicate
        if self.kernel_data[&kid].outputs.len() > 1 {
            let reduce_dims_big = self.kernels[kid].is_preceded_by_reduce(op_id);
            if reduce_dims_big {
                self.add_store(x)?;
                (kid, op_id) = self.visited[&x];
            } else {
                kid = self.duplicate_kernel(x, kid);
            }
        }

        Ok((kid, op_id))
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
            let load_op_id = kernel.load_contiguous(dtype, &shape);
            let shape_id = self.push_shape(shape);
            let tid = self.tensors.push(TensorData { shape_id, dtype });
            let load_kid = self.kernels.push(kernel);
            self.buffer_map.insert(tid, buf_id);
            self.kernel_data.insert(
                load_kid,
                KernelData {
                    outputs: vec![tid],
                    loads: vec![x],
                    stores: Vec::new(),
                },
            );
            self.visited.insert(tid, (load_kid, load_op_id));
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid} (load kernel, shares buffer with x={x})");
            return tid;
        }
        let (kid, op_id) = self.duplicate_or_store(x).unwrap();
        let shape_id = self.push_shape(shape.clone());
        let dtype = self.tensors[x].dtype;
        let tid = self.tensors.push(TensorData { shape_id, dtype });

        let op_id = self.kernels[kid].reshape(op_id, &shape);
        self.kernel_data.get_mut(&kid).unwrap().outputs.push(tid);
        self.visited.insert(tid, (kid, op_id));
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kid:?}, op_id={op_id:?}");
        tid
    }

    pub fn expand(&mut self, x: TensorId, shape: Vec<Dim>) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::expand(x={x}, shape={shape:?})");
        let sh = self.shape(x);
        if shape == sh {
            self.retain(x);
            return Ok(x);
        }
        let (mut kid, mut op_id) = self.visited[&x];

        // Expand also checks is_preceded_by_compute (unlike permute/reshape/pad)
        if self.kernels[kid].contains_stores() | self.kernels[kid].is_preceded_by_compute(op_id) {
            self.add_store(x)?;
            (kid, op_id) = self.visited[&x];
        }

        if self.kernel_data[&kid].outputs.len() > 1 {
            let reduce_dims_big = self.kernels[kid].is_preceded_by_reduce(op_id);
            if reduce_dims_big {
                self.add_store(x)?;
                (kid, op_id) = self.visited[&x];
            } else {
                kid = self.duplicate_kernel(x, kid);
            }
        }

        let shape_id = self.push_shape(shape.clone());
        let dtype = self.tensors[x].dtype;
        let tid = self.tensors.push(TensorData { shape_id, dtype });

        let op_id = self.kernels[kid].expand(op_id, &shape);
        self.kernel_data.get_mut(&kid).unwrap().outputs.push(tid);
        self.visited.insert(tid, (kid, op_id));
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kid:?}, op_id={op_id:?}");
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
        let (kid, op_id) = self.duplicate_or_store(x).unwrap();
        let new_shape = crate::shape::permute(self.shape(x), &axes);
        let shape_id = self.push_shape(new_shape);
        let dtype = self.tensors[x].dtype;
        let tid = self.tensors.push(TensorData { shape_id, dtype });

        let op_id = self.kernels[kid].permute(op_id, &axes);
        self.kernel_data.get_mut(&kid).unwrap().outputs.push(tid);
        self.visited.insert(tid, (kid, op_id));
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kid:?}, op_id={op_id:?}");
        tid
    }

    pub fn pad_zeros(&mut self, x: TensorId, padding: Vec<(i64, i64)>) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::pad_zeros(x={x}, padding={padding:?})");
        let (kid, op_id) = self.duplicate_or_store(x).unwrap();
        let mut new_shape = self.shape(x).to_vec();
        crate::shape::pad(&mut new_shape, &padding);
        let shape_id = self.push_shape(new_shape);
        let dtype = self.tensors[x].dtype;
        let tid = self.tensors.push(TensorData { shape_id, dtype });

        let op_id = self.kernels[kid].pad(op_id, &padding);
        self.kernel_data.get_mut(&kid).unwrap().outputs.push(tid);
        self.visited.insert(tid, (kid, op_id));
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, kid={kid:?}, op_id={op_id:?}");
        tid
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
            let kernel_id = KernelId::from(
                self.kernel_map
                    .values()
                    .copied()
                    .max()
                    .map_or(0, |id| usize::from(id).checked_add(1).unwrap()),
            );
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
        let kd = self.kernel_data.remove(&kid).unwrap();

        debug_assert!(kd.outputs.is_empty(), "all outputs must be stored before materialize");

        let loads = kd.loads.clone();
        let store_tids: Vec<TensorId> = kd.stores; // already deduplicated by add_store

        debug_assert!(
            loads.iter().all(|&tid| {
                self.buffer_map.contains_key(&tid)
                    || kd.outputs.contains(&tid)
                    || self
                        .kernel_data
                        .values()
                        .any(|d| d.outputs.contains(&tid) || d.stores.contains(&tid))
            }),
            "load tid must be realized or in kernel's own outputs, or in some other kernel's outputs/stores"
        );

        // Debug: ensure each store tid is in exactly one kernel's outputs
        if cfg!(debug_assertions) {
            for &tid in &store_tids {
                let count = self.kernel_data.iter().filter(|(_, d)| d.outputs.contains(&tid)).count();
                debug_assert!(count <= 1, "store tid={tid} is in {count} kernels' outputs");
            }
        }

        // Recursively materialize any un-realized loads first via add_store.
        // A kernel must never both load and store the same tid — assert this.
        let unrealized_tensors: Set<TensorId> = loads.iter().filter(|x| !self.buffer_map.contains_key(&x)).copied().collect();
        // We have to materialize kernel that produces tid
        // But which one is it?
        // TODO later replace this crude search with something better
        let kids: Vec<KernelId> = self.kernels.ids().collect();
        for id in kids {
            println!("yo kid={id:?}");
            if self.kernel_data[&id].stores.iter().any(|x| unrealized_tensors.contains(x)) {
                for output in self.kernel_data[&id].outputs.clone().into_iter() {
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
        let dev_id = *dev_ids
            .first()
            .ok_or_else(|| ZyxError::AllocationError("no available device".into()))?;
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
                let dst_global = BufferId {
                    pool: pool_id,
                    buffer: dst,
                };
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
        for &tid in &store_tids {
            let bytes = self.shape(tid).iter().product::<Dim>() as usize * (self.dtype(tid).bit_size() / 8) as usize;
            let alloc_bytes = bytes as Dim + Dim::from(self.dtype(tid).bit_size() / 8);
            let (buf, event) = self.pools[pool_id].allocate(alloc_bytes)?;
            let global_id = BufferId {
                pool: pool_id,
                buffer: buf,
            };
            self.buffer_map.insert(tid, global_id);
            kernel_buffers.insert(global_id);
            event_wait_list.push(event);
        }

        // Remove kernel from the slab (it is consumed)
        let kernel = unsafe {
            eprintln!("F: kernels.remove_and_return({kid:?})");
            self.kernels.remove_and_return(kid)
        };

        // Build args: load buffers first, then store buffers
        let mut args = Vec::new();
        for &tid in &loads {
            args.push(self.buffer_map[&tid].buffer);
        }
        for &tid in &store_tids {
            args.push(self.buffer_map[&tid].buffer);
        }

        // Compile and launch (caches in kernel_map / programs)
        let debug = self.debug;
        if debug.sched() {
            eprintln!("tensors: {:?}", self.tensors.ids().collect::<Vec<TensorId>>());
            eprintln!("loads (tids): {loads:?}");
            eprintln!("stores (tids): {store_tids:?}");
            for (info_kid, info_kd) in &self.kernel_data {
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
            let dev_info_id = DeviceInfoId(
                self.device_infos
                    .values()
                    .copied()
                    .max()
                    .map_or(0, |id| id.0.checked_add(1).unwrap()),
            );
            let newly_inserted = self.device_infos.insert(device_info.clone(), dev_info_id).is_none();
            assert!(newly_inserted);
            dev_info_id
        }
    }

    pub fn load<T: Scalar>(&mut self, x: TensorId, data: &mut [T]) -> Result<(), ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::load(x={x})");
        let dt = self.tensors[x].dtype;
        if dt != T::dtype() {
            return Err(ZyxError::DTypeError(
                format!("loading dtype {}, but the data has dtype {dt}", T::dtype()).into(),
            ));
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
                    println!("  -> x={x}, kid={:?}, op_id={:?}", self.visited[&x].0, self.visited[&x].1);
                    return Ok(());
                }
            }
            self.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, Vec::new())?;
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> x={x}, kid={:?}, op_id={:?}", self.visited[&x].0, self.visited[&x].1);
            return Ok(());
        }

        // Slow path: add store for each output, last one triggers materialize
        self.initialize_devices()?;
        let (kid, _) = self.visited[&x];
        // Deduplicate: add_store removes ALL occurrences at once and creates a load kernel,
        // so we must process each unique tid only once
        let mut seen = Set::default();
        let outputs: Vec<TensorId> = self.kernel_data[&kid]
            .outputs
            .iter()
            .copied()
            .filter(|&tid| seen.insert(tid))
            .collect();
        for &tid in &outputs {
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
                println!("  -> x={x}, kid={:?}, op_id={:?}", self.visited[&x].0, self.visited[&x].1);
                return Ok(());
            }
        }
        self.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, Vec::new())?;
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> x={x}, kid={:?}, op_id={:?}", self.visited[&x].0, self.visited[&x].1);
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
            let mut timings: Vec<_> = lock
                .iter()
                .map(|(name, &(total_us, count))| (name.clone(), total_us, count))
                .collect();
            timings.sort_by_key(|a| std::cmp::Reverse(a.1));
            println!("\n=== Timing Info (sorted by total time, descending) ===");
            for (name, total_us, count) in timings {
                let per_call = total_us.checked_div(count).unwrap_or(0);
                println!("{name}: {total_us}us total, {per_call}us/call ({count} calls)");
            }
        }
        //println!("DEINIT runtime");
        self.tensors = Slab::new();
        self.shape_map = Map::default();
        self.shapes = Slab::new();
        self.visited = Map::default();
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
