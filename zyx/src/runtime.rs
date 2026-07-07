use std::{
    collections::BTreeSet,
    env,
    hash::BuildHasherDefault,
    path::{Path, PathBuf},
};

use nanoserde::DeJson;

use crate::{
    DType, DebugMask, Map, Scalar, ZyxError,
    backend::{AutotuneConfig, BufferId, Config, Device, DeviceInfo, Event, MemoryPool, OpCapability, PoolId, ProgramId},
    dtype::Constant,
    kernel::{BOp, DeviceId, Kernel, Op, OpId, UOp},
    kernel_cache::{DeviceInfoId, KernelId},
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
    programs: Map<KernelId, ProgramId>,
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
        let kid = self.visited.get_mut(&x).unwrap().0;
        self.kernel_data.get_mut(&kid).unwrap().outputs.push(x);
    }

    pub fn release(&mut self, x: TensorId) {
        let kid = self.visited.get_mut(&x).unwrap().0;
        let outputs = &mut self.kernel_data.get_mut(&kid).unwrap().outputs;
        outputs.push(x);
        if outputs.is_empty() {
            todo!("check if we can remove kernel or realize one")
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
        let op = Op::ConstView(Box::new((value, View::contiguous(&[1]))));
        self.new_kernel(op, [1].into(), value.dtype())
    }

    pub fn new_full(&mut self, shape: Vec<Dim>, value: Constant) -> TensorId {
        let dtype = value.dtype();
        let op = Op::ConstView(Box::new((value, View::contiguous(&[1]))));
        let x = self.new_kernel(op, [1].into(), dtype);
        let expanded = self.expand(x, shape).unwrap();
        self.release(x);
        expanded
    }

    // Creates new tensor in host memory
    pub fn new_host_tensor<T: Scalar>(&mut self, shape: Vec<Dim>, data: Box<[T]>) -> Result<TensorId, ZyxError> {
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
        let shape_id = self.tensors[x].shape_id;
        let tid = self.tensors.push(TensorData { shape_id, dtype });
        let (kid, op_id) = self.visited[&x];
        let op_id = self.kernels[kid].cast(op_id, dtype);
        self.kernel_data.get_mut(&kid).unwrap().outputs.push(tid);
        self.visited.insert(tid, (kid, op_id));
        tid
    }

    pub fn bitcast(&mut self, x: TensorId, dtype: DType) -> TensorId {
        let shape_id = self.tensors[x].shape_id;
        let tid = self.tensors.push(TensorData { shape_id, dtype });
        let (kid, op_id) = self.visited[&x];
        let op_id = self.kernels[kid].bitcast(op_id, dtype);
        self.kernel_data.get_mut(&kid).unwrap().outputs.push(tid);
        self.visited.insert(tid, (kid, op_id));
        tid
    }

    pub fn unary(&mut self, x: TensorId, uop: UOp) -> TensorId {
        let shape_id = self.tensors[x].shape_id;
        let dtype = self.tensors[x].dtype;
        let tid = self.tensors.push(TensorData { shape_id, dtype });
        let (kid, op_id) = self.visited[&x];
        let op_id = self.kernels[kid].unary(op_id, uop);
        self.kernel_data.get_mut(&kid).unwrap().outputs.push(tid);
        self.visited.insert(tid, (kid, op_id));
        tid
    }

    pub fn binary(&mut self, x: TensorId, y: TensorId, bop: BOp) -> Result<TensorId, ZyxError> {
        todo!()
    }

    pub fn reduce(&mut self, x: TensorId, axes: Vec<UAxis>, rop: BOp) -> Result<TensorId, ZyxError> {
        todo!()
    }

    pub fn to_device(&mut self, x: TensorId, device_id: DeviceId) -> Result<TensorId, ZyxError> {
        todo!()
    }

    pub(super) fn reshape(&mut self, x: TensorId, shape: Vec<Dim>) -> TensorId {
        todo!()
    }

    pub fn expand(&mut self, x: TensorId, shape: Vec<Dim>) -> Result<TensorId, ZyxError> {
        todo!()
    }

    pub fn permute(&mut self, x: TensorId, axes: Vec<UAxis>) -> TensorId {
        todo!()
    }

    pub fn pad_zeros(&mut self, x: TensorId, padding: Vec<(i64, i64)>) -> TensorId {
        todo!()
    }

    pub fn load<T: Scalar>(&mut self, x: TensorId, data: &mut [T]) -> Result<(), ZyxError> {
        todo!()
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
