use std::{
    collections::BTreeSet,
    env,
    hash::BuildHasherDefault,
    path::{Path, PathBuf},
};

use nanoserde::DeJson;

use crate::{
    DType, DebugMask, Map, Scalar, ZyxError,
    backend::{AutotuneConfig, BufferId, Config, Device, DeviceInfo, Event, MemoryPool, PoolId, ProgramId},
    dtype::Constant,
    kernel::{BOp, DeviceId, Kernel, OpId, UOp},
    kernel_cache::{DeviceInfoId, KernelId},
    rng::Rng,
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    tensor::TensorId,
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
    rc: u32,
    shape: ShapeId,
    dtype: DType,
}

struct KernelData {
    outputs: Vec<TensorId>,
    loads: Vec<TensorId>,
    stores: Vec<TensorId>,
}

struct Runtime {
    tensors: Slab<TensorId, TensorData>,
    shape_map: Map<Box<[Dim]>, ShapeId>,
    shapes: Slab<ShapeId, Box<[Dim]>>,
    tensor_kernel: Map<TensorId, (KernelId, OpId)>,
    kernel_data: Map<KernelId, KernelData>,
    device_infos: Map<DeviceInfo, DeviceInfoId>,
    kernels: Map<Kernel, KernelId>,
    programs: Map<(KernelId, DeviceId), ProgramId>,
    devices: Slab<DeviceId, Device>,
    pools: Slab<PoolId, MemoryPool>,
    config_dir: Option<PathBuf>,
    buffer_map: Map<TensorId, BufferId>,
    events: Map<BTreeSet<BufferId>, Event>,
    rng: Rng,
    autotune_config: AutotuneConfig,
    debug: DebugMask,
}

impl Runtime {
    pub const fn new() -> Self {
        Runtime {
            tensors: Slab::new(),
            shape_map: Map::with_hasher(BuildHasherDefault::new()),
            shapes: Slab::new(),
            tensor_kernel: Map::with_hasher(BuildHasherDefault::new()),
            kernel_data: Map::with_hasher(BuildHasherDefault::new()),
            device_infos: Map::with_hasher(BuildHasherDefault::new()),
            kernels: Map::with_hasher(BuildHasherDefault::new()),
            programs: Map::with_hasher(BuildHasherDefault::new()),
            devices: Slab::new(),
            pools: Slab::new(),
            config_dir: None,
            buffer_map: Map::with_hasher(BuildHasherDefault::new()),
            events: Map::with_hasher(BuildHasherDefault::new()),
            rng: Rng::seed_from_u64(42069),
            autotune_config: AutotuneConfig::new(),
            debug: DebugMask::new(0),
        }
    }

    pub fn retain(&mut self, x: TensorId) {
        self.tensors[x].rc += 1;
    }

    pub fn release(&mut self, x: TensorId) {}

    pub fn new_constant_tensor(&mut self, value: Constant) -> TensorId {
        todo!()
    }

    pub fn new_disk_tensor(
        &mut self,
        shape: Box<[Dim]>,
        dtype: DType,
        path: &Path,
        offset_bytes: u64,
    ) -> Result<TensorId, ZyxError> {
        /*let bytes = shape.iter().product::<Dim>() * Dim::from(dtype.bit_size() / 8);
        self.initialize_devices()?;
        if let Some(disk) = self.pools[PoolId::from(1)].disk_pool() {
            let buffer_id = disk.buffer_from_path(bytes, path, offset_bytes);
            let id = self.graph.push_wshape(Node::Leaf { dtype }, shape);
            self.buffer_map.insert(
                id,
                BufferId {
                    pool: PoolId::from(1),
                    buffer: buffer_id,
                },
            );
            Ok(id)
        } else {
            Err(ZyxError::NoBackendAvailable)
        }*/
        todo!()
    }

    // Creates new tensor in host memory
    pub fn new_host_tensor<T: Scalar>(&mut self, shape: Box<[Dim]>, data: Box<[T]>) -> Result<TensorId, ZyxError> {
        let shape_id = if let Some(&shape_id) = self.shape_map.get(&shape) {
            shape_id
        } else {
            let shape_id = self.shapes.push(shape.clone());
            self.shape_map.insert(shape, shape_id);
            shape_id
        };
        todo!()
    }

    pub fn new_zeros(&mut self, shape: Box<[Dim]>, dtype: DType) -> TensorId {
        todo!()
    }

    pub fn new_ones(&mut self, shape: Box<[Dim]>, dtype: DType) -> TensorId {
        todo!()
    }

    pub fn new_full<T: Scalar>(&mut self, shape: Box<[Dim]>, value: T) -> TensorId {
        todo!()
    }

    pub fn cast(&mut self, x: TensorId, dtype: DType) -> TensorId {
        todo!()
    }

    pub fn bitcast(&mut self, x: TensorId, dtype: DType) -> TensorId {
        todo!()
    }

    pub fn unary(&mut self, x: TensorId, uop: UOp) -> TensorId {
        todo!()
    }

    pub fn binary(&mut self, x: TensorId, bop: BOp) -> Result<TensorId, ZyxError> {
        todo!()
    }

    pub fn reduce(&mut self, x: TensorId, rop: BOp, axes: Box<[UAxis]>) -> Result<TensorId, ZyxError> {
        todo!()
    }

    pub fn to_device(&mut self) -> Result<TensorId, ZyxError> {
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
        self.tensor_kernel = Map::default();
    }

    pub const fn manual_seed(&mut self, seed: u64) {
        self.rng = Rng::seed_from_u64(seed);
    }
}
