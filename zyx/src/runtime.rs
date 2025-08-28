//! Runtime handles tensor graph and connects tensors to device buffers.
use crate::backend::{BufferId, Config, Device, Event, MemoryPool, SearchConfig};
use crate::dtype::{Constant, DType};
use crate::error::ZyxError;
use crate::graph::{BOp, Graph, Node, ROp, UOp};
use crate::kernel::Cache;
use crate::rng::Rng;
use crate::scalar::Scalar;
use crate::shape::{Axis, Dim, permute, reduce};
use crate::tensor::TensorId;
use crate::{DebugMask, Map, Set};
use nanoserde::DeJson;
use std::collections::BTreeSet;
use std::env;
use std::hash::BuildHasherDefault;
use std::path::{Path, PathBuf};
use std::{vec, vec::Vec};

// Maximum number of constants to cache. Too high number will cause lot of specialized kernels to be generated,
// which is unnecessary.
const NUM_CONSTANTS: usize = 32;

/// This is the whole global state of zyx
pub struct Runtime {
    /// Current graph of tensor operations as nodes
    pub graph: Graph,
    /// Physical memory pools
    pub pools: Vec<Pool>,
    /// Physical compute devices, each has their own program cache
    pub devices: Vec<Device>,
    /// Kernel and optimizer cache, maps between unoptimized kernels and available/done optimizations and cached kernels
    pub cache: Cache,
    /// Zyx configuration directory path
    config_dir: Option<PathBuf>, // Why the hell isn't PathBuf::new const?????
    /// Random number generator
    pub rng: Rng,
    /// Are we in training mode?
    pub training: bool,
    /// Search configuration
    pub search_config: SearchConfig,
    /// Debug mask
    pub debug: DebugMask,
    /// Temporary storage, TODO limit the number of elements in temporary storage
    temp_data: Vec<Box<dyn TempData>>,
    /// Cache for constants
    constants: [Constant; NUM_CONSTANTS],
    /// Current number of constants
    constants_len: usize,
    /// Enables implicit casting to different dtype in binary operations with different dtypes
    /// and unary operations that are not implemented for the provided dtype.
    /// This tries to copy the default behaviour of pytorch, but since rust does not
    /// have implicit casting, we do not recommend using this feature.
    pub implicit_casts: bool,
}

pub trait TempData: Send {
    fn read(&self) -> &[u8];
    fn bytes(&self) -> Dim;
    fn dtype(&self) -> DType;
}

#[derive(Debug)]
pub struct Pool {
    #[allow(clippy::struct_field_names)]
    pub pool: MemoryPool,
    pub events: Map<BTreeSet<BufferId>, Event>,
    pub buffer_map: Map<TensorId, BufferId>,
}

impl Pool {
    pub(crate) fn new(pool: MemoryPool) -> Self {
        Self {
            pool,
            events: Map::with_capacity_and_hasher(100, BuildHasherDefault::default()),
            buffer_map: Map::with_capacity_and_hasher(100, BuildHasherDefault::default()),
        }
    }
}

fn get_mut_buffer(pools: &mut [Pool], tensor_id: TensorId) -> Option<(&mut Pool, BufferId)> {
    for pool in pools {
        if let Some(&id) = pool.buffer_map.get(&tensor_id) {
            return Some((pool, id));
        }
    }
    None
}

impl Runtime {
    #[must_use]
    pub(super) const fn new() -> Self {
        Runtime {
            graph: Graph::new(),
            devices: Vec::new(),
            pools: Vec::new(),
            rng: Rng::seed_from_u64(42069),
            config_dir: None,
            cache: Cache::new(),
            training: false,
            search_config: SearchConfig::new(),
            debug: DebugMask(0),
            temp_data: Vec::new(),
            constants: [Constant::I32(0); NUM_CONSTANTS],
            constants_len: 0,
            implicit_casts: true,
        }
    }

    pub(super) fn retain(&mut self, x: TensorId) {
        self.graph.retain(x);
    }

    pub(super) fn release(&mut self, x: TensorId) {
        let to_remove = self.graph.release(x);
        self.deallocate_tensors(&to_remove);
        if self.graph.is_empty() && self.pools.iter().all(|mp| mp.buffer_map.is_empty()) {
            self.deinitialize();
        }
    }

    // Initializes all available devices, creating a device for each compute
    // device and a memory pool for each physical memory.
    // Does nothing if devices were already initialized.
    // Returns error if all devices failed to initialize
    // DeviceParameters allows to disable some devices if requested
    pub(crate) fn initialize_devices(&mut self) -> Result<(), ZyxError> {
        if !self.devices.is_empty() {
            return Ok(());
        }

        // Set env vars
        if let Ok(x) = std::env::var("ZYX_DEBUG")
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
        if let Some(mut path) = self.config_dir.clone() {
            path.push("cached_kernels");
            if let Ok(mut file) = std::fs::File::open(path) {
                use std::io::Read;
                let mut buf = Vec::new();
                file.read_to_end(&mut buf).unwrap();
                if let Ok(kernel_cache) = nanoserde::DeBin::deserialize_bin(&buf) {
                    self.cache = kernel_cache;
                }
            }
        }

        crate::backend::initialize_backends(&config, &mut self.pools, &mut self.devices, self.debug.dev())?;
        self.pools.shrink_to_fit();
        self.devices.shrink_to_fit();

        self.search_config = config.search;
        Ok(())
    }

    /// This function deinitializes the whole runtime, deallocates all allocated memory and deallocates all caches
    /// It does not reset the rng and it does not change debug, search, training and `config_dir` fields
    fn deinitialize(&mut self) {
        //println!("Deinitialize");
        // drop graph
        self.graph = Graph::new();
        // Drop programs
        self.cache.deinitialize(&mut self.devices);
        // drop devices
        while let Some(mut dev) = self.devices.pop() {
            dev.deinitialize();
        }
        // drop memory pools
        while let Some(mp) = self.pools.pop() {
            let Pool { mut pool, events, .. } = mp;
            pool.release_events(events.into_values().collect());
            pool.deinitialize();
        }
        // Timer
        /*for (name, (iters, time)) in crate::ET.lock().iter() {
            println!("Timer {name} took {time}us for {iters} iterations, {}us/iter", time/iters);
        }*/
        self.config_dir = None;
        self.temp_data = Vec::new();
        // These variables are persistent:
        /*self.rng
        self.training
        self.search_iterations
        self.debug*/
    }

    pub(super) const fn manual_seed(&mut self, seed: u64) {
        self.rng = Rng::seed_from_u64(seed);
    }

    /// Creates dot plot of graph between given tensors
    #[must_use]
    pub(super) fn plot_dot_graph(&self, tensors: &Set<TensorId>) -> String {
        //println!("Tensor storage {:?}", self.tensor_buffer_map);
        self.graph.plot_dot_graph(tensors, &self.pools)
    }

    #[must_use]
    pub(super) fn shape(&self, x: TensorId) -> &[Dim] {
        self.graph.shape(x)
    }

    #[must_use]
    pub(super) fn dtype(&self, x: TensorId) -> DType {
        self.graph.dtype(x)
    }

    pub(super) fn tensor_from_path(
        &mut self,
        shape: Vec<Dim>,
        dtype: DType,
        path: &Path,
        offset_bytes: u64,
    ) -> Result<TensorId, ZyxError> {
        let bytes = shape.iter().product::<Dim>() * dtype.byte_size() as Dim;
        self.initialize_devices()?;
        if bytes == dtype.byte_size() as Dim {
            /*let value = data.read();
            let value = Constant::from_bytes(value, dtype);
            if self.constants_len < NUM_CONSTANTS {
                if !self.constants.contains(&value) {
                    self.constants[self.constants_len] = value;
                    self.constants_len += 1;
                }
                return Ok(self.graph.push(Node::Const { value }));
            } else if self.constants.contains(&value) {
                return Ok(self.graph.push(Node::Const { value }));
            }*/
            todo!();
        }
        if let Some(disk) = self.pools[0].pool.disk_pool() {
            let buffer_id = disk.buffer_from_path(bytes, path, offset_bytes);
            let id = self.graph.push_wshape(Node::Leaf { dtype }, shape);
            self.pools[0].buffer_map.insert(id, buffer_id);
            Ok(id)
        } else {
            Err(ZyxError::NoBackendAvailable)
        }
    }

    pub(super) fn new_tensor(&mut self, shape: Vec<Dim>, data: Box<dyn TempData>) -> Result<TensorId, ZyxError> {
        let bytes = data.bytes();
        let dtype = data.dtype();
        debug_assert_eq!(shape.iter().product::<Dim>() * dtype.byte_size() as Dim, bytes);
        if bytes == dtype.byte_size() as Dim {
            let value = data.read();
            let value = Constant::from_le_bytes(value, dtype);
            if self.constants_len < NUM_CONSTANTS {
                if !self.constants.contains(&value) {
                    self.constants[self.constants_len] = value;
                    self.constants_len += 1;
                }
                return Ok(self.graph.push(Node::Const { value }));
            } else if self.constants.contains(&value) {
                return Ok(self.graph.push(Node::Const { value }));
            }
        }
        self.initialize_devices()?;
        // Put it into memory pool with fastest device out of memory pools with enough free capacity
        let mem_pools: Vec<u32> = self
            .pools
            .iter()
            .enumerate()
            .filter_map(|(id, mp)| {
                if mp.pool.free_bytes() > bytes {
                    Some(u32::try_from(id).unwrap())
                } else {
                    None
                }
            })
            .collect();
        if mem_pools.is_empty() {
            return Err(ZyxError::AllocationError);
        }
        //println!("Memory pools: {mem_pools:?}");
        // Pick memory pool with fastest device
        let mut memory_pool_id = mem_pools[0];
        let mut max_compute = 0;
        for dev in &self.devices {
            let mpid = dev.memory_pool_id();
            //println!("Compute: {}, id: {mpid}", dev.free_compute());
            if dev.free_compute() > max_compute && mem_pools.contains(&mpid) {
                max_compute = dev.free_compute();
                memory_pool_id = mpid;
            }
        }
        let mpid = memory_pool_id as usize;
        let (buffer_id, event) = self.pools[mpid].pool.allocate(bytes)?;
        self.temp_data.push(data);
        let event =
            self.pools[mpid].pool.host_to_pool(self.temp_data.last().unwrap().read(), buffer_id, vec![event])?;
        let id = self.graph.push_wshape(Node::Leaf { dtype }, shape);
        self.pools[mpid].buffer_map.insert(id, buffer_id);
        self.pools[mpid].events.insert([buffer_id].into(), event);
        Ok(id)
    }

    #[must_use]
    pub(super) fn constant(&mut self, value: impl Scalar) -> TensorId {
        self.graph.push(Node::Const { value: Constant::new(value) })
    }

    // Initialization
    pub(super) fn full(&mut self, shape: Vec<Dim>, value: impl Scalar) -> TensorId {
        let x = self.constant(value);
        let expanded = self.expand(x, shape).unwrap();
        self.release(x);
        expanded
    }

    #[must_use]
    pub(super) fn ones(&mut self, shape: Vec<Dim>, dtype: DType) -> TensorId {
        let x = self.graph.push(Node::Const { value: dtype.one_constant() });
        let expanded = self.expand(x, shape).unwrap();
        self.release(x);
        expanded
    }

    #[must_use]
    pub(super) fn zeros(&mut self, shape: Vec<Dim>, dtype: DType) -> TensorId {
        let x = self.graph.push(Node::Const { value: dtype.zero_constant() });
        let expanded = self.expand(x, shape).unwrap();
        self.release(x);
        expanded
    }

    /// Bitcast self to other type, currently immediatelly realizes the tensor.
    /// The caller is responsible for ensuring that destination dtype is representable
    /// with bytes of source data.
    pub(super) unsafe fn bitcast(&mut self, x: TensorId, dtype: DType) -> Result<TensorId, ZyxError> {
        if dtype == self.dtype(x) {
            self.retain(x);
            return Ok(x);
        }
        let mut to_eval = Set::with_capacity_and_hasher(10, BuildHasherDefault::default());
        to_eval.insert(x);
        self.realize(&to_eval)?;
        let mut shape = self.shape(x).to_vec();
        // We create a new pointer in tensor_buffer_map to the same buffer
        // and create a new Leaf in graph
        //self.tensor_buffer_map.find();
        let cd = dtype.byte_size() as Dim / self.dtype(x).byte_size() as Dim;
        if let Some(d) = shape.last_mut() {
            if *d % cd != 0 {
                return Err(ZyxError::DTypeError(
                    "Can't bitcast due to tensor's last dimension not being correct multiple of dtype.".into(),
                ));
            }
            *d /= cd;
        }
        //let id = self.graph.push_wshape_and_dtype(Node::Leaf, shape.clone(), dtype);
        let id = self.graph.push_wshape(Node::Leaf { dtype }, shape.clone());
        if let Some(pool) = self.pools.iter_mut().find(|pool| pool.buffer_map.contains_key(&x)) {
            //println!("Bitcast {x}, res {id}, new shape {shape:?} buffer id {bid:?}");
            let x = *pool.buffer_map.get(&x).unwrap();
            pool.buffer_map.insert(id, x);
        }
        //println!("TBM:\n{:?}", self.tensor_buffer_map);
        Ok(id)
    }

    #[must_use]
    pub(super) fn reshape(&mut self, x: TensorId, shape: Vec<Dim>) -> TensorId {
        //println!("reshaping to {shape:?}, {:?}", self.shape(x));
        let sh = self.shape(x);
        debug_assert_eq!(shape.iter().product::<Dim>(), sh.iter().product::<Dim>());
        if shape == sh {
            self.retain(x);
            return x;
        }
        let id = self.graph.push_wshape(Node::Reshape { x }, shape);
        // Reshape on realized variable is NOOP, buffer_maps trace ownership
        if let Some((pool, bid)) = get_mut_buffer(&mut self.pools, x) {
            pool.buffer_map.insert(id, bid);
        }
        id
    }

    /// Expand verification is so complex, that it may be simpler to just do it here instead of in tensor
    pub(super) fn expand(&mut self, x: TensorId, shape: Vec<Dim>) -> Result<TensorId, ZyxError> {
        let sh: Vec<Dim> = self.shape(x).into();
        if shape == sh {
            self.retain(x);
            return Ok(x);
        }
        //println!("Expanding {x} from {sh:?} to {shape:?}");
        if shape.len() < sh.len() {
            return Err(ZyxError::ShapeError(
                format!("Cannot expand {sh:?} into {shape:?}").into(),
            ));
        }
        let mut expanded = false;
        let new_shape = if shape.len() > sh.len() {
            expanded = true;
            std::iter::repeat_n(1, shape.len() - sh.len()).chain(sh.iter().copied()).collect()
        } else {
            sh
        };
        debug_assert_eq!(shape.len(), new_shape.len());
        for (&s, &d) in new_shape.iter().zip(shape.iter()) {
            if !(s == d || s == 1) {
                return Err(ZyxError::ShapeError(
                    format!("Cannot expand {new_shape:?} into {shape:?}").into(),
                ));
            }
        }
        if expanded {
            let x = self.reshape(x, new_shape.clone());
            if shape == new_shape {
                self.retain(x);
                return Ok(x);
            }
            let y = self.graph.push_wshape(Node::Expand { x }, shape);
            self.release(x);
            return Ok(y);
        }
        Ok(self.graph.push_wshape(Node::Expand { x }, shape))
    }

    #[must_use]
    pub(super) fn permute(&mut self, x: TensorId, axes: &[Axis]) -> TensorId {
        if axes.len() < 2 || axes == (0..axes.len() as Axis).collect::<Vec<Axis>>() {
            self.retain(x);
            return x;
        }
        let shape = permute(self.shape(x), axes);
        let id = self.graph.push_wshape(Node::Permute { x }, shape);
        self.graph.push_axes(id, axes.to_vec());
        id
    }

    #[must_use]
    pub(super) fn pad_zeros(&mut self, x: TensorId, padding: Vec<(isize, isize)>) -> TensorId {
        let mut shape: Vec<Dim> = self.shape(x).into();
        //println!("Self shape: {shape:?}, padding: {padding:?}");
        apply_padding(&mut shape, &padding);
        let id = self.graph.push_wshape(Node::Pad { x }, shape);
        self.graph.push_padding(id, padding);
        id
    }

    #[must_use]
    pub(super) fn sum_reduce(&mut self, x: TensorId, mut axes: Vec<Axis>) -> TensorId {
        let sh = self.shape(x);
        if axes.is_empty() {
            axes = (0..sh.len() as Axis).collect();
        }
        let shape = reduce(sh, &axes);
        let id = self.graph.push_wshape(Node::Reduce { x, rop: ROp::Sum }, shape);
        self.graph.push_axes(id, axes);
        id
    }

    #[must_use]
    pub(super) fn max_reduce(&mut self, x: TensorId, mut axes: Vec<Axis>) -> TensorId {
        let sh = self.shape(x);
        if axes.is_empty() {
            axes = (0..sh.len() as Axis).collect();
        }
        let shape = reduce(sh, &axes);
        let id = self.graph.push_wshape(Node::Reduce { x, rop: ROp::Max }, shape);
        self.graph.push_axes(id, axes);
        id
    }

    // Unary ops
    #[must_use]
    pub(super) fn cast(&mut self, x: TensorId, dtype: DType) -> TensorId {
        if dtype == self.dtype(x) {
            self.retain(x);
            return x;
        }
        //self.graph.push_wdtype(Node::Unary { x, uop: UOp::Cast(dtype) }, dtype)
        self.graph.push(Node::Cast { x, dtype })
    }

    #[must_use]
    pub(super) fn unary(&mut self, x: TensorId, uop: UOp) -> TensorId {
        self.graph.push(Node::Unary { x, uop })
    }

    #[must_use]
    pub(super) fn binary(&mut self, x: TensorId, y: TensorId, bop: BOp) -> TensorId {
        self.graph.push(Node::Binary { x, y, bop })
    }

    /// Loads data with beginning elements of the tensor x.
    /// If `data.len()` == `x.numel()`, then it loads the whole tensor.
    pub fn load<T: Scalar>(&mut self, x: TensorId, data: &mut [T]) -> Result<(), ZyxError> {
        let n: Dim = self.shape(x).iter().product();
        let dt = self.dtype(x);
        if dt != T::dtype() {
            return Err(ZyxError::DTypeError(
                format!("loading dtype {}, but the data has dtype {dt}", T::dtype()).into(),
            ));
        }
        debug_assert!(data.len() as Dim <= n, "Return buffer is bigger than tensor");
        // Check if tensor is evaluated
        if !self.pools.iter().any(|pool| pool.buffer_map.contains_key(&x)) {
            let mut to_eval = Set::with_capacity_and_hasher(1, BuildHasherDefault::default());
            to_eval.insert(x);
            self.realize(&to_eval)?;
        }

        let (pool, buffer_id) = get_mut_buffer(&mut self.pools, x).unwrap();

        Self::load_buffer(data, pool, buffer_id)
    }

    pub fn load_buffer<T: Scalar>(data: &mut [T], pool: &mut Pool, buffer_id: BufferId) -> Result<(), ZyxError> {
        let byte_slice: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), data.len() * T::byte_size()) };
        for buffers in pool.events.keys() {
            if buffers.contains(&buffer_id) {
                let event = pool.events.remove(&buffers.clone()).unwrap();
                //println!("Loading with event {event:?}");
                pool.pool.pool_to_host(buffer_id, byte_slice, vec![event])?;
                return Ok(());
            }
        }
        pool.pool.pool_to_host(buffer_id, byte_slice, Vec::new())?;
        Ok(())
    }

    pub fn deallocate_tensors(&mut self, to_remove: &Set<TensorId>) {
        // This is basically tracing GC, seems faster than reference counting
        // remove all buffers that are not used by any tensors
        // Check which buffers will possibly need to be dropped
        for tensor_id in to_remove {
            let mut buffer = None;
            for (pool_id, pool) in self.pools.iter_mut().enumerate() {
                if let Some(buffer_id) = pool.buffer_map.remove(tensor_id) {
                    buffer = Some((pool_id, buffer_id));
                    break;
                }
            }
            if let Some((pool_id, buffer_id)) = buffer
                && !self.pools.iter().any(|pool| pool.buffer_map.values().any(|bid| *bid == buffer_id))
            {
                let pool = &mut self.pools[pool_id];
                let mut events = Vec::new();
                if let Some(key) = pool.events.keys().find(|key| key.contains(&buffer_id)) {
                    let event = pool.events.remove(&key.clone()).unwrap();
                    events.push(event);
                }
                pool.pool.deallocate(buffer_id, events);
            }
        }
    }
}

pub fn apply_padding(shape: &mut [Dim], padding: &[(isize, isize)]) {
    let mut i = 0;
    for d in shape.iter_mut().rev() {
        *d = Dim::try_from(isize::try_from(*d).unwrap() + padding[i].0 + padding[i].1).unwrap();
        i += 1;
        if i >= padding.len() {
            break;
        }
    }
}
