// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Runtime handles tensor graph and connects tensors to device buffers.
use crate::backend::{AutotuneConfig, BufferId, Config, Device, DeviceId, Event, MemoryPool, PoolId};
use crate::dtype::{Constant, DType};
use crate::error::ZyxError;
use crate::graph::compiled::{CachedGraph, CompiledGraph};
use crate::graph::{Graph, Node};
use crate::kernel::{BOp, UOp};
use crate::kernel_cache::KernelCache;
use crate::rng::Rng;
use crate::scalar::Scalar;
use crate::shape::{Dim, UAxis, permute, reduce};
use crate::slab::Slab;
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
    /// Physical compute devices, each has their own program cache
    pub devices: Slab<DeviceId, Device>,
    /// Physical memory pools
    pub pools: Slab<PoolId, MemoryPool>,
    /// Global mapping from tensor ID to (`pool_index`, `buffer_id`).
    pub buffer_map: Map<TensorId, BufferId>,
    /// Global event tracking - all buffers with pending events
    pub events: Map<BTreeSet<BufferId>, Event>,
    /// Kernel and optimizer cache, maps between unoptimized kernels and available/done optimizations and cached kernels
    pub kernel_cache: KernelCache,
    /// Zyx configuration directory path
    pub config_dir: Option<PathBuf>,
    /// Random number generator
    pub rng: Rng,
    /// Autotune configuration
    pub autotune_config: AutotuneConfig,
    /// Debug mask
    pub debug: DebugMask,
    /// Temporary storage
    pub temp_data: Map<BufferId, Box<[u8]>>,
    /// Cache for constants
    constants: [Constant; NUM_CONSTANTS],
    /// Current number of constants
    constants_len: usize,
    /// Enables implicit casting to different dtype in binary operations with different dtypes
    /// and unary operations that are not implemented for the provided dtype.
    /// This tries to copy the default behaviour of pytorch, but since rust does not
    /// have implicit casting, we do not recommend using this feature.
    pub implicit_casts: bool,
    /// Are we in training mode?
    pub training: bool,
    // Cache for compiled kernels, maps kernels to compiled result.
    #[allow(unused)]
    pub(crate) graph_cache: Map<CachedGraph, CompiledGraph>,
}

pub trait TempData: Send {
    fn read(&self) -> Box<[u8]>;
    fn bytes(&self) -> Dim;
    fn dtype(&self) -> DType;
}

fn get_mut_buffer(buffer_map: &Map<TensorId, BufferId>, tensor_id: TensorId) -> Option<BufferId> {
    buffer_map.get(&tensor_id).copied()
}

impl Runtime {
    #[must_use]
    pub(super) const fn new() -> Self {
        Runtime {
            graph: Graph::new(),
            devices: Slab::new(),
            pools: Slab::new(),
            buffer_map: Map::with_hasher(BuildHasherDefault::new()),
            events: Map::with_hasher(BuildHasherDefault::new()),
            rng: Rng::seed_from_u64(42069),
            config_dir: None,
            kernel_cache: KernelCache::new(),
            training: false,
            autotune_config: AutotuneConfig::new(),
            debug: DebugMask(0),
            temp_data: Map::with_hasher(BuildHasherDefault::new()),
            constants: [Constant::I32(0); NUM_CONSTANTS],
            constants_len: 0,
            implicit_casts: true,
            graph_cache: Map::with_hasher(BuildHasherDefault::new()),
        }
    }

    pub fn is_realized(&self, x: TensorId) -> bool {
        self.buffer_map.contains_key(&x)
    }

    /// Returns true if any device supports the given dtype.
    pub fn supports_dtype(&self, dtype: DType) -> bool {
        self.devices.iter().any(|(_, d)| d.info().supports_dtype(dtype))
    }

    pub fn debug_graph(&self) {
        for (id, (rc, node)) in self.graph.nodes.iter() {
            println!("{id} x {rc} -> {node:?} {:?} {:?}", self.shape(id), self.dtype(id));
        }
    }

    pub(super) fn retain(&mut self, x: TensorId) {
        self.graph.retain(x);
    }

    pub(super) fn release(&mut self, x: TensorId) {
        let to_remove = self.graph.release(&[x]);
        deallocate_tensors(
            &to_remove,
            &mut self.pools,
            &mut self.events,
            &mut self.buffer_map,
            &mut self.temp_data,
        );
        if self.graph.is_empty() && self.buffer_map.is_empty() {
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
                if let Ok(cache) = nanoserde::DeBin::deserialize_bin(&buf) {
                    self.kernel_cache = cache;
                }
            }
        }

        crate::backend::initialize_backends(&config, &mut self.pools, &mut self.devices, self.debug.dev())?;

        self.autotune_config = config.autotune;
        //println!("INIT runtime");
        Ok(())
    }

    /// This function deinitializes the whole runtime, deallocates all allocated memory and deallocates all caches
    /// It does not reset the rng and it does not change debug, search, training and `config_dir` fields
    fn deinitialize(&mut self) {
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
        // drop graph
        self.graph = Graph::new();

        /*
        // It seems there is no point in actually deinitializing anything...
        // Drop programs (kernels)
        self.cache.deinitialize(&mut self.devices);
        // drop devices
        while let Some(mut dev) = self.devices.pop() {
            dev.deinitialize();
        }
        // drop memory pools
        while let Some(mut pool) = self.pools.pop() {
            pool.release_events(self.events.values().cloned().collect());
            pool.deinitialize();
        }
        self.config_dir = None;
        */

        // These variables are persistent:
        /*self.rng
        self.training
        self.search_iterations
        self.debug*/

        // Timer
        /*for (name, (time, iters)) in crate::ET.lock().iter() {
            println!(
                "Timer {name} took {time}us for {iters} iterations, {}us/iter",
                time / iters
            );
        }*/
    }

    pub(super) const fn manual_seed(&mut self, seed: u64) {
        self.rng = Rng::seed_from_u64(seed);
    }

    /// Creates dot plot of graph between given tensors
    #[must_use]
    pub(super) fn plot_dot_graph(&self, tensors: &Set<TensorId>) -> String {
        //println!("Tensor storage {:?}", self.tensor_buffer_map);
        self.graph.plot_dot_graph(tensors, &self.buffer_map)
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
        let bytes = shape.iter().product::<Dim>() * Dim::from(dtype.bit_size() / 8);
        self.initialize_devices()?;
        if let Some(disk) = self.pools[PoolId::from(1)].disk_pool() {
            let buffer_id = disk.buffer_from_path(bytes, path, offset_bytes);
            let id = self.graph.push_wshape(Node::Leaf { dtype }, shape);
            self.buffer_map
                .insert(id, BufferId { pool: PoolId::from(1), buffer: buffer_id });
            Ok(id)
        } else {
            Err(ZyxError::NoBackendAvailable)
        }
    }

    pub(super) fn new_constant(&mut self, value: Constant) -> TensorId {
        self.constants[self.constants_len] = value;
        self.constants_len += 1;
        self.graph.push(Node::Const { value })
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(super) fn new_tensor(&mut self, shape: Vec<Dim>, data: impl TempData) -> Result<TensorId, ZyxError> {
        let bytes = data.bytes();
        let dtype = data.dtype();
        //println!("bytes={} dtype={} shape={:?}", data.bytes(), data.dtype(), shape);
        debug_assert_eq!(shape.iter().product::<Dim>() * Dim::from(dtype.bit_size() / 8), bytes);
        if bytes == Dim::from(dtype.bit_size() / 8) && shape.len() == 1 {
            let value = data.read();
            let value = Constant::from_le_bytes(&value, dtype);
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
        let mem_pools: Vec<PoolId> = self
            .pools
            .iter()
            .filter_map(|(id, mp)| if mp.free_bytes() > bytes { Some(id) } else { None })
            .collect();
        if mem_pools.is_empty() {
            return Err(ZyxError::AllocationError("no memory pool has been initialized.".into()));
        }
        // Pick memory pool with fastest device
        let mut memory_pool_id = mem_pools[0];
        let mut max_compute = 0;
        for (_id, dev) in self.devices.iter() {
            let mpid = dev.memory_pool_id();
            if dev.free_compute() > max_compute && mem_pools.contains(&mpid) {
                max_compute = dev.free_compute();
                memory_pool_id = mpid;
            }
        }
        let (buffer_id, event) = self.pools[memory_pool_id].allocate(bytes)?;
        let global_id = BufferId { pool: memory_pool_id, buffer: buffer_id };
        self.temp_data.insert(global_id, data.read());

        let event = self.pools[memory_pool_id].host_to_pool(&self.temp_data[&global_id], buffer_id, vec![event])?;
        let id = self.graph.push_wshape(Node::Leaf { dtype }, shape);
        self.buffer_map.insert(id, global_id);
        self.events.insert(BTreeSet::from([global_id]), event);
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
        self.realize_selected(&to_eval)?;
        let mut shape = self.shape(x).to_vec();
        // We create a new pointer in tensor_buffer_map to the same buffer
        // and create a new Leaf in graph
        //self.tensor_buffer_map.find();
        let cd = Dim::from(dtype.bit_size() / 8) / Dim::from(self.dtype(x).bit_size() / 8);
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
        if let Some(buf_id) = get_mut_buffer(&self.buffer_map, x) {
            self.buffer_map.insert(id, buf_id);
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
        if let Some(buf_id) = get_mut_buffer(&self.buffer_map, x) {
            self.buffer_map.insert(id, buf_id);
        }
        id
    }

    /// Expand verification is so complex, that it may be simpler to just do it here instead of in tensor
    pub(super) fn expand(&mut self, x: TensorId, shape: Vec<Dim>) -> Result<TensorId, ZyxError> {
        //println!("Expanding {x}");
        let sh: Vec<Dim> = self.shape(x).into();
        if shape == sh {
            self.retain(x);
            return Ok(x);
        }
        //println!("Expanding {x} from {sh:?} to {shape:?}");
        if shape.len() < sh.len() {
            return Err(ZyxError::ShapeError(format!("Cannot expand {sh:?} into {shape:?}").into()));
        }
        let mut reshaped = false;
        let new_shape = if shape.len() > sh.len() {
            reshaped = true;
            std::iter::repeat_n(1, shape.len() - sh.len())
                .chain(sh.iter().copied())
                .collect()
        } else {
            sh
        };

        debug_assert_eq!(shape.len(), new_shape.len());
        #[cfg(debug_assertions)]
        for (&s, &d) in new_shape.iter().zip(shape.iter()) {
            if !(s == d || s == 1) {
                return Err(ZyxError::ShapeError(
                    format!("Cannot expand {new_shape:?} into {shape:?}").into(),
                ));
            }
        }

        if reshaped {
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
    pub(super) fn permute(&mut self, x: TensorId, axes: &[UAxis]) -> TensorId {
        if axes.len() < 2 || axes == (0..axes.len() as UAxis).collect::<Vec<UAxis>>() {
            self.retain(x);
            return x;
        }
        let shape = permute(self.shape(x), axes);
        let id = self.graph.push_wshape(Node::Permute { x }, shape);
        self.graph.push_axes(id, axes.to_vec());
        id
    }

    /// Expects padding in the same order as the shape, that is padding[0] pads shape[0]
    #[must_use]
    pub(super) fn pad_zeros(&mut self, x: TensorId, padding: Vec<(i64, i64)>) -> TensorId {
        let mut shape: Vec<Dim> = self.shape(x).into();
        debug_assert_eq!(shape.len(), padding.len());
        //println!("self shape: {shape:?}, padding: {padding:?}");
        apply_padding(&mut shape, &padding);
        //println!("out={shape:?}");
        let id = self.graph.push_wshape(Node::Pad { x }, shape);
        self.graph.push_padding(id, padding);
        id
    }

    #[must_use]
    pub(super) fn reduce(&mut self, x: TensorId, mut axes: Vec<UAxis>, rop: BOp) -> TensorId {
        let sh = self.shape(x);
        axes.sort_unstable();
        if axes.is_empty() {
            axes = (0..sh.len() as UAxis).collect();
        }
        let shape = reduce(sh, &axes);
        let id = self.graph.push_wshape(Node::Reduce { x, rop }, shape);
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
        if !self.buffer_map.contains_key(&x) {
            let mut to_eval = Set::with_capacity_and_hasher(1, BuildHasherDefault::default());
            to_eval.insert(x);
            self.realize_selected(&to_eval)?;
        }

        let buffer_id = get_mut_buffer(&self.buffer_map, x).unwrap();
        let byte_slice: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), data.len() * (T::bit_size() / 8) as usize) };
        for buffers in self.events.keys() {
            if buffers.contains(&buffer_id) {
                let buffers = buffers.clone();
                let event = self.events.remove(&buffers).unwrap();
                self.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, vec![event])?;
                return Ok(());
            }
        }
        self.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, Vec::new())?;
        Ok(())
    }

    pub fn realize_selected(&mut self, to_eval: &Set<TensorId>) -> Result<(), ZyxError> {
        //let time_w = std::time::Instant::now();
        let realized_nodes: Set<TensorId> = self.buffer_map.keys().copied().collect();

        let to_eval: Set<TensorId> = to_eval.difference(&realized_nodes).copied().collect();

        if to_eval.is_empty() {
            return Ok(());
        }

        if self.devices.is_empty() {
            self.initialize_devices()?;
        }

        let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
        let mut rcs: Map<TensorId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
        while let Some(nid) = params.pop() {
            rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                if !realized_nodes.contains(&nid) {
                    params.extend(self.graph.nodes[nid].1.parameters());
                }
                1
            });
        }
        //println!("elapsed sdfsdl {:?}", time_w.elapsed());
        // Order them using rcs reference counts
        let mut order = Vec::new();
        let mut internal_rcs: Map<TensorId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
        let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
        while let Some(nid) = params.pop() {
            if let Some(&rc) = rcs.get(&nid) {
                if rc == *internal_rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1) {
                    order.push(nid);
                    if !realized_nodes.contains(&nid) {
                        params.extend(self.graph.nodes[nid].1.parameters());
                    }
                }
            }
        }
        order.reverse();
        //println!("Order {order:?}");
        //println!("To eval {to_eval:?}");

        debug_assert!(!order.is_empty());
        debug_assert!(!to_eval.is_empty());

        if self.debug.perf() {
            println!(
                "Runtime realize graph order for {}/{} tensors with gradient_tape={}",
                order.len(),
                usize::from(self.graph.nodes.len()),
                self.graph.gradient_tape.is_some(),
            );
        }

        self.realize_with_order(rcs, realized_nodes, &order, &to_eval)?;
        //self.launch_or_store_graph_with_order(rcs, realized_nodes, &order, &to_eval)?;

        // Delete all unnecessary nodes no longer needed after realization
        let mut to_release = Vec::new();
        if let Some(tape) = self.graph.gradient_tape.as_ref() {
            for &nid in &to_eval {
                if !tape.contains(&nid) {
                    let dtype = self.dtype(nid);
                    let shape = self.shape(nid).into();
                    self.graph.shapes.insert(nid, shape);
                    to_release.extend(self.graph[nid].parameters());
                    self.graph.nodes[nid].1 = Node::Leaf { dtype };
                }
            }
        } else {
            for &nid in &to_eval {
                self.graph.add_shape(nid);
                let dtype = self.dtype(nid);
                to_release.extend(self.graph[nid].parameters());
                self.graph[nid] = Node::Leaf { dtype };
            }
        }
        let to_remove = self.graph.release(&to_release);
        deallocate_tensors(
            &to_remove,
            &mut self.pools,
            &mut self.events,
            &mut self.buffer_map,
            &mut self.temp_data,
        );

        #[cfg(debug_assertions)]
        {
            let realized_nodes: Set<TensorId> = self.buffer_map.keys().copied().collect();
            debug_assert!(realized_nodes.is_superset(&to_eval));
        }

        Ok(())
    }

    pub fn realize_all(&mut self) -> Result<(), ZyxError> {
        let realized_nodes: Set<TensorId> = self.buffer_map.keys().copied().collect();

        if self.devices.is_empty() {
            self.initialize_devices()?;
        }

        let mut rcs: Map<TensorId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
        for (_, node) in self.graph.nodes.values() {
            for nid in node.parameters() {
                rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
            }
        }

        let mut to_eval = Set::with_hasher(BuildHasherDefault::new()); // TODO
        for (id, (rc, _)) in self.graph.nodes.iter() {
            if let Some(graph_rc) = rcs.get(&id) {
                if rc > graph_rc {
                    to_eval.insert(id);
                }
            } else {
                to_eval.insert(id);
            }
        }
        for id in &to_eval {
            rcs.entry(*id).and_modify(|rc| *rc += 1).or_insert(1);
        }

        // Order them using rcs reference counts
        let mut order = Vec::new();
        let mut internal_rcs: Map<TensorId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
        let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
        while let Some(nid) = params.pop() {
            if let Some(&rc) = rcs.get(&nid) {
                if rc == *internal_rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1) {
                    order.push(nid);
                    if !realized_nodes.contains(&nid) {
                        params.extend(self.graph.nodes[nid].1.parameters());
                    }
                }
            }
        }
        order.reverse();
        //println!("Order {order:?}");
        //println!("To eval {to_eval:?}");

        debug_assert!(!order.is_empty());
        debug_assert!(!to_eval.is_empty());

        if self.debug.perf() {
            println!(
                "Runtime realize graph order for {}/{} tensors with gradient_tape={}",
                order.len(),
                usize::from(self.graph.nodes.len()),
                self.graph.gradient_tape.is_some(),
            );
        }

        self.realize_with_order(rcs, realized_nodes, &order, &to_eval)?;
        //self.launch_or_store_graph_with_order(rcs, realized_nodes, &order, &to_eval)?;

        // Delete all unnecessary nodes no longer needed after realization
        let mut to_release = Vec::new();
        if let Some(tape) = self.graph.gradient_tape.as_ref() {
            for &nid in &to_eval {
                if !tape.contains(&nid) {
                    let dtype = self.dtype(nid);
                    let shape = self.shape(nid).into();
                    self.graph.shapes.insert(nid, shape);
                    to_release.extend(self.graph[nid].parameters());
                    self.graph.nodes[nid].1 = Node::Leaf { dtype };
                }
            }
        } else {
            for &nid in &to_eval {
                self.graph.add_shape(nid);
                let dtype = self.dtype(nid);
                to_release.extend(self.graph[nid].parameters());
                self.graph[nid] = Node::Leaf { dtype };
            }
        }
        let to_remove = self.graph.release(&to_release);
        deallocate_tensors(
            &to_remove,
            &mut self.pools,
            &mut self.events,
            &mut self.buffer_map,
            &mut self.temp_data,
        );

        #[cfg(debug_assertions)]
        {
            let realized_nodes: Set<TensorId> = self.buffer_map.keys().copied().collect();
            debug_assert!(realized_nodes.is_superset(&to_eval));
        }

        Ok(())
    }
}

pub fn deallocate_tensors(
    to_remove: &Set<TensorId>,
    pools: &mut Slab<PoolId, MemoryPool>,
    events: &mut Map<BTreeSet<BufferId>, Event>,
    buffer_map: &mut Map<TensorId, BufferId>,
    temp_data: &mut Map<BufferId, Box<[u8]>>,
) {
    for tensor_id in to_remove {
        if let Some(buffer_id) = buffer_map.remove(tensor_id) {
            if !buffer_map.values().any(|&bid| bid == buffer_id) {
                let mut event_wait = Vec::new();
                if let Some(key) = events.keys().find(|key| key.contains(&buffer_id)).cloned() {
                    let event = events.remove(&key).unwrap();
                    event_wait.push(event);
                }
                pools[buffer_id.pool].deallocate(buffer_id.buffer, event_wait);
                temp_data.remove(&buffer_id);
            }
        }
    }
}

pub fn apply_padding(shape: &mut [Dim], padding: &[(i64, i64)]) {
    let mut i = 0;
    for d in shape.iter_mut() {
        *d = Dim::try_from(i64::try_from(*d).unwrap() + padding[i].0 + padding[i].1).unwrap();
        i += 1;
        if i >= padding.len() {
            break;
        }
    }
}
