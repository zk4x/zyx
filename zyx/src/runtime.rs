//! Runtime handles tensor graph and connects tensors to device buffers.
use crate::backend::{BackendError, Device, DeviceConfig, Event, MemoryPool};
use crate::dtype::{Constant, DType};
use crate::graph::Graph;
use crate::node::{BOp, Node, ROp, UOp};
use crate::kernel_cache::KernelCache;
use crate::rng::Rng;
use crate::scalar::Scalar;
use crate::shape::{permute, reduce, Axis, Dimension};
use crate::slab::Id;
use crate::static_graph::GraphOp;
use crate::tensor::TensorId;
use crate::{DebugMask, Map, Set};
use nanoserde::DeJson;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::{vec, vec::Vec};

const NUM_CONSTANTS: usize = 32;

// This is the whole global state of zyx
pub struct Runtime {
    // Current graph of tensor operations as nodes
    pub(super) graph: Graph,
    // Physical memory pools
    pools: Vec<Pool>,
    // Physical compute devices, each has their own program cache
    devices: Vec<Device>,
    // Optimizer cache, maps between unoptimized kernels and available/done optimizations
    optimizer: KernelCache,
    // Zyx configuration directory path
    config_dir: Option<PathBuf>, // Why the hell isn't PathBuf::new const?????
    // Random number generator
    pub(super) rng: Rng,
    // Are we in training mode?
    pub(super) training: bool,
    /// How many variations of one kernel to try during optimization
    pub(super) search_iterations: usize,
    /// Debug mask
    pub(super) debug: DebugMask,
    /// Temporary storage, TODO limit the number of elements in temporary storage
    temp_data: Vec<Box<dyn TempData>>,
    constants: [Constant; NUM_CONSTANTS],
    constants_len: usize,
}

pub trait TempData: Send {
    fn read(&self) -> &[u8];
    fn bytes(&self) -> Dimension;
    fn dtype(&self) -> DType;
}

pub struct Pool {
    #[allow(clippy::struct_field_names)]
    pub pool: MemoryPool,
    pub events: Map<BTreeSet<Id>, Event>,
    // tensor id => buffer id
    pub buffer_map: Map<TensorId, Id>,
}

impl Pool {
    pub(crate) fn new(pool: MemoryPool) -> Self {
        Self {
            pool,
            events: Map::with_capacity_and_hasher(100, Default::default()),
            buffer_map: Map::with_capacity_and_hasher(100, Default::default()),
        }
    }
}

fn get_mut_buffer(pools: &mut [Pool], tensor_id: TensorId) -> Option<(&mut Pool, Id)> {
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
            optimizer: KernelCache::new(),
            training: false,
            search_iterations: 0,
            debug: DebugMask(0),
            temp_data: Vec::new(),
            constants: [Constant::I32(0); NUM_CONSTANTS],
            constants_len: 0,
        }
    }

    pub(super) fn retain(&mut self, x: TensorId) {
        self.graph.retain(x);
    }

    pub(super) fn release(&mut self, x: TensorId) -> Result<(), ZyxError> {
        let to_remove = self.graph.release(x);
        self.deallocate_tensors(&to_remove)?;
        if self.graph.is_empty() && self.pools.iter().all(|mp| mp.buffer_map.is_empty()) {
            self.deinitialize()?;
        }
        Ok(())
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
        if let Ok(x) = std::env::var("ZYX_DEBUG") {
            if let Ok(x) = x.parse::<u32>() {
                self.debug = DebugMask(x);
            }
        }

        // ZYX_SEARCH is number of variations of one kernel that will be tried
        // during each run of the program. Timings are cached to disk,
        // so rerunning the same kernels will continue the search where it left of.
        if let Ok(x) = std::env::var("ZYX_SEARCH") {
            if let Ok(x) = x.parse() {
                self.search_iterations = x;
            }
        }

        // Search through config directories and find zyx/backend_config.json
        // If not found or failed to parse, use defaults.
        let device_config = xdg::BaseDirectories::new()
            .map_err(|e| {
                if self.debug.dev() {
                    println!("Failed to find config directories for device_config.json, {e}");
                }
            })
            .ok()
            .map(|bd| {
                let mut dirs = bd.get_config_dirs();
                dirs.push(bd.get_config_home());
                dirs
            })
            .and_then(|paths| {
                paths.into_iter().find_map(|mut path| {
                    path.push("zyx/device_config.json");
                    if let Ok(file) = std::fs::read_to_string(&path) {
                        path.pop();
                        self.config_dir = Some(path);
                        Some(file)
                    } else {
                        None
                    }
                })
            })
            .and_then(|file| {
                DeJson::deserialize_json(&file)
                    .map_err(|e| {
                        if self.debug.dev() {
                            println!("Failed to parse device_config.json, {e}");
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
                DeviceConfig::default()
            });

        // Load optimizer cache from disk if it exists
        #[cfg(feature = "disk_cache")]
        {
            if let Some(mut path) = self.config_dir.clone() {
                path.push("cached_kernels");
                if let Ok(mut file) = std::fs::File::open(path) {
                    use std::io::Read;
                    let mut buf = Vec::new();
                    file.read_to_end(&mut buf).unwrap();
                    if let Ok(optimizer_cache) = bitcode::decode(&buf) {
                        self.optimizer_cache = optimizer_cache;
                    }
                }
            }
        }
        crate::backend::initialize_backends(
            &device_config,
            &mut self.pools,
            &mut self.devices,
            self.debug.dev(),
        )?;
        self.pools.shrink_to_fit();
        self.devices.shrink_to_fit();
        Ok(())
    }

    /// This function deinitializes the whole runtime, deallocates all allocated memory and deallocates all caches
    /// It does not reset the rng and it does not change debug, search, training and `config_dir` fields
    fn deinitialize(&mut self) -> Result<(), ZyxError> {
        //println!("Deinitialize");
        // drop graph
        self.graph = Graph::new();
        // Drop programs
        self.optimizer.deinitialize(&mut self.devices);
        // drop devices
        while let Some(mut dev) = self.devices.pop() {
            dev.deinitialize()?;
        }
        // drop memory pools
        while let Some(mp) = self.pools.pop() {
            let Pool { mut pool, events, .. } = mp;
            pool.release_events(events.into_iter().map(|(_, v)| v).collect())?;
            pool.deinitialize()?;
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
        Ok(())
    }

    pub(super) fn manual_seed(&mut self, seed: u64) {
        self.rng = Rng::seed_from_u64(seed);
    }

    /// Creates dot plot of graph between given tensors
    #[must_use]
    pub(super) fn plot_dot_graph(&self, tensors: &Set<TensorId>) -> String {
        //println!("Tensor storage {:?}", self.tensor_buffer_map);
        self.graph.plot_dot_graph(tensors, &self.pools)
    }

    #[must_use]
    pub(super) fn shape(&self, x: TensorId) -> &[Dimension] {
        self.graph.shape(x)
    }

    #[must_use]
    pub(super) fn dtype(&self, x: TensorId) -> DType {
        self.graph.dtype(x)
    }

    pub(super) fn variable(
        &mut self,
        shape: Vec<Dimension>,
        data: Box<dyn TempData>,
    ) -> Result<TensorId, ZyxError> {
        let bytes = data.bytes();
        let dtype = data.dtype();
        debug_assert_eq!(
            shape.iter().product::<Dimension>() * dtype.byte_size() as Dimension,
            bytes
        );
        if bytes == dtype.byte_size() as usize {
            let value = data.read();
            let value = Constant::from_bytes(value, dtype);
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
        // TODO rewrite this such that we try to allocate memory pools in fastest device
        // order and we use first one that does not fail.
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
        // Pick memory pool with fastest device
        let mut memory_pool_id = mem_pools[0];
        let mut max_compute = 0;
        for dev in &self.devices {
            if dev.compute() > max_compute && mem_pools.contains(&dev.memory_pool_id()) {
                max_compute = dev.compute();
                memory_pool_id = dev.memory_pool_id();
            }
        }
        let mpid = memory_pool_id as usize;
        let (buffer_id, event) = self.pools[mpid].pool.allocate(bytes)?;
        self.temp_data.push(data);
        let event = self.pools[mpid].pool.host_to_pool(
            self.temp_data.last().unwrap().read(),
            buffer_id,
            vec![event],
        )?;
        let id = self.graph.push_wshape(Node::Leaf { dtype }, shape);
        self.pools[mpid].buffer_map.insert(id, buffer_id);
        self.pools[mpid].events.insert(BTreeSet::from([buffer_id]), event);
        Ok(id)
    }

    #[must_use]
    pub(super) fn constant(&mut self, value: impl Scalar) -> TensorId {
        self.graph.push(Node::Const { value: Constant::new(value) })
    }

    // Initialization
    pub(super) fn full(&mut self, shape: Vec<Dimension>, value: impl Scalar) -> TensorId {
        let x = self.constant(value);
        let expanded = self.expand(x, shape).unwrap();
        self.release(x).unwrap();
        expanded
    }

    #[must_use]
    pub(super) fn ones(&mut self, shape: Vec<Dimension>, dtype: DType) -> TensorId {
        let x = self.graph.push(Node::Const { value: dtype.one_constant() });
        let expanded = self.expand(x, shape).unwrap();
        self.release(x).unwrap();
        expanded
    }

    #[must_use]
    pub(super) fn zeros(&mut self, shape: Vec<Dimension>, dtype: DType) -> TensorId {
        let x = self.graph.push(Node::Const { value: dtype.zero_constant() });
        let expanded = self.expand(x, shape).unwrap();
        self.release(x).unwrap();
        expanded
    }

    /// Bitcast self to other type, currently immediatelly realizes the tensor.
    /// The caller is responsible for ensuring that destination dtype is representable
    /// with bytes of source data.
    pub(super) unsafe fn bitcast(
        &mut self,
        x: TensorId,
        dtype: DType,
    ) -> Result<TensorId, ZyxError> {
        if dtype == self.dtype(x) {
            self.retain(x);
            return Ok(x);
        }
        let mut to_eval = Set::with_capacity_and_hasher(10, Default::default());
        to_eval.insert(x);
        self.realize(to_eval)?;
        let mut shape = self.shape(x).to_vec();
        // We create a new pointer in tensor_buffer_map to the same buffer
        // and create a new Leaf in graph
        //self.tensor_buffer_map.find();
        let cd = dtype.byte_size() as Dimension / self.dtype(x).byte_size() as Dimension;
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
    pub(super) fn reshape(&mut self, x: TensorId, shape: Vec<Dimension>) -> TensorId {
        //println!("reshaping to {shape:?}, {:?}", self.shape(x));
        let sh = self.shape(x);
        debug_assert_eq!(
            shape.iter().product::<Dimension>(),
            sh.iter().product::<Dimension>()
        );
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
    #[must_use]
    pub(super) fn expand(
        &mut self,
        x: TensorId,
        shape: Vec<Dimension>,
    ) -> Result<TensorId, ZyxError> {
        let sh: Vec<Dimension> = self.shape(x).into();
        if shape == sh {
            self.retain(x);
            return Ok(x);
        }
        //println!("Expanding {x} from {sh:?} to {shape:?}");
        if shape.len() < sh.len() {
            return Err(ZyxError::ShapeError(format!(
                "Cannot expand {sh:?} into {shape:?}"
            )));
        }
        let new_shape = if shape.len() > sh.len() {
            std::iter::repeat(1).take(shape.len() - sh.len()).chain(sh.iter().copied()).collect()
        } else {
            sh.clone()
        };
        debug_assert_eq!(shape.len(), new_shape.len());
        for (&s, &d) in new_shape.iter().zip(shape.iter()) {
            if !(s == d || s == 1) {
                return Err(ZyxError::ShapeError(format!(
                    "Cannot expand {sh:?} into {shape:?}"
                )));
            }
        }
        if new_shape != sh {
            let x = self.reshape(x, new_shape.clone());
            if shape == new_shape {
                self.retain(x);
                return Ok(x);
            }
            let y = self.graph.push_wshape(Node::Expand { x }, shape);
            self.release(x).unwrap();
            return Ok(y);
        }
        Ok(self.graph.push_wshape(Node::Expand { x }, shape))
    }

    #[must_use]
    pub(super) fn permute(&mut self, x: TensorId, axes: &[Axis]) -> TensorId {
        if axes.len() < 2 || axes == (0..axes.len()).collect::<Vec<usize>>() {
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
        let mut shape: Vec<Dimension> = self.shape(x).into();
        //println!("Self shape: {shape:?}, padding: {padding:?}");
        apply_padding(&mut shape, &padding);
        let id = self.graph.push_wshape(Node::Pad { x }, shape);
        self.graph.push_padding(id, padding);
        id
    }

    #[must_use]
    pub(super) fn sum_reduce(&mut self, x: TensorId, mut axes: Vec<usize>) -> TensorId {
        let sh = self.shape(x);
        if axes.is_empty() {
            axes = (0..sh.len()).collect();
        };
        let shape = reduce(sh, &axes);
        let id = self.graph.push_wshape(Node::Reduce { x, rop: ROp::Sum }, shape);
        self.graph.push_axes(id, axes);
        id
    }

    #[must_use]
    pub(super) fn max_reduce(&mut self, x: TensorId, mut axes: Vec<usize>) -> TensorId {
        let sh = self.shape(x);
        if axes.is_empty() {
            axes = (0..sh.len()).collect();
        };
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
}

pub fn apply_padding(shape: &mut [Dimension], padding: &[(isize, isize)]) {
    let mut i = 0;
    for d in shape.iter_mut().rev() {
        *d = Dimension::try_from(isize::try_from(*d).unwrap() + padding[i].0 + padding[i].1)
            .unwrap();
        i += 1;
        if i >= padding.len() {
            break;
        }
    }
}

impl Runtime {
    /// Loads data with beginning elements of the tensor x.
    /// If `data.len()` == `x.numel()`, then it loads the whole tensor.
    pub(super) fn load<T: Scalar>(&mut self, x: TensorId, data: &mut [T]) -> Result<(), ZyxError> {
        let n: Dimension = self.shape(x).iter().product();
        let dt = self.dtype(x);
        if dt != T::dtype() {
            return Err(ZyxError::DTypeError(format!(
                "loading dtype {}, but the data has dtype {dt}",
                T::dtype()
            )));
        }
        debug_assert!(data.len() <= n, "Return buffer is bigger than tensor");
        // Check if tensor is evaluated
        if !self.pools.iter().any(|pool| pool.buffer_map.contains_key(&x)) {
            let mut to_eval = Set::with_capacity_and_hasher(10, Default::default());
            to_eval.insert(x);
            self.realize(to_eval)?;
        }

        let byte_slice: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), data.len() * T::byte_size())
        };

        let (pool, buffer_id) = get_mut_buffer(&mut self.pools, x).unwrap();
        for buffers in pool.events.keys() {
            if buffers.contains(&x) {
                let event = pool.events.remove(&buffers.clone()).unwrap();
                //println!("Loading with event {event:?}");
                pool.pool.pool_to_host(buffer_id, byte_slice, vec![event])?;
                return Ok(());
            }
        }
        pool.pool.pool_to_host(buffer_id, byte_slice, Vec::new())?;
        Ok(())
    }

    pub(super) fn realize(&mut self, to_eval: Set<TensorId>) -> Result<(), ZyxError> {
        let begin = std::time::Instant::now();

        let realized_nodes: Set<TensorId> =
            self.pools.iter().map(|pool| pool.buffer_map.keys()).flatten().copied().collect();
        let mut to_eval: Set<TensorId> = to_eval.difference(&realized_nodes).copied().collect();
        if to_eval.is_empty() {
            return Ok(());
        }
        if self.devices.is_empty() {
            self.initialize_devices()?;
        }

        let (order, mut to_delete, new_leafs, rcs) = if self.graph.gradient_tape.is_some() {
            // Get order for evaluation using DFS with ref counting to resolve
            // nodes with more than one parent.
            let (outside_nodes, mut order) = {
                let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
                let mut rcs: Map<TensorId, u32> =
                    Map::with_capacity_and_hasher(100, Default::default());
                while let Some(nid) = params.pop() {
                    rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                        params.extend(self.graph.nodes[nid].1.parameters());
                        1
                    });
                }
                // Order them using rcs reference counts
                let mut order = Vec::new();
                let mut internal_rcs: Map<TensorId, u32> =
                    Map::with_capacity_and_hasher(100, Default::default());
                let mut outside_nodes = Set::with_capacity_and_hasher(100, Default::default());
                let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
                while let Some(nid) = params.pop() {
                    if let Some(&rc) = rcs.get(&nid) {
                        if rc == *internal_rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1) {
                            order.push(nid);
                            let node = &self.graph.nodes[nid];
                            params.extend(node.1.parameters());
                            if node.0 > rc {
                                outside_nodes.insert(nid);
                            }
                        }
                    }
                }
                outside_nodes.extend(to_eval.clone());
                order.reverse();
                (outside_nodes, order)
            };
            //println!("Outside nodes: {outside_nodes:?}");
            // Constant folding and deleting unused parts of graph
            let mut new_leafs = Set::with_capacity_and_hasher(100, Default::default());
            let mut to_delete = Set::with_capacity_and_hasher(100, Default::default());
            for &nid in &order {
                let node = &self.graph.nodes[nid].1;
                match node.num_parameters() {
                    0 => {
                        if !outside_nodes.contains(&nid) {
                            to_delete.insert(nid);
                        }
                    }
                    1 => {
                        let x = node.param1();
                        if to_delete.contains(&x) {
                            if outside_nodes.contains(&nid) {
                                to_eval.insert(nid);
                                new_leafs.insert(nid);
                            } else {
                                to_delete.insert(nid);
                            }
                        }
                    }
                    2 => {
                        let (x, y) = node.param2();
                        let xc = to_delete.contains(&x);
                        let yc = to_delete.contains(&y);
                        match (xc, yc) {
                            (true, true) => {
                                if outside_nodes.contains(&nid) {
                                    to_eval.insert(nid);
                                    new_leafs.insert(nid);
                                } else {
                                    to_delete.insert(nid);
                                }
                            }
                            (true, false) => {
                                to_eval.insert(nid);
                                new_leafs.insert(x);
                            }
                            (false, true) => {
                                to_eval.insert(nid);
                                new_leafs.insert(y);
                            }
                            (false, false) => {}
                        }
                    }
                    _ => unreachable!(),
                }
            }
            //println!("To eval: {to_eval:?}");
            //println!("New leafs: {new_leafs:?}");
            //println!("To delete: {to_delete:?}");
            let to_eval: Set<TensorId> = to_eval.difference(&realized_nodes).copied().collect();
            let mut rcs: Map<TensorId, u32> =
                Map::with_capacity_and_hasher(100, Default::default());
            let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
            while let Some(nid) = params.pop() {
                if let Some(rc) = rcs.get_mut(&nid) {
                    *rc += 1;
                } else {
                    rcs.insert(nid, 1);
                    if !realized_nodes.contains(&nid) {
                        params.extend(self.graph.nodes[nid].1.parameters());
                    }
                }
            }
            /*for x in &to_eval {
                *rcs.get_mut(x).unwrap() -= 1;
                if *rcs.get(x).unwrap() == 0 {
                    rcs.remove(x);
                }
            }*/
            order.retain(|x| rcs.contains_key(x));
            // Currently rcs with gradient tape cannot be used by scheduler, so we give it empty ids
            (order, to_delete, new_leafs, Map::with_hasher(Default::default()))
        } else {
            let old_to_eval = to_eval.clone();
            let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
            let mut rcs: Map<TensorId, u32> =
                Map::with_capacity_and_hasher(100, Default::default());
            while let Some(nid) = params.pop() {
                rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                    if !realized_nodes.contains(&nid) {
                        params.extend(self.graph.nodes[nid].1.parameters());
                    }
                    1
                });
            }
            // Order them using rcs reference counts
            let mut to_delete = Set::with_capacity_and_hasher(100, Default::default());
            let mut new_leafs = Set::with_capacity_and_hasher(10, Default::default());
            let mut order = Vec::new();
            let mut internal_rcs: Map<TensorId, u32> =
                Map::with_capacity_and_hasher(100, Default::default());
            let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
            while let Some(nid) = params.pop() {
                if let Some(&rc) = rcs.get(&nid) {
                    if rc == *internal_rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1) {
                        order.push(nid);
                        let node = &self.graph.nodes[nid];
                        if node.0 > rc {
                            new_leafs.insert(nid);
                            if !realized_nodes.contains(&nid) {
                                to_eval.insert(nid);
                            }
                        } else {
                            if !to_eval.contains(&nid) {
                                to_delete.insert(nid);
                            } else {
                                new_leafs.insert(nid);
                            }
                        }
                        if !realized_nodes.contains(&nid) {
                            params.extend(node.1.parameters());
                        }
                    }
                }
            }
            order.reverse();
            for x in &old_to_eval {
                *rcs.get_mut(x).unwrap() -= 1;
                if *rcs.get(x).unwrap() == 0 {
                    rcs.remove(x);
                }
            }
            //println!("Order {order:?}");
            //println!("ToEval {to_eval:?}");
            //println!("ToDelete {to_delete:?}");
            //println!("NewLeafs {new_leafs:?}");
            (order, to_delete, new_leafs, rcs)
        };
        let elapsed = begin.elapsed();
        if self.debug.perf() {
            println!(
                "Runtime realize graph order took {} us for {}/{} tensors with gradient_tape = {}",
                elapsed.as_micros(),
                order.len(),
                self.graph.nodes.len(),
                self.graph.gradient_tape.is_some(),
            );
        }

        crate::scheduler::schedule(
            &self.graph,
            &order,
            rcs,
            &to_eval,
            &mut self.devices,
            &mut self.pools,
            &mut self.optimizer,
            self.search_iterations,
            realized_nodes,
            self.debug,
        )?;

        // Deallocate them from devices, new_leafs can be deallocated too
        self.deallocate_tensors(&to_delete)?;

        // Remove evaluated part of graph unless needed for backpropagation
        for tensor in new_leafs {
            self.graph.add_shape(tensor);
            self.graph[tensor] = Node::Leaf { dtype: self.graph.dtype(tensor) };
            to_delete.remove(&tensor);
        }
        // Delete the node, but do not use release function, just remove it from graph.nodes
        self.graph.delete_tensors(&to_delete);

        Ok(())
    }

    pub(super) fn compile_graph(&mut self, _inputs: &Set<TensorId>, _outputs: &Set<TensorId>) -> Vec<GraphOp> {
        let ops = Vec::new();
        ops
    }

    fn deallocate_tensors(&mut self, to_remove: &Set<TensorId>) -> Result<(), ZyxError> {
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
            if let Some((pool_id, buffer_id)) = buffer {
                if !self
                    .pools
                    .iter()
                    .any(|pool| pool.buffer_map.values().any(|bid| *bid == buffer_id))
                {
                    let pool = &mut self.pools[pool_id];
                    if let Some(key) = pool.events.keys().find(|key| key.contains(&buffer_id)) {
                        let event = pool.events.remove(&key.clone()).unwrap();
                        pool.pool.deallocate(buffer_id, vec![event])?;
                    } else {
                        pool.pool.deallocate(buffer_id, Vec::new())?;
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn drop_gradient_tape(&mut self) {
        self.graph.gradient_tape_ref_count -= 1;
        if self.graph.gradient_tape_ref_count == 0 {
            self.graph.gradient_tape = None;
            // TODO delete all unneeded nodes
        }
    }

    #[allow(clippy::similar_names)]
    pub(super) fn backward(
        &mut self,
        x: TensorId,
        sources: &Set<TensorId>,
    ) -> Map<TensorId, TensorId> {
        fn insert_or_add_grad(
            r: &mut Runtime,
            grads: &mut Map<TensorId, TensorId>,
            x: TensorId,
            grad: TensorId,
        ) {
            match grads.entry(x) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(grad);
                }
                std::collections::hash_map::Entry::Occupied(e) => {
                    let (k, prev_grad) = e.remove_entry();
                    grads.insert(
                        k,
                        r.graph.push(Node::Binary { x: prev_grad, y: grad, bop: BOp::Add }),
                    );
                    // These can never fail as it just decreses ref count,
                    // there is no deallocation.
                    r.release(prev_grad).unwrap();
                    r.release(grad).unwrap();
                }
            }
        }

        // Does not allocate new tensors, only constant and op nodes
        let topo = self.graph.build_topo(x, sources);
        //println!("Topo: {topo:?}");

        let req_grad: Set<TensorId> = topo.iter().copied().chain(sources.iter().copied()).collect();
        // Node -> Grad
        let mut grads: Map<TensorId, TensorId> =
            Map::with_capacity_and_hasher(100, Default::default());

        // Initial gradient of ones
        grads.insert(x, self.ones(self.shape(x).into(), self.dtype(x)));
        //println!("{:?}", self.nodes.last().unwrap());

        // All releases that cannot fail use unwrap to catch incorrect refcounts immediatelly.
        // reverse gradient calculation
        for nid in topo {
            let grad = grads[&nid];
            match self.graph[nid] {
                Node::Const { .. } | Node::Leaf { .. } => {}
                Node::Binary { x, y, bop } => match bop {
                    BOp::Add => {
                        if req_grad.contains(&x) {
                            self.retain(grad);
                            insert_or_add_grad(self, &mut grads, x, grad);
                        }
                        if req_grad.contains(&y) {
                            self.retain(grad);
                            insert_or_add_grad(self, &mut grads, y, grad);
                        }
                    }
                    BOp::Sub => {
                        if req_grad.contains(&x) {
                            self.retain(grad);
                            insert_or_add_grad(self, &mut grads, x, grad);
                        }
                        if req_grad.contains(&y) {
                            let grad = self.unary(grad, UOp::Neg);
                            insert_or_add_grad(self, &mut grads, y, grad);
                        }
                    }
                    BOp::Mul => {
                        if req_grad.contains(&x) {
                            let grad = self.binary(y, grad, BOp::Mul);
                            insert_or_add_grad(self, &mut grads, x, grad);
                        }
                        if req_grad.contains(&y) {
                            let grad = self.binary(x, grad, BOp::Mul);
                            insert_or_add_grad(self, &mut grads, y, grad);
                        }
                    }
                    BOp::Mod => {
                        todo!("Mod backward")
                    }
                    BOp::Div => {
                        if req_grad.contains(&x) {
                            let x_grad = self.binary(grad, y, BOp::Div);
                            insert_or_add_grad(self, &mut grads, x, x_grad);
                        }
                        if req_grad.contains(&y) {
                            // -(grad*x/(y*y))
                            let grad_neg = self.unary(grad, UOp::Neg);
                            let x_mul = self.binary(grad_neg, x, BOp::Mul);
                            self.release(grad_neg).unwrap();
                            let y_squared = self.binary(y, y, BOp::Mul);
                            let y_grad = self.binary(x_mul, y_squared, BOp::Div);
                            self.release(y_squared).unwrap();
                            insert_or_add_grad(self, &mut grads, y, y_grad);
                        }
                    }
                    BOp::Pow => {
                        if req_grad.contains(&x) {
                            // grad * y * x.pow(y-1)
                            let ones = self.ones(self.shape(y).into(), self.dtype(y));
                            let y_1 = self.binary(y, ones, BOp::Sub);
                            self.release(ones).unwrap();
                            let pow_y_1 = self.binary(x, y_1, BOp::Pow);
                            self.release(y_1).unwrap();
                            let y_mul = self.binary(y, pow_y_1, BOp::Mul);
                            self.release(pow_y_1).unwrap();
                            let x_grad = self.binary(grad, y_mul, BOp::Mul);
                            self.release(y_mul).unwrap();
                            insert_or_add_grad(self, &mut grads, x, x_grad);
                        }
                        if req_grad.contains(&y) {
                            // grad * x.pow(y) * log2(x) * (1/E.log2)
                            let sh = self.shape(y).into();
                            let dtype = self.dtype(y);
                            let one_elog2 = self.graph.push(Node::Const {
                                value: Constant::new(1f64 / std::f64::consts::E.log2()).cast(dtype),
                            });
                            let one_elog2_ex = self.expand(one_elog2, sh).unwrap();
                            self.release(one_elog2).unwrap();
                            let log2 = self.unary(x, UOp::Log2);
                            let log2_one_elog2 = self.binary(log2, one_elog2_ex, BOp::Mul);
                            self.release(log2).unwrap();
                            self.release(one_elog2_ex).unwrap();
                            let xpowy_log2_one_elog2 = self.binary(nid, log2_one_elog2, BOp::Mul);
                            self.release(log2_one_elog2).unwrap();
                            let y_grad = self.binary(grad, xpowy_log2_one_elog2, BOp::Mul);
                            self.release(xpowy_log2_one_elog2).unwrap();
                            insert_or_add_grad(self, &mut grads, y, y_grad);
                        }
                    }
                    BOp::Cmplt => {
                        panic!("Cmplt is not a differentiable operation.");
                    }
                    BOp::Cmpgt => {
                        panic!("Cmpgt is not a differentiable operation.");
                    }
                    BOp::NotEq => {
                        panic!("NotEq is not a differentiable operation.");
                    }
                    BOp::Max => {
                        todo!("Max backward.");
                    }
                    BOp::Or => {
                        todo!("Or backward.");
                    }
                    BOp::And => {
                        todo!("And backward.");
                    }
                    BOp::BitAnd => {
                        todo!("BitAnd backward.");
                    }
                    BOp::BitOr => {
                        todo!("BitOr backward.");
                    }
                    BOp::BitXor => {
                        todo!("BitXor backward.");
                    }
                    BOp::BitShiftLeft => {
                        todo!("BitXor backward.");
                    }
                    BOp::BitShiftRight => {
                        todo!("BitXor backward.");
                    }
                },
                Node::Cast { x, .. } => {
                    let grad = self.cast(grad, self.dtype(x));
                    insert_or_add_grad(self, &mut grads, x, grad);
                }
                Node::Unary { x, uop } => match uop {
                    UOp::Reciprocal => {
                        // -1/(x*x)
                        let x_2_inv = self.binary(nid, nid, BOp::Mul);
                        let x_grad = self.unary(x_2_inv, UOp::Neg);
                        self.release(x_2_inv).unwrap();
                        insert_or_add_grad(self, &mut grads, x, x_grad);
                    }
                    UOp::ReLU => {
                        let zeros = self.zeros(self.shape(x).into(), self.dtype(x));
                        let zl = self.binary(zeros, x, BOp::Cmplt);
                        self.release(zeros).unwrap();
                        let zl_cast = self.cast(zl, self.dtype(x));
                        self.release(zl).unwrap();
                        let x_grad = self.binary(zl_cast, grad, BOp::Mul);
                        self.release(zl_cast).unwrap();
                        insert_or_add_grad(self, &mut grads, x, x_grad);
                    }
                    UOp::Exp2 => {
                        let temp = self.constant(std::f64::consts::E.log2());
                        let temp1 = self.expand(temp, self.shape(x).into()).unwrap();
                        self.release(temp).unwrap();
                        let temp2 = self.binary(nid, temp1, BOp::Mul);
                        self.release(temp1).unwrap();
                        let grad = self.binary(nid, temp2, BOp::Mul);
                        self.release(temp2).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Log2 => {
                        let temp = self.constant(std::f64::consts::E.log2());
                        let temp1 = self.expand(temp, self.shape(x).into()).unwrap();
                        self.release(temp).unwrap();
                        let temp2 = self.binary(x, temp1, BOp::Mul);
                        self.release(temp1).unwrap();
                        let grad = self.binary(grad, temp2, BOp::Div);
                        self.release(temp2).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Sin => {
                        let x_temp = self.unary(x, UOp::Cos);
                        let grad = self.binary(x_temp, grad, BOp::Mul);
                        self.release(x_temp).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Cos => {
                        let x_temp1 = self.unary(x, UOp::Sin);
                        let x_temp = self.unary(x_temp1, UOp::Neg);
                        self.release(x_temp1).unwrap();
                        let grad = self.binary(x_temp, grad, BOp::Mul);
                        self.release(x_temp).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Sqrt => {
                        // x_grad = grad/(2*sqrt(x))
                        let sqrt_x = self.unary(x, UOp::Sqrt);
                        let sqrtx_2 = self.binary(sqrt_x, sqrt_x, BOp::Add);
                        self.release(sqrt_x).unwrap();
                        let grad = self.binary(grad, sqrtx_2, BOp::Div);
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Neg => {
                        let grad = self.unary(grad, UOp::Neg);
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    /*UOp::Tanh => {
                        // 1 - tanh^2(x)
                        let tanh_x_2 = self.mul(nid, nid);
                        let ones = self.ones(self.shape(x).into(), self.dtype(x));
                        let grad = self.sub(ones, tanh_x_2);
                        self.release(ones).unwrap();
                        self.release(tanh_x_2).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }*/
                    UOp::Not => {
                        todo!("Not backward")
                        //self.x.e(TernaryOps.WHERE, grad_output, grad_output.const(0)) if self.needs_input_grad[1] else None,
                        //self.x.e(TernaryOps.WHERE, grad_output.const(0), grad_output) if self.needs_input_grad[2] else None
                    }
                },
                Node::Reshape { x, .. } => {
                    let grad = self.reshape(grad, self.shape(x).into());
                    insert_or_add_grad(self, &mut grads, x, grad);
                }
                Node::Expand { x } => {
                    let sh = self.graph.shape(nid);
                    let x_shape: Vec<Dimension> = self.shape(x).into();
                    debug_assert_eq!(sh.len(), x_shape.len());
                    let expand_axes: Vec<usize> = sh
                        .into_iter()
                        .zip(&x_shape)
                        .enumerate()
                        .filter_map(|(a, (&d, &e))| if d == e { None } else { Some(a) })
                        .collect();
                    //println!("x shape {:?}, nid shape {:?}, expand_axes: {:?}", x_shape, sh, expand_axes);
                    debug_assert!(!expand_axes.is_empty());
                    let temp = self.sum_reduce(grad, expand_axes);
                    let grad = self.reshape(temp, x_shape);
                    self.release(temp).unwrap();
                    insert_or_add_grad(self, &mut grads, x, grad);
                }
                Node::Permute { x } => {
                    let axes = self.graph.axes(nid);
                    let mut axes: Vec<(usize, usize)> = axes.iter().copied().enumerate().collect();
                    axes.sort_by_key(|(_, v)| *v);
                    let argsort_axes: Vec<usize> = axes.iter().map(|(k, _)| *k).collect();
                    let grad = self.permute(grad, &argsort_axes);
                    insert_or_add_grad(self, &mut grads, x, grad);
                }
                Node::Pad { x } => {
                    let padding = self.graph.padding(x);
                    let inv_padding = padding.iter().map(|(lp, rp)| (-lp, -rp)).collect();
                    let grad = self.pad_zeros(grad, inv_padding);
                    insert_or_add_grad(self, &mut grads, x, grad);
                }
                Node::Reduce { x, rop } => match rop {
                    ROp::Sum => {
                        let x_shape: Vec<Dimension> = self.shape(x).into();
                        let mut z_shape: Vec<Dimension> = self.shape(nid).into();
                        //println!("Reduce backward, z shape: {z_shape:?}, x shape: {x_shape:?}, reduce axes: {:?}", self.graph.axes(nid));
                        for &axis in self.graph.axes(nid) {
                            z_shape.insert(axis, 1);
                        }
                        if self.graph.axes(nid).len() == x_shape.len() {
                            z_shape.remove(0);
                        }
                        let temp = self.reshape(grad, z_shape);
                        let grad = self.expand(temp, x_shape).unwrap();
                        self.release(temp).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    ROp::Max => {
                        // x_grad = (1 - (x < z.expand(x.shape()))) * grad
                        let x_shape: Vec<Dimension> = self.shape(x).into();
                        let z_temp = self.expand(nid, x_shape.clone()).unwrap();
                        let cmp_t = self.binary(x, z_temp, BOp::Cmplt);
                        self.release(z_temp).unwrap();
                        let ones = self.zeros(x_shape, self.dtype(x));
                        let max_1s = self.binary(ones, cmp_t, BOp::Sub);
                        self.release(ones).unwrap();
                        self.release(cmp_t).unwrap();
                        let grad = self.binary(max_1s, grad, BOp::Mul);
                        self.release(max_1s).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                },
            }
        }
        //println!("gradients: {grads:?}");
        let mut res = Map::with_capacity_and_hasher(10, Default::default());
        for (k, v) in grads {
            if sources.contains(&k) {
                res.insert(k, v);
            } else {
                self.release(v).unwrap();
            }
        }
        //println!("res: {res:?}");
        res
    }
}

/// Enumeration representing the various errors that can occur within the Zyx library.
#[derive(Debug)]
pub enum ZyxError {
    /// Invalid shapes for operation
    ShapeError(String),
    /// Wrong dtype for given operation
    DTypeError(String),
    /// Backend configuration error
    BackendConfig(&'static str),
    /// Error from file operations
    IOError(std::io::Error),
    /// Error parsing some data
    ParseError(String),
    /// Memory allocation error
    AllocationError,
    /// There are no available backends
    NoBackendAvailable,
    /// Error returned by backends
    BackendError(BackendError),
}

impl std::fmt::Display for ZyxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZyxError::ShapeError(e) => f.write_str(e),
            ZyxError::DTypeError(e) => f.write_fmt(format_args!("Wrong dtype {e:?}")),
            ZyxError::IOError(e) => f.write_fmt(format_args!("IO {e}")),
            ZyxError::ParseError(e) => f.write_fmt(format_args!("IO {e}")),
            ZyxError::BackendConfig(e) => f.write_fmt(format_args!("Backend config {e:?}'")),
            ZyxError::NoBackendAvailable => f.write_fmt(format_args!("No available backend")),
            ZyxError::AllocationError => f.write_fmt(format_args!("Allocation error")),
            ZyxError::BackendError(e) => f.write_fmt(format_args!("Backend {e}")),
        }
    }
}

impl std::error::Error for ZyxError {}

impl From<std::io::Error> for ZyxError {
    fn from(value: std::io::Error) -> Self {
        Self::IOError(value)
    }
}
