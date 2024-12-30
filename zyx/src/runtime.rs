//! Runtime handles tensor graph and connects tensors to device buffers.
use crate::backend::{BackendError, Device, DeviceConfig, Event, MemoryPool};
use crate::dtype::{Constant, DType};
use crate::graph::Graph;
use crate::node::{BOp, Node, ROp, UOp};
use crate::optimizer::Optimizer;
use crate::rng::Rng;
use crate::scalar::Scalar;
use crate::shape::{permute, reduce, Dimension};
use crate::slab::Id;
use crate::tensor::TensorId;
use crate::DebugMask;
use std::path::PathBuf;
use std::{
    collections::{btree_map::Entry, BTreeMap, BTreeSet},
    vec,
    vec::Vec,
};

use half::{bf16, f16};
use nanoserde::DeJson;

// This is the whole global state of zyx
pub struct Runtime {
    // Current graph of tensor operations as nodes
    graph: Graph,
    // Physical memory pools
    pools: Vec<Pool>,
    // Physical compute devices, each has their own program cache
    devices: Vec<Box<dyn Device>>,
    // Optimizer cache, maps between unoptimized kernels and available/done optimizations
    optimizer: Optimizer,
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
    /// Temporary storage
    temp_data: Vec<Box<dyn TempData>>,
}

pub trait TempData: Send {
    fn read(&self) -> &[u8];
    fn bytes(&self) -> usize;
    fn dtype(&self) -> DType;
}

pub struct Pool {
    #[allow(clippy::struct_field_names)]
    pub pool: Box<dyn MemoryPool>,
    pub events: BTreeMap<BTreeSet<Id>, Event>,
    pub buffer_map: BTreeMap<TensorId, Id>,
}

fn get_mut_buffer(pools: &mut [Pool], tensor_id: TensorId) -> (&mut Pool, Id) {
    for pool in pools {
        if let Some(&id) = pool.buffer_map.get(&tensor_id) {
            return (pool, id);
        }
    }
    unreachable!()
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
            optimizer: Optimizer::new(),
            training: false,
            search_iterations: 0,
            debug: DebugMask(0),
            temp_data: Vec::new(),
        }
    }

    pub(super) fn retain(&mut self, x: TensorId) {
        self.graph.retain(x);
    }

    pub(super) fn release(&mut self, x: TensorId) -> Result<(), ZyxError> {
        let to_remove = self.graph.release(x);
        self.deallocate_tensors(&to_remove)?;
        // TODO Check the number of tensors. If there are no tensors remaining, deinitialize the runtime,
        // since rust does not implement drop for us.
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
        // drop devices
        while let Some(mut dev) = self.devices.pop() {
            dev.deinitialize()?;
        }
        // drop memory pools
        while let Some(mut mp) = self.pools.pop() {
            //for (_, event) in mp.events { event.sync()?; }
            mp.pool.deinitialize()?;
        }
        self.optimizer = Optimizer::new();
        // Timer
        /*for (name, time) in crate::ET.lock().iter() {
            println!("Timer {name} took {time} us");
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
    pub(super) fn plot_dot_graph(&self, tensors: &BTreeSet<TensorId>) -> String {
        //println!("Tensor storage {:?}", self.tensor_buffer_map);
        self.graph.plot_dot_graph(tensors)
    }

    #[must_use]
    pub(super) fn shape(&self, x: TensorId) -> &[usize] {
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
        debug_assert_eq!(shape.iter().product::<usize>() * dtype.byte_size(), bytes);
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
        let id = self.graph.push_wshape_and_dtype(Node::Leaf, shape, dtype);
        self.pools[mpid].buffer_map.insert(id, buffer_id);
        self.pools[mpid].events.insert(BTreeSet::from([buffer_id]), event);
        Ok(id)
    }

    #[must_use]
    pub(super) fn constant(&mut self, value: impl Scalar) -> TensorId {
        self.graph.push(Node::Const { value: Constant::new(value) })
    }

    // Initialization
    pub(super) fn full(
        &mut self,
        shape: Vec<usize>,
        value: impl Scalar,
    ) -> Result<TensorId, ZyxError> {
        let x = self.constant(value);
        let expanded = self.expand(x, shape);
        self.release(x)?;
        Ok(expanded)
    }

    #[must_use]
    pub(super) fn ones(&mut self, shape: Vec<usize>, dtype: DType) -> TensorId {
        let x = match dtype {
            DType::BF16 => self.constant(bf16::ONE),
            DType::F8 => todo!(),
            DType::F16 => self.constant(f16::ONE),
            DType::F32 => self.constant(1f32),
            DType::F64 => self.constant(1f64),
            DType::U8 => self.constant(1u8),
            DType::U16 => self.constant(1u16),
            DType::U32 => self.constant(1u32),
            DType::U64 => self.constant(1u64),
            DType::I8 => self.constant(1i8),
            DType::I16 => self.constant(1i16),
            DType::I32 => self.constant(1i32),
            DType::I64 => self.constant(1i64),
            DType::Bool => self.constant(true),
        };
        let expanded = self.expand(x, shape);
        self.release(x).unwrap();
        expanded
    }

    #[must_use]
    pub(super) fn zeros(&mut self, shape: Vec<usize>, dtype: DType) -> TensorId {
        let x = match dtype {
            DType::BF16 => self.constant(bf16::ZERO),
            DType::F8 => todo!(),
            DType::F16 => self.constant(f16::ZERO),
            DType::F32 => self.constant(0f32),
            DType::F64 => self.constant(0f64),
            DType::U8 => self.constant(0u8),
            DType::U16 => self.constant(0u16),
            DType::U32 => self.constant(0u32),
            DType::U64 => self.constant(0u64),
            DType::I8 => self.constant(0i8),
            DType::I16 => self.constant(0i16),
            DType::I32 => self.constant(0i32),
            DType::I64 => self.constant(0i64),
            DType::Bool => self.constant(false),
        };
        let expanded = self.expand(x, shape);
        self.release(x).unwrap();
        expanded
    }

    // Unary ops
    #[must_use]
    pub(super) fn cast(&mut self, x: TensorId, dtype: DType) -> TensorId {
        if dtype == self.dtype(x) {
            self.retain(x);
            return x;
        }
        self.graph.push_wdtype(Node::Unary { x, uop: UOp::Cast(dtype) }, dtype)
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
        self.realize(BTreeSet::from([x]))?;
        let mut shape = self.shape(x).to_vec();
        // We create a new pointer in tensor_buffer_map to the same buffer
        // and create a new Leaf in graph
        //self.tensor_buffer_map.find();
        let cd = dtype.byte_size() / self.dtype(x).byte_size();
        if let Some(d) = shape.last_mut() {
            if *d % cd != 0 {
                return Err(ZyxError::DTypeError(
                    "Can't bitcast due to tensor's last dimension not being correct multiple of dtype.".into(),
                ));
            }
            *d /= cd;
        }
        let id = self.graph.push_wshape_and_dtype(Node::Leaf, shape.clone(), dtype);
        if let Some(pool) = self.pools.iter_mut().find(|pool| pool.buffer_map.contains_key(&x)) {
            //println!("Bitcast {x}, res {id}, new shape {shape:?} buffer id {bid:?}");
            let x = *pool.buffer_map.get(&x).unwrap();
            pool.buffer_map.insert(id, x);
        }
        //println!("TBM:\n{:?}", self.tensor_buffer_map);
        Ok(id)
    }

    #[must_use]
    pub(super) fn reciprocal(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Inv })
    }

    #[must_use]
    pub(super) fn neg(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Neg })
    }

    #[must_use]
    pub(super) fn relu(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::ReLU })
    }

    #[must_use]
    pub(super) fn exp2(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Exp2 })
    }

    #[must_use]
    pub(super) fn log2(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Log2 })
    }

    #[must_use]
    pub(super) fn inv(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Inv })
    }

    #[must_use]
    pub(super) fn sin(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Sin })
    }

    #[must_use]
    pub(super) fn cos(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Cos })
    }

    #[must_use]
    pub(super) fn sqrt(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Sqrt })
    }

    #[must_use]
    pub(super) fn not(&mut self, x: TensorId) -> TensorId {
        self.graph.push(Node::Unary { x, uop: UOp::Not })
    }

    #[must_use]
    pub(super) fn reshape(&mut self, x: TensorId, shape: Vec<usize>) -> TensorId {
        //println!("reshaping to {shape:?}, {:?}", self.shape(x));
        let sh = self.shape(x);
        assert_eq!(
            shape.iter().product::<usize>(),
            sh.iter().product::<usize>()
        );
        if shape == sh {
            self.retain(x);
            return x;
        }
        // Reshape on leaf is NOOP, tensor_buffer_map traces ownership
        if self.graph[x] == Node::Leaf {
            let (pool, bid) = get_mut_buffer(&mut self.pools, x);
            let id = self.graph.push_wshape_and_dtype(Node::Leaf, shape, self.graph.dtype(x));
            pool.buffer_map.insert(id, bid);
            return id;
        }
        self.graph.push_wshape(Node::Reshape { x }, shape)
    }

    #[must_use]
    pub(super) fn expand(&mut self, x: TensorId, shape: Vec<usize>) -> TensorId {
        let sh = self.shape(x);
        //println!("Expanding {x} from {sh:?} to {shape:?}");
        if shape == sh {
            self.retain(x);
            return x;
        }
        // Expand with only inserting first dimensions is noop
        if self.graph[x] == Node::Leaf
            && sh.iter().product::<usize>() == shape.iter().product::<usize>()
        {
            let (pool, bid) = get_mut_buffer(&mut self.pools, x);
            let id = self.graph.push_wshape_and_dtype(Node::Leaf, shape, self.graph.dtype(x));
            pool.buffer_map.insert(id, bid);
            return id;
        }
        if shape.len() > sh.len() {
            let sh: Vec<usize> = std::iter::repeat(1)
                .take(shape.len() - sh.len())
                .chain(sh.iter().copied())
                .collect();
            assert_eq!(shape.len(), sh.len());
            let y = self.reshape(x, sh);
            let x = self.graph.push_wshape(Node::Expand { x: y }, shape);
            self.release(y).unwrap();
            return x;
        }
        assert_eq!(shape.len(), sh.len());
        self.graph.push_wshape(Node::Expand { x }, shape)
    }

    #[must_use]
    pub(super) fn permute(&mut self, x: TensorId, axes: &[usize]) -> TensorId {
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
        let mut shape: Vec<usize> = self.shape(x).into();
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

    #[must_use]
    pub(super) fn add(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary { x, y, bop: BOp::Add })
    }

    #[must_use]
    pub(super) fn sub(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary { x, y, bop: BOp::Sub })
    }

    #[must_use]
    pub(super) fn mul(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary { x, y, bop: BOp::Mul })
    }

    #[must_use]
    pub(super) fn div(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary { x, y, bop: BOp::Div })
    }

    #[must_use]
    pub(super) fn and(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary { x, y, bop: BOp::And })
    }

    #[must_use]
    pub(super) fn or(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary { x, y, bop: BOp::Or })
    }

    #[must_use]
    pub(super) fn bitor(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary { x, y, bop: BOp::BitOr })
    }

    #[must_use]
    pub(super) fn bitxor(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary { x, y, bop: BOp::BitXor })
    }

    #[must_use]
    pub(super) fn bitand(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary { x, y, bop: BOp::BitAnd })
    }

    #[must_use]
    pub(super) fn pow(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary { x, y, bop: BOp::Pow })
    }

    #[must_use]
    pub(super) fn cmplt(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary { x, y, bop: BOp::Cmplt })
    }

    #[must_use]
    pub(super) fn cmpgt(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary { x, y, bop: BOp::Cmpgt })
    }

    #[must_use]
    pub(super) fn not_eq(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary { x, y, bop: BOp::NotEq })
    }

    #[must_use]
    pub(super) fn maximum(&mut self, x: TensorId, y: TensorId) -> TensorId {
        self.graph.push(Node::Binary { x, y, bop: BOp::Max })
    }
}

pub fn apply_padding(shape: &mut [usize], padding: &[(isize, isize)]) {
    let mut i = 0;
    for d in shape.iter_mut().rev() {
        *d = usize::try_from(isize::try_from(*d).unwrap() + padding[i].0 + padding[i].1).unwrap();
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
        let n: usize = self.shape(x).iter().product();
        let dt = self.dtype(x);
        if dt != T::dtype() {
            return Err(ZyxError::DTypeError(format!(
                "loading dtype {}, but the data has dtype {dt}",
                T::dtype()
            )));
        }
        assert!(data.len() <= n, "Return buffer is bigger than tensor");
        // Check if tensor is evaluated
        if !self.pools.iter().any(|pool| pool.buffer_map.contains_key(&x)) {
            self.realize(BTreeSet::from([x]))?;
        }

        let byte_slice: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), data.len() * T::byte_size())
        };

        let (pool, buffer_id) = get_mut_buffer(&mut self.pools, x);
        for buffers in pool.events.keys() {
            if buffers.contains(&x) {
                let event = pool.events.remove(&buffers.clone()).unwrap();
                pool.pool.pool_to_host(buffer_id, byte_slice, vec![event])?;
                return Ok(());
            }
        }
        pool.pool.pool_to_host(buffer_id, byte_slice, Vec::new())?;
        Ok(())
    }

    pub(super) fn realize(&mut self, mut to_eval: BTreeSet<TensorId>) -> Result<(), ZyxError> {
        //let timer = backend::Timer::new();
        // Runs in O(4n) where n = self.graph.len(),
        // first pass for visited nodes, second pass for outisde_rcs, third pass for order,
        // fourth pass for to_delete and new_leafs
        // Could possibly be optimized a bit
        if to_eval.is_empty() {
            return Ok(());
        }
        if to_eval.iter().all(|id| self.pools.iter().any(|pool| pool.buffer_map.contains_key(id))) {
            return Ok(());
        }
        if self.devices.is_empty() {
            self.initialize_devices()?;
        }
        //let t = crate::Timer::new("realize create graph");
        // Outside nodes are nodes that exist in realization graph, but also are needed by other parts of the graph or by user.
        // That is outside nodes have reference counts greater than their reference counts in realization graph.
        let (outside_nodes, order) = {
            let this = &self.graph;
            let to_eval: &BTreeSet<TensorId> = &to_eval;
            // Following loops visit all children nodes up to leafs,
            // because we need to know which parts of the graph can be dropped.
            // Get refcounts of all nodes
            let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
            let mut rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
            while let Some(nid) = params.pop() {
                rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                    params.extend(this.nodes[nid].1.parameters());
                    1
                });
            }
            // Order them using rcs reference counts
            let mut order = Vec::new();
            let mut internal_rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
            let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
            while let Some(nid) = params.pop() {
                if let Some(&rc) = rcs.get(&nid) {
                    if rc == *internal_rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1) {
                        order.push(nid);
                        params.extend(this.nodes[nid].1.parameters());
                    }
                }
            }
            order.reverse();
            // TODO can we get rid of this third pass?
            let outside_nodes: BTreeSet<TensorId> = self
                .graph
                .nodes
                .iter()
                .filter_map(|(id, (rc, _))| {
                    if let Some(&rc2) = rcs.get(&id) {
                        if *rc > rc2 {
                            Some(id)
                        } else {
                            None
                        }
                    } else {
                        Some(id)
                    }
                })
                .chain(to_eval.iter().copied())
                .collect();
            (outside_nodes, order)
        };
        //, |x| realized_tensors.contains(&x));
        // Which parts of graph are no longer needed and can be deleted and which nodes will be new leafs?
        // New leafs never store data, so we can deallocate them if they are allocated.
        let mut to_delete = BTreeSet::new();
        let mut new_leafs = BTreeSet::new();
        //println!("Graph: {:?}", self.graph);
        //println!("Outside nodes: {outside_nodes:?}");
        //println!("Order: {order:?}");
        // Calculates which tensors are not needed and which tensors need to be evaluated
        // in order to drop those unneeded tensors. This is basically constant folding.
        for tensor in &order {
            if matches!(self.graph[*tensor], Node::Leaf | Node::Const { .. }) {
                if !outside_nodes.contains(tensor) {
                    to_delete.insert(*tensor);
                    continue;
                }
            } else if self.graph[*tensor].parameters().all(|tensor| to_delete.contains(&tensor)) {
                if outside_nodes.contains(tensor) {
                    to_eval.insert(*tensor);
                    new_leafs.insert(*tensor);
                } else {
                    to_delete.insert(*tensor);
                }
            } else {
                for param in self.graph[*tensor].parameters() {
                    if to_delete.contains(&param) {
                        new_leafs.insert(param);
                    }
                }
            }
        }
        //println!("New leafs: {new_leafs:?}");
        //println!("Realizing {:?}", to_eval);

        crate::scheduler::realize_graph(
            &self.graph,
            &order,
            &to_eval,
            &mut self.devices,
            &mut self.pools,
            &mut self.optimizer,
            self.search_iterations,
            self.debug,
        )?;

        // Deallocate them from devices
        self.deallocate_tensors(&to_delete)?;
        // Remove evaluated part of graph unless needed for backpropagation
        for tensor in new_leafs {
            self.graph.add_shape_dtype(tensor);
            self.graph[tensor] = Node::Leaf;
            to_delete.remove(&tensor);
        }
        //println!("To delete: {to_delete:?}");
        // Delete the node, but do not use release function, just remove it from graph.nodes
        self.graph.delete_tensors(&to_delete);
        Ok(())
    }

    fn deallocate_tensors(&mut self, to_remove: &BTreeSet<TensorId>) -> Result<(), ZyxError> {
        // This is basically tracing GC, seems faster than reference counting
        // remove all buffers that are not used by any tensors
        // Check which buffers will possibly need to be dropped
        for pool in &mut self.pools {
            for tensor_id in to_remove {
                if let Some(buffer_id) = pool.buffer_map.remove(tensor_id) {
                    if !pool.buffer_map.values().any(|bid| *bid == buffer_id) {
                        if let Some(key) = pool.events.keys().find(|key| key.contains(&buffer_id)) {
                            let event = pool.events.remove(&key.clone()).unwrap();
                            pool.pool.deallocate(buffer_id, vec![event])?;
                        } else {
                            pool.pool.deallocate(buffer_id, Vec::new())?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::similar_names)]
    pub(super) fn backward(
        &mut self,
        x: TensorId,
        sources: &BTreeSet<TensorId>,
    ) -> BTreeMap<TensorId, TensorId> {
        fn insert_or_add_grad(
            r: &mut Runtime,
            grads: &mut BTreeMap<TensorId, TensorId>,
            x: TensorId,
            grad: TensorId,
        ) {
            match grads.entry(x) {
                Entry::Vacant(e) => {
                    e.insert(grad);
                }
                Entry::Occupied(e) => {
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

        let req_grad: BTreeSet<TensorId> =
            topo.iter().copied().chain(sources.iter().copied()).collect();
        // Node -> Grad
        let mut grads: BTreeMap<TensorId, TensorId> = BTreeMap::new();
        // Initial gradient of ones
        let grad1 = self.ones(vec![1], self.dtype(x));
        let sh: Vec<Dimension> = self.shape(x).into();
        grads.insert(x, self.graph.push_wshape(Node::Expand { x: grad1 }, sh));
        self.release(grad1).unwrap();
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
                            let grad = self.neg(grad);
                            insert_or_add_grad(self, &mut grads, y, grad);
                        }
                    }
                    BOp::Mul => {
                        if req_grad.contains(&x) {
                            let grad = self.mul(y, grad);
                            insert_or_add_grad(self, &mut grads, x, grad);
                        }
                        if req_grad.contains(&y) {
                            let grad = self.mul(x, grad);
                            insert_or_add_grad(self, &mut grads, y, grad);
                        }
                    }
                    BOp::Mod => {
                        todo!("Mod backward")
                    }
                    BOp::Div => {
                        if req_grad.contains(&x) {
                            grads.insert(x, self.div(grad, y));
                            insert_or_add_grad(self, &mut grads, x, grad);
                        }
                        if req_grad.contains(&y) {
                            // -grad*x/(y^2)
                            let dtype = self.dtype(y);
                            let two_temp = self.ones(vec![1], dtype);
                            let two = self.add(two_temp, two_temp);
                            self.release(two_temp).unwrap();
                            let two_e = self.expand(two, self.shape(y).into());
                            self.release(two).unwrap();
                            let two_2 = self.pow(y, two_e);
                            self.release(two_e).unwrap();
                            let temp = self.mul(x, grad);
                            let temp_neg = self.neg(temp);
                            self.release(temp).unwrap();
                            let y_grad = self.div(temp_neg, two_2);
                            self.release(temp_neg).unwrap();
                            self.release(two_2).unwrap();
                            grads.insert(y, y_grad);
                            insert_or_add_grad(self, &mut grads, y, grad);
                        }
                    }
                    BOp::Pow => {
                        if req_grad.contains(&x) {
                            // grad * y * x.pow(y-1)
                            let ones = self.ones(self.shape(y).into(), self.dtype(y));
                            let y_1 = self.sub(y, ones);
                            self.release(ones).unwrap();
                            let pow_y_1 = self.pow(x, y_1);
                            self.release(y_1).unwrap();
                            let y_mul = self.mul(y, pow_y_1);
                            self.release(pow_y_1).unwrap();
                            let x_grad = self.mul(grad, y_mul);
                            self.release(y_mul).unwrap();
                            insert_or_add_grad(self, &mut grads, x, x_grad);
                        }
                        if req_grad.contains(&y) {
                            // grad * x.pow(y) * ln(x)
                            let temp1 = self.log2(x);
                            let temp2 = self.mul(nid, temp1);
                            self.release(temp1).unwrap();
                            let y_grad = self.mul(grad, temp2);
                            self.release(temp2).unwrap();
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
                Node::Unary { x, uop } => match uop {
                    UOp::Inv => {
                        // -1/(x*x)
                        let x_2_inv = self.mul(nid, nid);
                        let x_grad = self.neg(x_2_inv);
                        self.release(x_2_inv).unwrap();
                        insert_or_add_grad(self, &mut grads, x, x_grad);
                    }
                    UOp::ReLU => {
                        let zeros = self.zeros(self.shape(x).into(), self.dtype(x));
                        let zl = self.cmplt(zeros, x);
                        self.release(zeros).unwrap();
                        let x_grad = self.mul(zl, grad);
                        self.release(zl).unwrap();
                        insert_or_add_grad(self, &mut grads, x, x_grad);
                    }
                    UOp::Exp2 => {
                        let temp = self.constant(std::f64::consts::E.log2());
                        let temp1 = self.expand(temp, self.shape(x).into());
                        self.release(temp).unwrap();
                        let temp2 = self.mul(nid, temp1);
                        self.release(temp1).unwrap();
                        let grad = self.mul(nid, temp2);
                        self.release(temp2).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Log2 => {
                        let temp = self.constant(std::f64::consts::E.log2());
                        let temp1 = self.expand(temp, self.shape(x).into());
                        self.release(temp).unwrap();
                        let temp2 = self.mul(x, temp1);
                        self.release(temp1).unwrap();
                        let grad = self.div(grad, temp2);
                        self.release(temp2).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Sin => {
                        let x_temp = self.cos(x);
                        let grad = self.mul(x_temp, grad);
                        self.release(x_temp).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Cos => {
                        let x_temp1 = self.sin(x);
                        let x_temp = self.neg(x_temp1);
                        self.release(x_temp1).unwrap();
                        let grad = self.mul(x_temp, grad);
                        self.release(x_temp).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Sqrt => {
                        // x_grad = grad/(2*sqrt(x))
                        let sqrt_x = self.sqrt(x);
                        let sqrtx_2 = self.add(sqrt_x, sqrt_x);
                        self.release(sqrt_x).unwrap();
                        let grad = self.div(grad, sqrtx_2);
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Cast(_) => {
                        let grad = self.cast(grad, self.dtype(x));
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Neg => {
                        let grad = self.neg(grad);
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
                    let shape = self.graph.shape(nid);
                    let mut vec: Vec<Dimension> = shape.into();
                    while vec.len() < shape.len() {
                        vec.insert(0, 1);
                    }
                    let expand_axes: Vec<usize> = vec
                        .into_iter()
                        .zip(shape)
                        .enumerate()
                        .filter_map(|(a, (d, e))| if d == *e { None } else { Some(a) })
                        .collect();
                    let temp = self.sum_reduce(grad, expand_axes);
                    let grad = self.reshape(temp, self.shape(x).into());
                    self.release(temp).unwrap();
                    insert_or_add_grad(self, &mut grads, x, grad);
                }
                Node::Permute { x } => {
                    let axes = self.graph.axes(x);
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
                Node::Reduce { x, rop, .. } => match rop {
                    ROp::Sum => {
                        let grad = self.expand(grad, self.shape(x).into());
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    ROp::Max => {
                        // x_grad = (1 - (x < z.expand(x.shape()))) * grad
                        let x_shape: Vec<usize> = self.shape(x).into();
                        let z_temp = self.expand(nid, x_shape.clone());
                        let cmp_t = self.cmplt(x, z_temp);
                        self.release(z_temp).unwrap();
                        let ones = self.zeros(x_shape, self.dtype(x));
                        let max_1s = self.sub(ones, cmp_t);
                        self.release(ones).unwrap();
                        self.release(cmp_t).unwrap();
                        let grad = self.mul(max_1s, grad);
                        self.release(max_1s).unwrap();
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                },
            }
        }
        let mut res = BTreeMap::new();
        for (k, v) in grads {
            if sources.contains(&k) {
                res.insert(k, v);
            } else {
                self.release(v).unwrap();
            }
        }
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

/*impl<Err: std::fmt::Display> From<Err> for ZyxError {
    #[track_caller]
    fn from(err: Err) -> Self {
        panic!("error: {}: {}", std::any::type_name::<Err>(), err);
    }
}*/

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
