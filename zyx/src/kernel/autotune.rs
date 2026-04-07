use crate::backend::{AutotuneConfig, Device, DeviceInfo, DeviceProgramId, MemoryPool, PoolBufferId};
use crate::error::BackendError;
use crate::kernel::{Kernel, Op, OpId, Scope};
use crate::rng::Rng;
use crate::shape::Dim;
use crate::slab::SlabId;
use crate::{DebugMask, Map, Set};
use nanoserde::{DeBin, SerBin};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, mpsc};
use std::{thread, u64};

const AVAILABLE_OPTIMIZATIONS: [fn(&Kernel) -> (Optimization, usize); 8] = [
    Kernel::opt_reassociate_commutative,
    //Kernel::opt_unroll,
    Kernel::opt_split_global_to_local,
    Kernel::opt_upcast,
    //Kernel::opt_register_tiling,
    Kernel::opt_fuse_mad,
    Kernel::opt_unfuse_mad,
    Kernel::opt_unroll_constant_loops,
    Kernel::opt_tiled_reduce,
    Kernel::opt_split_loop,
    //Kernel::opt_licm,
];

#[derive(Debug)]
pub enum Optimization {
    ReassociateCommutative,
    UnrollLoops {
        factors: Vec<usize>,
    },
    SplitGlobalToLocal {
        factors: Vec<(OpId, usize)>,
    },
    Upcast {
        factors: Vec<(OpId, usize)>,
    },
    RegisterTiling {
        reduce_splits: Map<OpId, Vec<usize>>,
        global_upcasts: Map<OpId, Vec<usize>>,
    },
    FuseMad,
    UnfuseMad,
    UnrollConstantLoops,
    TiledReduce {
        factors: Vec<(OpId, usize, usize)>,
    },
    SplitLoop {
        factors: Vec<(OpId, usize)>,
    },
    Licm,
}

impl Optimization {
    /// Applies the optimization with the given config ID.
    /// Config IDs are ordered such that lower IDs use hardware-aligned factors
    /// (e.g., warp size 32 for CUDA, wavefront size 64 for AMD) which are likely to perform better.
    pub fn apply(&self, kernel: &mut Kernel, config: usize) {
        match self {
            Optimization::ReassociateCommutative => {
                kernel.reassociate_commutative();
            }
            Optimization::UnrollLoops { factors } => {
                let factor = factors[config];
                if (kernel.ops.len().0 as usize) < 500 {
                    kernel.unroll_loops(factor);
                }
            }
            Optimization::SplitGlobalToLocal { factors } => {
                let (op_id, factor) = factors[config];
                let Op::Index { len, scope, axis } = kernel.ops[op_id].op else {
                    unreachable!()
                };
                debug_assert_eq!(scope, Scope::Global);
                //println!("Splitting global axis={axis} to factor={factor}");
                kernel.split_dim(
                    op_id,
                    vec![
                        Op::Index { len: len / factor, scope: Scope::Global, axis },
                        Op::Index { len: factor, scope: Scope::Local, axis },
                    ],
                );
            }
            Optimization::Upcast { factors } => {
                if factors.is_empty() {
                    return;
                }
                let (op_id, factor) = factors[config];
                kernel.upcast(op_id, factor);
            }
            Optimization::RegisterTiling { reduce_splits, global_upcasts } => {
                let n_global = global_upcasts.len();
                let n_reduce = reduce_splits.len();
                if n_global == 0 || n_reduce == 0 {
                    return;
                }

                let n_global_options: usize = global_upcasts.values().map(|v| v.len() + 1).product();
                //let n_reduce_options: usize = reduce_factors.values().map(|v| v.len()).product();

                let mut remaining_global = config % n_global_options;
                let mut remaining_reduce = config / n_global_options;

                for (reduce_id, factors) in reduce_splits.iter() {
                    let n_options = factors.len();
                    let factor_idx = remaining_reduce % n_options;
                    remaining_reduce /= n_options;
                    let reduce_factor = factors[factor_idx];

                    let original_len = if let Op::Loop { len, .. } = kernel.ops[*reduce_id].op {
                        len
                    } else {
                        continue;
                    };

                    kernel.split_dim(
                        *reduce_id,
                        vec![
                            Op::Loop { len: original_len / reduce_factor },
                            Op::Loop { len: reduce_factor },
                        ],
                    );
                }

                let mut new_global_upcasts = Vec::new();
                for (_, factors) in global_upcasts.iter() {
                    let n_options = factors.len() + 1;
                    let factor_idx = remaining_global % n_options;
                    remaining_global /= n_options;

                    let factor = if factor_idx == 0 { 1 } else { factors[factor_idx - 1] };
                    new_global_upcasts.push(factor);
                }

                let mut idx = 0;
                for (op_id, _) in global_upcasts.iter() {
                    let factor = new_global_upcasts[idx];
                    if factor > 1 {
                        kernel.upcast(*op_id, factor);
                    }
                    idx += 1;
                }
            }
            Optimization::FuseMad => {
                kernel.fuse_mad();
            }
            Optimization::UnfuseMad => {
                kernel.unfuse_mad();
            }
            Optimization::UnrollConstantLoops => {
                kernel.unroll_constant_loops();
            }
            Optimization::TiledReduce { factors } => {
                let (op_id, factor, tree_branch) = factors[config];
                kernel.tiled_reduce(op_id, factor, tree_branch);
            }
            Optimization::SplitLoop { factors } => {
                let (op_id, factor) = factors[config];
                let Op::Loop { len } = kernel.ops[op_id].op else {
                    unreachable!()
                };
                kernel.split_dim(op_id, vec![Op::Loop { len: len / factor }, Op::Loop { len: factor }]);
            }
            Optimization::Licm => {
                kernel.loop_invariant_code_motion();
            }
        }
    }
}

impl Kernel {
    pub fn run_always_on_optimizations(&mut self) {
        self.eliminate_zero_len_index();
        self.constant_folding();
        self.move_constants_to_beginning();
        self.loop_invariant_code_motion();
        self.common_subexpression_elimination();
        self.fold_accs();
        self.delete_empty_loops();
        self.unfold_pows();
        self.dead_code_elimination();
    }

    /// Autotune for debugging, applying only a selected series of optimizations
    #[allow(unused)]
    pub fn apply_selected_optimizations(
        &self,
        buffers: &[PoolBufferId],
        device: &mut Device,
        memory_pool: &mut MemoryPool,
        config: &AutotuneConfig,
        flop: u64,
        read_bytes: u64,
        write_bytes: u64,
        debug: DebugMask,
    ) -> (DeviceProgramId, OptSeq) {
        //eprintln!("=== autotune_debug called ===");
        let mut kernel = self.clone();
        //println!("Before associate_commutative:");
        //kernel.debug_colorless();
        kernel.run_always_on_optimizations();

        // Here come series of custom optimizations

        let (tiled_reduce_opt, n_tiled_reduce_configs) = kernel.opt_split_loop();
        if n_tiled_reduce_configs > 0 {
            tiled_reduce_opt.apply(&mut kernel, 0);
        }
        let (tiled_reduce_opt, n_tiled_reduce_configs) = kernel.opt_upcast();
        if n_tiled_reduce_configs > 0 {
            tiled_reduce_opt.apply(&mut kernel, 0);
        }
        let (tiled_reduce_opt, n_tiled_reduce_configs) = kernel.opt_upcast();
        if n_tiled_reduce_configs > 0 {
            tiled_reduce_opt.apply(&mut kernel, 1);
        }
        kernel.unroll_loops(8);

        //kernel.fuse_mad();
        /*let (tiled_reduce_opt, n_tiled_reduce_configs) = kernel.opt_tiled_reduce();
        if n_tiled_reduce_configs > 0 {
            tiled_reduce_opt.apply(&mut kernel, 1);
        }
        kernel.unroll_loops(16);
        kernel.unfuse_mad();
        kernel.run_always_on_optimizations();
        kernel.reassociate_commutative();
        kernel.loop_invariant_code_motion();*/
        //kernel.unroll_constant_loops();

        // Apply upcast (vectorization) with factor 2
        /*let (upcast_opt, n_upcast_configs) = kernel.opt_upcast();
        if n_upcast_configs > 0 {
            upcast_opt.apply(&mut kernel, 0);
        }
        kernel.run_always_on_optimizations();
        let (upcast_opt, n_upcast_configs) = kernel.opt_upcast();
        if n_upcast_configs > 0 {
            upcast_opt.apply(&mut kernel, 0);
        }*/
        //kernel.unroll_loops(2);

        // Tiled reduce disabled
        // Apply tiled reduce optimization
        //kernel.unroll_loops(4);

        //kernel.run_always_on_optimizations();
        kernel.run_always_on_optimizations();
        //kernel.debug_colorless();

        let (program_id, _) = kernel
            .launch_with_timings(buffers, device, memory_pool, debug, flop, read_bytes, write_bytes)
            .unwrap();

        (program_id, OptSeq { opts: Vec::new(), cost: Cost::default() })
    }

    /// Release mode autotune with beam like search and multithreading
    pub fn autotune(
        &self,
        buffers: &[PoolBufferId],
        device: &mut Device,
        memory_pool: &mut MemoryPool,
        config: &AutotuneConfig,
        flop: u64,
        read_bytes: u64,
        write_bytes: u64,
        debug: DebugMask,
    ) -> (DeviceProgramId, OptSeq) {
        if true {
            return self.apply_selected_optimizations(buffers, device, memory_pool, config, flop, read_bytes, write_bytes, debug);
        }

        let dev_info_ptr: *const DeviceInfo = device.info();
        let dev_info_ref = unsafe { &*dev_info_ptr };

        let n_launches = config.n_launches;
        let n_seeds = config.n_seeds;
        let n_added_per_step = config.n_added_per_step;
        let n_removed_per_step = config.n_removed_per_step;
        let n_total_opts = config.n_total_opts;

        let n_threads = std::thread::available_parallelism().map_or(4, |p| p.get());

        let pool = ThreadPool::new(n_threads);

        let mut items = Vec::new();
        let mut visited = Set::default();

        // Initial seed
        let mut kernel = self.clone();
        kernel.run_always_on_optimizations();

        let avail_configs = AVAILABLE_OPTIMIZATIONS.map(|config_fn| config_fn(&kernel));
        let total_configs = avail_configs.iter().map(|(_, x)| *x).sum::<usize>();
        let mult = n_seeds.min(total_configs);
        for opt_id in 0..AVAILABLE_OPTIMIZATIONS.len() {
            let n_configs_to_try = ((avail_configs[opt_id].1 * mult) as f32 / total_configs as f32).ceil() as usize;
            let mut config_id = 0;
            while config_id < n_configs_to_try {
                let mut new_kernel = kernel.clone();
                avail_configs[opt_id].0.apply(&mut new_kernel, config_id);
                new_kernel.run_always_on_optimizations();
                let hash = new_kernel.get_hash();
                if visited.contains(&hash) {
                    config_id += 1;
                    continue;
                }
                let new_seq = OptSeq { opts: vec![(opt_id, config_id)], cost: new_kernel.get_cost(dev_info_ref) };
                visited.insert(hash);
                items.push(new_seq);
                config_id += 1;
            }
        }

        let mut rng = Rng::seed_from_u64(3498203498);
        while items.len() < n_total_opts && items.len() > 0 {
            let items_ptr: *const Vec<OptSeq> = &items;
            let visited_ptr: *const Set<u64> = &visited;

            let jobs: Vec<_> = (0..n_threads)
                .map(|thread_id| {
                    let mut thread_kernel = kernel.clone();

                    let items_ref: &Vec<OptSeq> = unsafe { &*items_ptr };
                    let visited_ref: &Set<_> = unsafe { &*visited_ptr };

                    move || {
                        let mut local_items = Vec::new();
                        let mut local_visited = Set::default();

                        let mut rng = Rng::seed_from_u64(3902938402398423 + thread_id as u64);

                        let opt_seq = sample_best(items_ref, &mut rng);
                        opt_seq.apply(&mut thread_kernel);

                        let avail_configs = AVAILABLE_OPTIMIZATIONS.map(|config_fn| config_fn(&thread_kernel));
                        let total_configs = avail_configs.iter().map(|(_, x)| *x).sum::<usize>();
                        let mult = n_added_per_step.min(total_configs);

                        for (opt_id, _config_fn) in AVAILABLE_OPTIMIZATIONS.iter().enumerate() {
                            let n_configs_to_try =
                                (((&avail_configs[opt_id]).1 * mult) as f32 / total_configs as f32).ceil() as usize;

                            for config_id in 0..n_configs_to_try {
                                let mut opts = opt_seq.opts.clone();
                                opts.push((opt_id, config_id));

                                let mut new_kernel = thread_kernel.clone();
                                avail_configs[opt_id].0.apply(&mut new_kernel, config_id);
                                new_kernel.run_always_on_optimizations();
                                let hash = new_kernel.get_hash();
                                if visited_ref.contains(&hash) || local_visited.contains(&hash) {
                                    continue;
                                }
                                let new_seq = OptSeq { opts, cost: new_kernel.get_cost(dev_info_ref) };
                                local_visited.insert(hash);
                                local_items.push(new_seq);
                            }
                        }

                        (local_items, local_visited)
                    }
                })
                .collect();

            let mut total_len = 0;
            let results = pool.execute_batch(jobs);
            for (local_items, local_visited) in results {
                total_len += local_items.len();
                items.extend(local_items);
                visited.extend(local_visited);
            }

            if total_len == 0 {
                break;
            }

            remove_worst(&mut items, n_removed_per_step, &mut rng);
        }

        pool.shutdown();

        let mut launched_kernels = Set::default();
        let mut best_time = u64::MAX;
        let mut best_program = DeviceProgramId::NULL;
        let mut best_opt_seq = OptSeq { opts: Vec::new(), cost: Cost::default() };
        let mut i = n_launches;
        while i > 0 {
            let opt_seq = sample_best(&items, &mut rng);
            let mut kernel = kernel.clone();

            for &(opt_id, opt_cfg) in &opt_seq.opts {
                let (opt, _) = AVAILABLE_OPTIMIZATIONS[opt_id](&kernel);
                println!(
                    "Running opt: {opt_id}, cfg={opt_cfg} -> {opt:?} -> {:?}",
                    AVAILABLE_OPTIMIZATIONS[opt_id]
                );
                opt.apply(&mut kernel, opt_cfg);
            }
            kernel.run_always_on_optimizations();

            if launched_kernels.insert(kernel.get_hash()) {
                if debug.ir() {
                    kernel.debug();
                }

                let Ok((program_id, time)) =
                    kernel.launch_with_timings(buffers, device, memory_pool, debug, flop, read_bytes, write_bytes)
                else {
                    continue;
                };

                if time < best_time {
                    best_program = program_id;
                    best_time = time;
                    best_opt_seq = opt_seq.clone();
                }
            }

            i -= 1;
        }

        // println!("DEBUG: Returning best_program={:?}, best_time={}", best_program, best_time);

        (best_program, best_opt_seq)
    }

    pub fn get_cost(&self, dev_info: &DeviceInfo) -> Cost {
        // TODO add measuring bank conflicts
        let mut n_instructions = 0;
        let mut n_scoped_loads = [0u64, 0, 0];
        let mut n_scoped_stores = [0u64, 0, 0];

        let mut gws = vec![1; 3];
        let mut lws = vec![1; 3];
        let mut loop_mult = 1;
        let mut latest_loop_lengths = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Cast { .. } => {
                    n_instructions += loop_mult;
                }
                Op::Unary { .. } => {
                    n_instructions += loop_mult;
                }
                Op::Binary { .. } => {
                    n_instructions += loop_mult;
                }
                Op::Const(_) => {}
                Op::Define { .. } => {}
                Op::Load { src, vlen, .. } => {
                    n_instructions += loop_mult;
                    let Op::Define { scope, .. } = self.ops[src].op else {
                        unreachable!()
                    };
                    match scope {
                        Scope::Global => n_scoped_loads[0] += loop_mult * vlen as u64,
                        Scope::Local => n_scoped_loads[1] += loop_mult * vlen as u64,
                        Scope::Register => n_scoped_loads[2] += loop_mult * vlen as u64,
                    }
                }
                Op::Store { dst, vlen, .. } => {
                    n_instructions += loop_mult * 3;
                    let Op::Define { scope, .. } = self.ops[dst].op else {
                        unreachable!()
                    };
                    match scope {
                        Scope::Global => n_scoped_stores[0] += loop_mult * vlen as u64,
                        Scope::Local => n_scoped_stores[1] += loop_mult * vlen as u64,
                        Scope::Register => n_scoped_stores[2] += loop_mult * vlen as u64,
                    }
                }
                Op::Index { len, scope, axis } => match scope {
                    Scope::Global => gws[axis as usize] = len,
                    Scope::Local => lws[axis as usize] = len,
                    Scope::Register => todo!(),
                },
                Op::Loop { len } => {
                    n_instructions += loop_mult * 3;
                    loop_mult *= len as u64;
                    latest_loop_lengths.push(len as u64);
                }
                Op::EndLoop => {
                    loop_mult /= latest_loop_lengths.pop().unwrap();
                }
                Op::Mad { .. } => {
                    n_instructions += loop_mult;
                }
                Op::WMMA { dims, .. } => {
                    let (m, n, k) = dims.decompose_mnk();
                    let warp = dev_info.warp_size as u64;
                    let cost = (m * n * k) as u64 / warp;
                    n_instructions += loop_mult * cost;
                }
                Op::Vectorize { .. } => {
                    // TODO multiply all ops that are vectorized by the vectorization factor
                }
                Op::Barrier { .. } => {
                    n_instructions += loop_mult * 5;
                }
                Op::If { .. } => {
                    n_instructions += loop_mult * 3;
                }
                Op::EndIf => {}
                Op::Devectorize { .. } => {}
                Op::ConstView(_) => todo!(),
                Op::LoadView(_) => todo!(),
                Op::StoreView { .. } => todo!(),
                Op::Move { .. } => todo!(),
                Op::Reduce { .. } => todo!(),
            }
            op_id = self.next_op(op_id);
        }

        let global_ws = gws.iter().product::<Dim>() as u64;
        let n_threads = lws.iter().product::<Dim>() as u32;
        let instructions_per_thread = n_instructions as u32;
        let global_loads_per_thread = n_scoped_loads[0] as u32;
        let local_loads_per_thread = n_scoped_loads[1] as u32;
        let global_stores_per_thread = n_scoped_stores[0] as u32;
        let local_stores_per_thread = n_scoped_stores[1] as u32;

        Cost {
            global_ws,
            n_threads,
            instructions_per_thread,
            global_loads_per_thread,
            local_loads_per_thread,
            global_stores_per_thread,
            local_stores_per_thread,
        }
    }

    pub fn get_hash(&self) -> u64 {
        let mut hasher = crate::chasher::CHasher::default();
        self.hash(&mut hasher);
        hasher.finish()
    }

    pub fn launch_with_timings(
        &self,
        buffers: &[PoolBufferId],
        device: &mut Device,
        memory_pool: &mut MemoryPool,
        debug: DebugMask,
        flops: u64,
        bytes_read: u64,
        bytes_written: u64,
    ) -> Result<(DeviceProgramId, u64), BackendError> {
        let program_id = device.compile(self, debug.asm())?;
        let begin = std::time::Instant::now();
        let event = device.launch(program_id, memory_pool, buffers, Vec::new())?;
        memory_pool.sync_events(vec![event])?;
        let nanos = begin.elapsed().as_nanos() as u64;
        let perf = crate::cache::get_perf(flops, bytes_read, bytes_written, nanos);
        if debug.perf() {
            println!("{perf}");
        }
        Ok((program_id, nanos))
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, DeBin, SerBin)]
pub struct Cost {
    global_ws: u64,
    n_threads: u32,
    instructions_per_thread: u32,
    global_loads_per_thread: u32,
    local_loads_per_thread: u32,
    global_stores_per_thread: u32,
    local_stores_per_thread: u32,
}

impl PartialEq for Cost {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == core::cmp::Ordering::Equal
    }
}

impl Eq for Cost {}

impl Ord for Cost {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Global memory accesses are most expensive, prioritize minimizing them
        // Global stores are most critical (write bandwidth)
        self.global_stores_per_thread
            .cmp(&other.global_stores_per_thread)
            // Then global loads
            .then(self.global_loads_per_thread.cmp(&other.global_loads_per_thread))
            // Then total instructions (local access counts as ~1 instruction)
            .then(
                (self.instructions_per_thread + self.local_loads_per_thread + self.local_stores_per_thread)
                    .cmp(&(other.instructions_per_thread + other.local_loads_per_thread + other.local_stores_per_thread)),
            )
            // Fewer threads with more work per thread is slightly preferred (less overhead)
            .then(other.n_threads.cmp(&self.n_threads))
    }
}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, DeBin, SerBin)]
pub struct OptSeq {
    opts: Vec<(usize, usize)>,
    cost: Cost,
}

impl OptSeq {
    pub fn apply(&self, kernel: &mut Kernel) {
        for &(opt_id, opt_cfg) in &self.opts {
            let (opt, _): (Optimization, usize) = AVAILABLE_OPTIMIZATIONS[opt_id](kernel);
            opt.apply(kernel, opt_cfg);
        }
    }
}

fn remove_worst(items: &mut Vec<OptSeq>, mut n: usize, rng: &mut Rng) {
    if items.len() < 10 * n {
        return;
    }
    while n > 0 && !items.is_empty() {
        // Tournament among random samples biased toward high cost
        const K: usize = 4; // number of candidates
        let mut worst_idx = rng.range::<u64>(0..items.len() as u64) as usize;
        let mut worst_cost = items[worst_idx].cost;
        for _ in 1..K {
            let i = rng.range::<u64>(0..items.len() as u64) as usize;
            let cost = items[i].cost;
            if cost > worst_cost {
                worst_idx = i;
                worst_cost = cost;
            }
        }

        items.swap_remove(worst_idx);

        n -= 1;
    }
}

fn sample_best<'a>(items: &'a [OptSeq], rng: &mut Rng) -> &'a OptSeq {
    debug_assert!(!items.is_empty(), "sample_best called with empty items");
    const K: usize = 16;
    let len = items.len();
    let mut best_idx = rng.range::<u64>(0..len as u64) as usize;
    let mut best_cost = items[best_idx].cost;
    for _ in 1..K {
        let i = rng.range::<u64>(0..len as u64) as usize;
        let cost = items[i].cost;
        if cost < best_cost {
            best_idx = i;
            best_cost = cost;
        }
    }

    &items[best_idx]
}

pub struct ThreadPool {
    workers: Vec<thread::JoinHandle<()>>,
    job_tx: mpsc::Sender<Box<dyn FnOnce() + Send>>,
}

impl ThreadPool {
    pub fn new(n_threads: usize) -> Self {
        let (job_tx, job_rx) = mpsc::channel::<Box<dyn FnOnce() + Send>>();
        let job_rx = Arc::new(Mutex::new(job_rx));

        let mut workers = Vec::with_capacity(n_threads);

        for _ in 0..n_threads {
            let job_rx = Arc::clone(&job_rx);
            let handle = thread::spawn(move || {
                loop {
                    let job = {
                        let lock = job_rx.lock().unwrap();
                        lock.recv()
                    };
                    match job {
                        Ok(job) => job(),
                        Err(_) => break, // channel closed → exit thread
                    }
                }
            });
            workers.push(handle);
        }

        Self { workers, job_tx }
    }

    /// Execute a batch of jobs and wait for them to finish
    pub fn execute_batch<T: Send + 'static>(&self, jobs: Vec<impl FnOnce() -> T + Send + 'static>) -> Vec<T> {
        let n_threads = self.workers.len();
        assert_eq!(jobs.len(), n_threads, "jobs must equal n_threads");

        let (res_tx, res_rx) = mpsc::channel();

        for job in jobs {
            let res_tx = res_tx.clone();
            self.job_tx
                .send(Box::new(move || {
                    let res = job();
                    let _ = res_tx.send(res);
                }))
                .unwrap();
        }

        let mut results = Vec::with_capacity(n_threads);
        for _ in 0..n_threads {
            results.push(res_rx.recv().unwrap());
        }

        results
    }

    /// Manually shutdown the pool by consuming it
    pub fn shutdown(self) {
        let ThreadPool { job_tx, workers } = self;

        // Close the channel
        drop(job_tx);

        // Join all threads
        for worker in workers {
            let _ = worker.join();
        }
    }
}
