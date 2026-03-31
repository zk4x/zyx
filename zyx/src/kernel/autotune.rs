use crate::backend::{AutotuneConfig, BufferId, Device, DeviceInfo, MemoryPool, ProgramId};
use crate::error::BackendError;
use crate::kernel::{Kernel, Op, OpId, Scope};
use crate::rng::Rng;
use crate::shape::Dim;
use crate::slab::SlabId;
use crate::{DebugMask, Map, Set};
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, mpsc};
use std::{thread, u64};

pub enum Optimization {
    Null,
    ReassociateCommutative,
    UnrollLoops { factors: Vec<usize> },
    SplitGlobalToLocal { factors: Vec<(OpId, usize)> },
}

impl Optimization {
    pub fn reassociate_commutative(_kernel: &Kernel) -> (Self, u16) {
        (Optimization::ReassociateCommutative, 1)
    }

    pub fn unroll(_kernel: &Kernel) -> (Self, u16) {
        (
            Optimization::UnrollLoops {
                factors: vec![8, 4, 16, 2],
            },
            4,
        )
    }

    pub fn split_global_to_local(kernel: &Kernel) -> (Self, u16) {
        let mut op_id = kernel.head;
        let mut factors = Vec::new();
        let mut seen_axes = Map::default();
        while !op_id.is_null() {
            if let Op::Index { len, scope, axis } = kernel.ops[op_id].op {
                let mut l_factors: Vec<usize> = vec![32, 64, 16, 8, 4, 2];
                if scope == Scope::Global {
                    l_factors.retain(|&f| len.is_multiple_of(f));
                    for &f in &l_factors {
                        factors.push((op_id, f));
                    }
                    seen_axes.insert(axis, op_id);
                }
                if scope == Scope::Local {
                    if let Some(global_id) = seen_axes.get(&axis) {
                        factors.retain(|(op_id, _)| global_id != op_id);
                    }
                }
            }
            op_id = kernel.next_op(op_id);
        }
        let n_configs = factors.len() as u16;
        (Optimization::SplitGlobalToLocal { factors }, n_configs)
    }

    /// Applies the optimization with the given config ID.
    /// Config IDs are ordered such that lower IDs use hardware-aligned factors
    /// (e.g., warp size 32 for CUDA, wavefront size 64 for AMD) which are likely to perform better.
    pub fn apply(&self, kernel: &mut Kernel, config: u16) {
        match self {
            Optimization::Null => {}
            Optimization::ReassociateCommutative => {
                kernel.reassociate_commutative(config);
            }
            Optimization::UnrollLoops { factors } => {
                let factor = factors[config as usize];
                kernel.unroll_loops(factor);
            }
            Optimization::SplitGlobalToLocal { factors } => {
                let (op_id, factor) = factors[config as usize];
                let Op::Index { len, scope, axis } = kernel.ops[op_id].op else {
                    unreachable!()
                };
                kernel.split_dim(
                    op_id,
                    vec![
                        Op::Index {
                            len: len / factor,
                            scope: Scope::Global,
                            axis,
                        },
                        Op::Index {
                            len: factor,
                            scope: Scope::Local,
                            axis,
                        },
                    ],
                );
            }
        }
    }
}

impl Kernel {
    pub fn run_always_on_optimizations(&mut self) {
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
    pub fn autotune1(
        &self,
        _buffers: &[BufferId],
        device: &mut Device,
        _memory_pool: &mut MemoryPool,
        _config: &AutotuneConfig,
        debug: DebugMask,
    ) -> ProgramId {
        let mut kernel = self.clone();
        //println!("Before associate_commutative:");
        //kernel.debug_colorless();
        kernel.run_always_on_optimizations();

        // Here come series of custom optimizations
        kernel.reassociate_commutative(0);
        kernel.unroll_loops(2); // unroll with dim=2

        kernel.run_always_on_optimizations();

        device.compile(&kernel, debug.asm()).unwrap()
    }

    /// Release mode autotune with beam like search and multithreading
    pub fn autotune(
        &self,
        buffers: &[BufferId],
        device: &mut Device,
        memory_pool: &mut MemoryPool,
        config: &AutotuneConfig,
        debug: DebugMask,
    ) -> ProgramId {
        let available_opts: [fn(&Kernel) -> (Optimization, u16); _] = [
            Optimization::reassociate_commutative,
            Optimization::unroll,
            Optimization::split_global_to_local,
        ];

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

        let avail_configs = available_opts.map(|config_fn| config_fn(&kernel));
        let total_configs = avail_configs.iter().map(|(_, x)| *x as usize).sum::<usize>();
        let mult = n_seeds.min(total_configs);
        for opt_id in 0..available_opts.len() {
            let n_configs_to_try =
                ((avail_configs[opt_id].1 as usize * mult) as f32 / total_configs as f32).ceil() as u16;
            let mut config_id = 0;
            while config_id < n_configs_to_try {
                let mut opt_seq = OptSeq {
                    opts: vec![(opt_id as u16, config_id)],
                    cost: 0,
                };
                let mut new_kernel = kernel.clone();
                avail_configs[opt_id].0.apply(&mut new_kernel, config_id);
                new_kernel.run_always_on_optimizations();
                let hash = new_kernel.get_hash();
                if visited.contains(&hash) {
                    config_id += 1;
                    continue;
                }
                opt_seq.cost = new_kernel.get_cost(dev_info_ref) as u64;
                visited.insert(hash);
                items.push(opt_seq);
                config_id += 1;
            }
        }

        //let mut rng = Rng::seed_from_systime();
        let mut rng = Rng::seed_from_u64(3498203498);
        while items.len() < n_total_opts && items.len() > 0 {
            let items_ptr: *const Vec<OptSeq> = &items;
            let visited_ptr: *const Set<u64> = &visited;

            let jobs: Vec<_> = (0..n_threads)
                .map(|thread_id| {
                    let base_kernel = self.clone();

                    let items_ref: &Vec<OptSeq> = unsafe { &*items_ptr };
                    let visited_ref: &Set<_> = unsafe { &*visited_ptr };

                    move || {
                        let mut kernel = base_kernel.clone();
                        let mut local_items = Vec::new();
                        let mut local_visited = Set::default();

                        let mut rng = Rng::seed_from_u64(
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_nanos() as u64
                                + thread_id as u64,
                        );

                        let opt_seq = sample_best(items_ref, &mut rng);

                        for &(opt_id, opt_cfg) in &opt_seq.opts {
                            let (opt, _) = available_opts[opt_id as usize](&kernel);
                            opt.apply(&mut kernel, opt_cfg);
                        }

                        let avail_configs = available_opts.map(|config_fn| config_fn(&kernel));
                        let total_configs = avail_configs.iter().map(|(_, x)| *x as usize).sum::<usize>();
                        let mult = n_added_per_step.min(total_configs);

                        for (opt_id, _config_fn) in available_opts.iter().enumerate() {
                            let av_configs = &avail_configs[opt_id];
                            let n_configs_to_try =
                                ((av_configs.1 as usize * mult) as f32 / total_configs as f32).ceil() as u16;

                            for config_id in 0..n_configs_to_try {
                                let mut opts = opt_seq.opts.clone();
                                opts.push((opt_id as u16, config_id));
                                let mut new_seq = OptSeq { opts, cost: 0 };

                                let mut new_kernel = kernel.clone();
                                avail_configs[opt_id].0.apply(&mut new_kernel, config_id);
                                new_kernel.run_always_on_optimizations();
                                let hash = new_kernel.get_hash();
                                if visited_ref.contains(&hash) || local_visited.contains(&hash) {
                                    continue;
                                }
                                new_seq.cost = new_kernel.get_cost(dev_info_ref) as u64;
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
        let mut best_program = ProgramId::NULL;
        let mut i = n_launches;
        while i > 0 {
            let opt_seq = sample_best(&items, &mut rng);
            let mut kernel = kernel.clone();

            for &(opt_id, opt_cfg) in &opt_seq.opts {
                let (opt, _) = available_opts[opt_id as usize](&kernel);
                opt.apply(&mut kernel, opt_cfg);
            }
            kernel.run_always_on_optimizations();

            if launched_kernels.insert(kernel.get_hash()) {
                if debug.ir() {
                    kernel.debug();
                }

                let Ok((program_id, time)) = kernel.launch_with_timings(buffers, device, memory_pool, debug) else {
                    continue;
                };

                if time < best_time {
                    best_program = program_id;
                    best_time = time;
                }
            }

            i -= 1;
        }

        // println!("DEBUG: Returning best_program={:?}, best_time={}", best_program, best_time);

        best_program
    }

    pub fn get_cost(&self, dev_info: &DeviceInfo) -> f64 {
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
                    n_instructions += loop_mult * 3;
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
                Op::Devectorize { .. } => {}
                Op::ConstView(_) => todo!(),
                Op::LoadView(_) => todo!(),
                Op::StoreView { .. } => todo!(),
                Op::Move { .. } => todo!(),
                Op::Reduce { .. } => todo!(),
            }
            op_id = self.next_op(op_id);
        }

        gws.iter().product::<Dim>() as f64
            * lws.iter().product::<Dim>() as f64
            * (n_instructions as f64
                + n_scoped_loads[0] as f64 * 10.
                + n_scoped_loads[1] as f64 * 3.
                + n_scoped_loads[2] as f64 * 1.1
                + n_scoped_stores[0] as f64 * 10.
                + n_scoped_stores[1] as f64 * 3.
                + n_scoped_stores[2] as f64 * 1.1)
    }

    pub fn get_hash(&self) -> u64 {
        let mut hasher = crate::chasher::CHasher::default();
        self.hash(&mut hasher);
        hasher.finish()
    }

    pub fn launch_with_timings(
        &self,
        buffers: &[BufferId],
        device: &mut Device,
        memory_pool: &mut MemoryPool,
        debug: DebugMask,
    ) -> Result<(ProgramId, u64), BackendError> {
        let program_id = device.compile(self, debug.asm())?;
        let begin = std::time::Instant::now();
        let event = device.launch(program_id, memory_pool, buffers, Vec::new())?;
        memory_pool.sync_events(vec![event])?;
        let nanos = begin.elapsed().as_nanos() as u64;
        Ok((program_id, nanos))
    }
}

#[derive(Debug)]
struct OptSeq {
    opts: Vec<(u16, u16)>,
    cost: u64,
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
