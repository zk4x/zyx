use crate::backend::{AutotuneConfig, BufferId, Device, DeviceInfo, MemoryPool, ProgramId};
use crate::error::BackendError;
use crate::kernel::{Kernel, Op, Scope};
use crate::rng::Rng;
use crate::{DebugMask, Set};
use std::hash::{Hash, Hasher};
use std::sync::mpsc;
use std::thread;
use std::u64;

impl Kernel {
    pub fn get_cost(&self, dev_info: &DeviceInfo) -> u64 {
        // TODO add measuring bank conflicts
        let mut n_instructions = 0;
        let mut n_scoped_loads = [0u64, 0, 0];
        let mut n_scoped_stores = [0u64, 0, 0];

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
                    let Op::Define { scope, .. } = self.ops[src].op else { unreachable!() };
                    match scope {
                        Scope::Global => n_scoped_loads[0] += loop_mult * vlen as u64,
                        Scope::Local => n_scoped_loads[1] += loop_mult * vlen as u64,
                        Scope::Register => n_scoped_loads[2] += loop_mult * vlen as u64,
                    }
                }
                Op::Store { dst, vlen, .. } => {
                    n_instructions += loop_mult * 3;
                    let Op::Define { scope, .. } = self.ops[dst].op else { unreachable!() };
                    match scope {
                        Scope::Global => n_scoped_stores[0] += loop_mult * vlen as u64,
                        Scope::Local => n_scoped_stores[1] += loop_mult * vlen as u64,
                        Scope::Register => n_scoped_stores[2] += loop_mult * vlen as u64,
                    }
                }
                Op::Index { .. } => {}
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

        n_instructions
            + n_scoped_loads[0] * 3
            + n_scoped_loads[1] * 2
            + n_scoped_loads[2] * 1
            + n_scoped_stores[0] * 3
            + n_scoped_stores[1] * 2
            + n_scoped_stores[2] * 1
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
    ) -> Result<(ProgramId, u64), BackendError> {
        let program_id = device.compile(self, false)?;
        let begin = std::time::Instant::now();
        let event = device.launch(program_id, memory_pool, buffers, Vec::new())?;
        memory_pool.sync_events(vec![event])?;
        let nanos = begin.elapsed().as_nanos() as u64;
        Ok((program_id, nanos))
    }

    pub fn opt_no_config(&self) -> u16 {
        1
    }

    pub fn autotune(
        &mut self,
        buffers: &[BufferId],
        device: &mut Device,
        memory_pool: &mut MemoryPool,
        config: &AutotuneConfig,
        debug: DebugMask,
    ) -> ProgramId {
        let available_opts: [(fn(&Kernel) -> u16, fn(&mut Kernel, u16)); _] =
            [(Self::opt_no_config, Self::constant_folding)];

        let dev_info_ptr: *const DeviceInfo = device.info();
        let dev_info_ref = unsafe { &*dev_info_ptr };

        let n_seeds = 100;
        let n_added_per_step = 10;
        let n_removed_per_step = 5;
        let n_total_opts = 1000;

        let n_threads = 4;

        let pool = ThreadPool::new(n_threads);

        let mut items = Vec::new();
        let mut visited = Set::default();

        // Initial seed
        let kernel = self.clone();

        let avail_configs = available_opts.map(|(config_fn, _)| config_fn(&kernel) as usize);
        let total_configs = avail_configs.iter().sum::<usize>();
        let mult = n_seeds.min(total_configs);
        let mut opt_id = 0;
        for (_, optimization_fn) in &available_opts {
            let n_configs_to_try =
                ((avail_configs[opt_id as usize] * mult) as f32 / total_configs as f32).ceil() as u16;
            let mut config_id = 0;
            while config_id < n_configs_to_try {
                let mut opt_seq = Optimization { opts: vec![(opt_id, config_id)], cost: 0 };
                let mut new_kernel = kernel.clone();
                optimization_fn(&mut new_kernel, config_id);
                let hash = kernel.get_hash();
                if visited.contains(&hash) {
                    config_id += 1;
                    continue;
                }
                opt_seq.cost = new_kernel.get_cost(dev_info_ref);
                visited.insert(hash);
                items.push(opt_seq);
                config_id += 1;
            }
            opt_id += 1;
        }

        let mut rng = Rng::seed_from_systime();
        while items.len() < n_total_opts && items.len() > 0 {
            let items_ptr: *const Vec<Optimization> = &items;
            let visited_ptr: *const Set<u64> = &visited;

            let jobs: Vec<_> = (0..n_threads)
                .map(|thread_id| {
                    let base_kernel = self.clone();

                    // SAFETY: we promise items/visited are read-only while threads run
                    // rust is showing it's amazingness again
                    let items_ref: &Vec<Optimization> = unsafe { &*items_ptr };
                    let visited_ref: &Set<_> = unsafe { &*visited_ptr };

                    move || {
                        let mut kernel = base_kernel.clone();
                        let mut local_items = Vec::new();
                        let mut local_visited = Set::default();

                        let mut rng = Rng::seed_from_u64(
                            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
                                as u64
                                + thread_id as u64,
                        );

                        let opt_seq = sample_best(items_ref, &mut rng);

                        for &(opt_id, opt_cfg) in &opt_seq.opts {
                            available_opts[opt_id as usize].1(&mut kernel, opt_cfg);
                        }

                        let avail_configs = available_opts.map(|(config_fn, _)| config_fn(&kernel) as usize);
                        let total_configs = avail_configs.iter().sum::<usize>();
                        let mult = n_added_per_step.min(total_configs);

                        for (opt_id, (_, optimization_fn)) in available_opts.iter().enumerate() {
                            let av_configs = avail_configs[opt_id];
                            let n_configs_to_try = ((av_configs * mult) as f32 / total_configs as f32).ceil() as u16;

                            for config_id in 0..n_configs_to_try {
                                let mut opts = opt_seq.opts.clone();
                                opts.push((opt_id as u16, config_id));
                                let mut new_seq = Optimization { opts, cost: 0 };

                                let mut new_kernel = kernel.clone();
                                optimization_fn(&mut new_kernel, config_id);
                                let hash = new_kernel.get_hash();
                                if visited_ref.contains(&hash) || local_visited.contains(&hash) {
                                    continue;
                                }
                                new_seq.cost = new_kernel.get_cost(dev_info_ref);
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

        // TODO add hardware validation of top n kernels

        let mut kernel = self.clone();
        if let Some(opt_seq) = items.iter().min_by_key(|x| x.cost) {
            println!("Selected sequence: {opt_seq:?}");
            for &(opt_id, opt_cfg) in &opt_seq.opts {
                available_opts[opt_id as usize].1(&mut kernel, opt_cfg);
            }
        }

        if debug.ir() {
            kernel.debug();
        }

        device.compile(&kernel, debug.asm()).unwrap()
    }
}

#[derive(Debug)]
struct Optimization {
    opts: Vec<(u16, u16)>,
    cost: u64,
}

fn remove_worst(items: &mut Vec<Optimization>, mut n: usize, rng: &mut Rng) {
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

fn sample_best<'a>(items: &'a [Optimization], rng: &mut Rng) -> &'a Optimization {
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
        let job_rx = std::sync::Arc::new(std::sync::Mutex::new(job_rx));

        let mut workers = Vec::with_capacity(n_threads);

        for _ in 0..n_threads {
            let job_rx = job_rx.clone();
            let handle = thread::spawn(move || {
                loop {
                    let job = job_rx.lock().unwrap().recv();
                    match job {
                        Ok(job) => job(),
                        Err(_) => break,
                    }
                }
            });
            workers.push(handle);
        }

        Self { workers, job_tx }
    }

    /// Execute a batch of jobs and wait for them to finish
    pub fn execute_batch<T: Send + 'static>(&self, jobs: Vec<impl FnOnce() -> T + Send + 'static>) -> Vec<T> {
        let n = jobs.len();
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

        let mut results = Vec::with_capacity(n);
        for _ in 0..n {
            results.push(res_rx.recv().unwrap());
        }

        results
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
