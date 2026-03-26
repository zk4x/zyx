use crate::backend::{AutotuneConfig, BufferId, Device, MemoryPool, ProgramId};
use crate::chasher::CHasher;
use crate::error::BackendError;
use crate::kernel::Kernel;
use crate::rng::Rng;
use crate::{DebugMask, Map};
use std::hash::{Hash, Hasher};
use std::u64;

impl Kernel {
    pub fn get_cost(&self) -> u64 {
        0
    }

    pub fn hash_kernel(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hash;
        use std::hash::Hasher;
        let mut hasher = DefaultHasher::new();
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
        let available_opts: [(fn(&Kernel) -> u16, fn(&mut Kernel, u16)); 1] =
            [(Self::opt_no_config, Self::constant_folding)];

        let n_seeds = 100;
        let n_added_per_step = 10;
        let n_removed_per_step = 5;
        let n_total_opts = 1000;

        let mut rng = &mut Rng::seed_from_u64(1098102948);
        let mut pool = Pool::new();

        // Initial seed
        let kernel = self.clone();
        let total_configs: usize = available_opts.iter().map(|(config_fn, _)| config_fn(&kernel) as usize).sum();
        let mut opt_id = 0;
        for (config_fn, optimization_fn) in &available_opts {
            let n_avail_configs = config_fn(&kernel) as usize;
            let n_configs_to_try = (n_avail_configs as f32 * n_seeds as f32 / total_configs as f32).ceil() as u16;
            let mut config_id = 0;
            while config_id < n_configs_to_try {
                let mut opt_seq = Optimization::new(vec![(opt_id, config_id)]);
                if pool.index.contains_key(&opt_seq.hash) {
                    config_id += 1;
                    continue;
                }
                let mut kernel = kernel.clone();
                optimization_fn(&mut kernel, config_id);
                opt_seq.cost = kernel.get_cost();
                pool.insert(opt_seq);
                config_id += 1;
            }
            opt_id += 1;
        }

        while pool.len() < n_total_opts {
            let mut kernel = self.clone();
            // randomly select a kernel, with lower cost ones having bigger prob
            let opt_seq = pool.sample_best(rng).clone();
            for &(opt_id, opt_cfg) in &opt_seq.opts {
                available_opts[opt_id as usize].1(&mut kernel, opt_cfg);
            }

            let total_configs: usize = available_opts.iter().map(|(config_fn, _)| config_fn(&kernel) as usize).sum();
            let mut opt_id = 0;
            for (config_fn, optimization_fn) in &available_opts {
                let n_avail_configs = config_fn(&kernel) as usize;
                let n_configs_to_try =
                    (n_avail_configs as f32 * n_added_per_step as f32 / total_configs as f32).ceil() as u16;
                let mut config_id = 0;
                while config_id < n_configs_to_try {
                    let mut opts = opt_seq.opts.clone();
                    opts.push((opt_id, config_id));
                    let mut opt_seq = Optimization::new(opts);
                    if pool.index.contains_key(&opt_seq.hash) {
                        config_id += 1;
                        continue;
                    }
                    let mut kernel = kernel.clone();
                    optimization_fn(&mut kernel, config_id);
                    opt_seq.cost = kernel.get_cost();
                    pool.insert(opt_seq);
                    config_id += 1;
                }
                opt_id += 1;
            }

            pool.remove_worst(n_removed_per_step, &mut rng);
        }

        // TODO add hardware validation of top n kernels

        if debug.ir() {
            kernel.debug();
        }

        device.compile(&kernel, debug.asm()).unwrap()
    }
}

#[derive(Clone)]
struct Optimization {
    opts: Vec<(u16, u16)>,
    cost: u64,
    hash: u64,
}

impl Optimization {
    fn new(opts: Vec<(u16, u16)>) -> Self {
        let mut h = CHasher::default();
        opts.hash(&mut h);
        let hash = h.finish();
        Self { opts, cost: 0, hash }
    }
}

struct Pool {
    items: Vec<Optimization>,
    index: Map<u64, usize>, // hash -> index
}

impl Pool {
    fn new() -> Self {
        Self { items: Vec::new(), index: Map::default() }
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    // insert with dedup
    fn insert(&mut self, opt: Optimization) -> bool {
        if self.index.contains_key(&opt.hash) {
            return false;
        }
        let idx = self.items.len();
        self.index.insert(opt.hash, idx);
        self.items.push(opt);
        true
    }

    // fast biased sampling
    pub fn sample_best<'a>(&'a self, rng: &mut Rng) -> &'a Optimization {
        const K: usize = 16;
        let len = self.items.len();
        let mut best_idx = rng.range::<u64>(0..len as u64) as usize;
        let mut best_cost = self.items[best_idx].cost;
        for _ in 1..K {
            let i = rng.range::<u64>(0..len as u64) as usize;
            let cost = self.items[i].cost;
            if cost < best_cost {
                best_idx = i;
                best_cost = cost;
            }
        }

        &self.items[best_idx]
    }

    // remove worst N
    fn remove_worst(&mut self, mut n: usize, rng: &mut Rng) {
        if self.items.is_empty() {
            return;
        }
        while n > 0 && !self.items.is_empty() {
            // Tournament among random samples biased toward high cost
            const K: usize = 4; // number of candidates
            let mut worst_idx = rng.range::<u64>(0..self.items.len() as u64) as usize;
            let mut worst_cost = self.items[worst_idx].cost;
            for _ in 1..K {
                let i = rng.range::<u64>(0..self.items.len() as u64) as usize;
                let cost = self.items[i].cost;
                if cost > worst_cost {
                    worst_idx = i;
                    worst_cost = cost;
                }
            }

            let removed = self.items.swap_remove(worst_idx);
            self.index.remove(&removed.hash);
            if worst_idx != self.items.len() {
                let moved = &self.items[worst_idx];
                self.index.insert(moved.hash, worst_idx);
            }

            n -= 1;
        }
    }
}
