use crate::backend::{AutotuneConfig, BufferId, Device, MemoryPool, ProgramId};
use crate::error::BackendError;
use crate::kernel::Kernel;
use std::collections::BinaryHeap;
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

    pub fn autotune(
        &mut self,
        buffers: &[BufferId],
        device: &mut Device,
        memory_pool: &mut MemoryPool,
        config: &AutotuneConfig,
    ) -> ProgramId {
        // These are all available optimization functions
        // In each row, first function gives available number of configurations for the optimization at the current kernel state,
        // while the other function applies the optimization on the kernel using given configuration
        // e.g. (Kernel::unroll_loops_configs, Kernel::unroll_loops),
        let optimizations: [(fn(&Kernel) -> u16, fn(&mut Kernel, u16)); 20] = todo!();

        let beam_width = 100; // how many kernels to keep
        let n_optimization_passes = 10000; // how many optimization configs to try at each step
        let n_steps = 7; // how many optimization steps

        let mut beam = BinaryHeap::new();
        beam.push(BeamEntry { kernel: self.clone(), cost: self.get_cost() });

        let mut max_cost = 0;

        for iteration in 0..n_steps {
            let total_configs: usize = optimizations.iter().map(|(config_fn, _)| config_fn(&self) as usize).sum();

            let mut new_beam = BinaryHeap::with_capacity(beam_width);

            while let Some(beam_kernel) = beam.pop() {
                for (config_fn, optimization_fn) in &optimizations {
                    let num_configs = config_fn(&beam_kernel.kernel) as usize;
                    if num_configs == 0 {
                        continue;
                    }

                    // Calculate how many configs to try for this optimization
                    let n_configs_to_try =
                        (num_configs as f32 * config.optimization_passes as f32 / total_configs as f32).ceil() as u16;

                    let kernel = beam_kernel.kernel.clone();
                    let mut config_id = 0;
                    while config_id < n_configs_to_try {
                        optimization_fn(&mut kernel, config_id);
                        let cost = kernel.get_cost();

                        if

                        config_id += 1;
                    }
                }
            }

            // If we didn't get any new kernels, break early
            if new_beam.is_empty() {
                break;
            }

            beam = new_beam;
        }

        // Compile and return the best kernel found
        let BeamEntry { kernel, cost } = beam.pop().unwrap();
        println!("Autotune complete. Best cost: {}, kernel:", cost);
        kernel.debug();
        device.compile(&kernel, false).unwrap()
    }
}
