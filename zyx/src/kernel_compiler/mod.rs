use crate::{
    DebugMask, Map,
    backend::{BufferId, Device, DeviceInfo, Event, ProgramId},
    error::BackendError,
    graph::kernel::{Kernel, Op},
    prog_bar::ProgressBar,
    runtime::Pool,
};
use std::hash::BuildHasherDefault;

mod ir;
mod optimizer;

use ir::lower_to_ir;
pub use ir::{IRKernel, IROp, IRScope};
use optimizer::{Optimization, Optimizer};

#[derive(Debug)]
pub struct KernelCompiler {
    device_infos: Map<DeviceInfo, u32>,
    kernels: Map<Vec<Op>, u32>,
    // Finished optimizations of kernels for given devices
    // kernel id, device info id => optimization
    optimizations: Map<(u32, u32), Optimization>,
    // This last one is not stored to disk
    // kernel id, device id => program id
    programs: Map<(u32, u32), ProgramId>,
}

impl KernelCompiler {
    pub const fn new() -> KernelCompiler {
        KernelCompiler {
            device_infos: Map::with_hasher(BuildHasherDefault::new()),
            kernels: Map::with_hasher(BuildHasherDefault::new()),
            optimizations: Map::with_hasher(BuildHasherDefault::new()),
            programs: Map::with_hasher(BuildHasherDefault::new()),
        }
    }

    pub(super) fn deinitialize(&mut self, devices: &mut [Device]) {
        for (&(_, device_id), program_id) in &mut self.programs {
            devices[device_id as usize].release(*program_id);
        }
        self.device_infos = Map::with_hasher(BuildHasherDefault::new());
        self.kernels = Map::with_hasher(BuildHasherDefault::new());
        self.optimizations = Map::with_hasher(BuildHasherDefault::new());
    }

    // If the kernel is cached, then launches kernel, otherwise if search_iters is zero,
    // compiles kernel with default optimizations and launches it, otherwise
    // searches over search_iters iterations, compiling and running each optimization
    // and saves the best optimization. The kernel is run at most search_iter.min(1) times.
    // Kernel is optimized on original data, so all buffers must be read only or write only.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch(
        &mut self,
        kernel: &Kernel,
        device_id: u32,
        device: &mut Device,
        pool: &mut Pool,
        args: &[BufferId],
        event_wait_list: Vec<Event>,
        search_iters: usize,
        debug: DebugMask,
    ) -> Result<Option<Event>, BackendError> {
        let dev_info_id = if let Some(&dev_info_id) = self.device_infos.get(device.info()) {
            dev_info_id
        } else {
            let dev_info_id =
                self.device_infos.values().max().map_or(0, |id| id.checked_add(1).unwrap());
            assert!(self.device_infos.insert(device.info().clone(), dev_info_id).is_none());
            dev_info_id
        };

        // Launch if it is in cache
        let kernel_id = if let Some(&kernel_id) = self.kernels.get(&kernel.ops) {
            // If it has been compiled for the device
            if let Some(&program_id) = self.programs.get(&(kernel_id, dev_info_id)) {
                let event = device.launch(program_id, &mut pool.pool, args, event_wait_list)?;
                return Ok(Some(event));
            // If we know the best optimization, but it has not been compiled yet
            } else if let Some(optimization) = self.optimizations.get(&(kernel_id, dev_info_id)) {
                let ir_kernel = lower_to_ir(kernel.ops.clone(), optimization);
                let program_id = device.compile(&ir_kernel, debug.asm())?;
                let event = device.launch(program_id, &mut pool.pool, args, event_wait_list)?;
                assert!(self.programs.insert((kernel_id, device_id), program_id).is_none());
                return Ok(Some(event));
            }
            kernel_id
        } else {
            let kernel_id =
                self.kernels.values().copied().max().unwrap_or(0).checked_add(1).unwrap();
            assert!(self.kernels.insert(kernel.ops.clone(), kernel_id).is_none());
            kernel_id
        };

        if debug.sched() {
            kernel.debug();
        }

        let (flop, mem_read, mem_write) = kernel.flop_mem_rw();
        // If the kernel has not be optimized yet, optimize it
        if search_iters == 0 {
            // if optimizations are not requested, use default optimizations
            let optimization = Optimization::new(kernel, device.info());
            let ir_kernel = lower_to_ir(kernel.ops.clone(), &optimization);
            if debug.ir() {
                ir_kernel.debug();
            }
            let program_id = device.compile(&ir_kernel, debug.asm())?;
            let nanos = std::time::Instant::now();
            let event = device.launch(program_id, &mut pool.pool, args, event_wait_list)?;
            pool.pool.sync_events(vec![event])?;
            let nanos = nanos.elapsed().as_nanos();
            assert!(self.programs.insert((kernel_id, device_id), program_id).is_none());
            if debug.perf() {
                println!("{}", get_perf(flop, mem_read, mem_write, nanos));
            }
            //self.optimizations.insert((kernel_id, dev_info_id), optimization);
        } else {
            // If number of search iterations is more than zero, then do the full optimization pass
            // with multiple iterations
            let rng = crate::rng::Rng::seed_from_u64(3_940_239);
            let mut optimizer = Optimizer::new(rng, kernel, device.info().clone());
            pool.pool.sync_events(event_wait_list)?;

            // Run the default optimization
            let optimization = optimizer.best_node.clone();
            let _ = optimizer.bench_optimization(&optimization, pool, device, args, debug)?;

            let mut progress_bar = if debug.perf() {
                Some(ProgressBar::new(search_iters as u64))
            } else {
                None
            };

            'a: for _ in 0..search_iters {
                let Some(optimization) = optimizer.next() else { break };
                let Ok(nanos) =
                    optimizer.bench_optimization(&optimization, pool, device, args, debug)
                else {
                    continue 'a;
                };
                if let Some(bar) = &mut progress_bar {
                    bar.inc(1, &get_perf(flop, mem_read, mem_write, nanos));
                }
            }

            self.optimizations.insert((kernel_id, dev_info_id), optimizer.best_node);
        }
        Ok(None)
    }
}

#[allow(clippy::similar_names)]
fn get_perf(flop: u128, bytes_read: u128, bytes_written: u128, nanos: u128) -> String {
    const fn value_unit(x: u128) -> (u128, &'static str) {
        match x {
            0..1000 => (x * 100, ""),
            1_000..1_000_000 => (x / 10, "k"),
            1_000_000..1_000_000_000 => (x / 10_000, "M"),
            1_000_000_000..1_000_000_000_000 => (x / 10_000_000, "G"),
            1_000_000_000_000..1_000_000_000_000_000 => (x / 10_000_000_000, "T"),
            1_000_000_000_000_000..1_000_000_000_000_000_000 => (x / 10_000_000_000_000, "P"),
            1_000_000_000_000_000_000.. => (x / 10_000_000_000_000_000, "E"),
        }
    }

    let (f, f_u) = value_unit(flop);
    let (br, br_u) = value_unit(bytes_read);
    let (bw, bw_u) = value_unit(bytes_written);
    let (t, t_u) = match nanos {
        0..1_000 => (nanos * 10, "ns"),
        1_000..1_000_000 => (nanos / 100, "μs"),
        1_000_000..1_000_000_000 => (nanos / 100_000, "ms"),
        1_000_000_000..1_000_000_000_000 => (nanos / 100_000_000, "s"),
        1_000_000_000_000.. => (nanos / 6_000_000_000, "min"),
    };

    let (fs, f_us) = value_unit(flop * 1_000_000_000 / nanos);
    let (brs, br_us) = value_unit(bytes_read * 1_000_000_000 / nanos);
    let (bws, bw_us) = value_unit(bytes_written * 1_000_000_000 / nanos);

    format!(
        "{}.{} {t_u} ~ {}.{:02} {f_us}FLOP/s, {}.{:02} {br_us}B/s r, {}.{:02} {bw_us}B/s w, {}.{:02} {f_u}FLOP, {}.{:02} {br_u}B r, {}.{:02} {bw_u}B w",
        t / 10,
        t % 10,
        fs / 100,
        fs % 100,
        brs / 100,
        brs % 100,
        bws / 100,
        bws % 100,
        f / 100,
        f % 100,
        br / 100,
        br % 100,
        bw / 100,
        bw % 100,
    )
}
