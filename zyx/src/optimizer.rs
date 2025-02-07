use crate::{
    backend::{BackendError, Device, DeviceInfo, Event, MemoryPool}, ir::IRKernel, kernel::{Kernel, Op}, runtime::Pool, shape::Dimension, slab::Id, DebugMask
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Optimization {
    pub local_work_size: [Dimension; 3],
    // upcast op id -> Dimension, where op is loop split from register to global
    pub upcast: Option<(u16, Dimension)>,
}

#[derive(Debug)]
pub struct Optimizer {
    device_infos: BTreeMap<DeviceInfo, u32>,
    kernels: BTreeMap<Vec<Op>, u32>,
    // Finished optimizations of kernels for given devices
    // kernel id, device info id => optimization
    optimizations: BTreeMap<(u32, Id), Optimization>,
    // This last one is not stored to disk
    // kernel id, device id => program id
    programs: BTreeMap<(u32, u32), Id>,
}

/*#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Optimization {
    splits: Vec<(usize, Vec<Dimension>)>,
    local_tiles: bool,
}*/

impl Optimizer {
    pub(super) const fn new() -> Optimizer {
        Optimizer {
            device_infos: BTreeMap::new(),
            kernels: BTreeMap::new(),
            optimizations: BTreeMap::new(),
            programs: BTreeMap::new(),
        }
    }

    pub(super) fn deinitialize(&mut self, devices: &mut [Box<dyn Device>]) {
        while let Some(((_, device_id), program_id)) = self.programs.pop_last() {
            let _ = devices[device_id as usize].release(program_id);
        }
        self.device_infos = BTreeMap::new();
        self.kernels = BTreeMap::new();
        self.optimizations = BTreeMap::new();
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
        device: &mut dyn Device,
        pool: &mut Pool,
        args: &[Id],
        outputs: BTreeSet<Id>,
        event_wait_list: Vec<Event>,
        search_iters: usize,
        debug: DebugMask,
    ) -> Result<(), BackendError> {
        //println!("Launch kernel with args {args:?}");
        //let t = crate::Timer::new("optimizer");
        // TODO if optimizer is not initialized yet, then first load from disk.

        //println!("Looking for kernel:");
        //kernel.debug();

        let dev_info_id = if let Some(&dev_info_id) = self.device_infos.get(device.info()) {
            dev_info_id
        } else {
            let dev_info_id = self.device_infos.last_key_value().map_or(0, |(_, x)| x.checked_add(1).unwrap());
            assert!(self.device_infos.insert(device.info().clone(), dev_info_id).is_none());
            dev_info_id
        };
        if let Some(&kernel_id) = self.kernels.get(&kernel.ops) {
            // if kernel was already optimized
            if let Some(&program_id) = self.programs.get(&(kernel_id, dev_info_id)) {
                // if it was compiled for the given device
                //println!("Launch cached, program id: {program_id}");
                //kernel.debug();
                let event = device.launch(program_id, pool.pool.as_mut(), args, event_wait_list)?;
                //pool.pool.sync_events(vec![event]).unwrap();
                pool.events.insert(outputs, event);
            } else if let Some(optimization) = self.optimizations.get(&(kernel_id, dev_info_id)) {
                let _ = optimization;
                todo!()
                // if it was optimized for similar device, but not compiled for the given device,
                // or if it was in disk cache.
                /*let ir_kernel = IRKernel::new(kernel.clone(), optimization, debug);
                let program_id = device.compile(&ir_kernel, debug.asm())?;
                let event = device.launch(program_id, pool.pool.as_mut(), args, event_wait_list)?;
                pool.events.insert(outputs, event);
                self.programs.insert((kernel_id, dev_info_id), program_id);*/
            } else {
                unreachable!();
            }
        } else {
            // if kernel was not optimized yet
            let kernel_id = self.kernels.values().copied().max().unwrap_or(0).checked_add(1).unwrap();
            //println!("Kernel ids: {:?}", self.kernels.keys());
            //println!("Program ids: {:?}", self.programs.keys());
            assert!(self.kernels.insert(kernel.ops.clone(), kernel_id).is_none());
            if search_iters == 0 {
                // if optimizations are not requested, use default optimizations
                let optimization = Optimizer::default_optimizations(kernel, device.info());
                let (flop, mem_read, mem_write) = kernel.flop_mem_rw();
                let ir_kernel = IRKernel::new(kernel.clone(), &optimization, debug);
                let program_id = device.compile(&ir_kernel, debug.asm())?;

                //println!("Default optimizations, program id: {program_id}");
                //kernel.debug();

                let nanos = std::time::Instant::now();
                let event = device.launch(program_id, pool.pool.as_mut(), args, event_wait_list)?;
                pool.pool.sync_events(vec![event])?;
                let nanos = nanos.elapsed().as_nanos();
                if debug.perf() {
                    print_perf(flop, mem_read, mem_write, nanos);
                }
                //pool.events.insert(outputs, event);
                assert!(self.programs.insert((kernel_id, dev_info_id), program_id).is_none());
            } else {
                pool.pool.sync_events(event_wait_list).unwrap();
                let (optimization, program_id) = optimize_kernel(
                    kernel,
                    device,
                    pool.pool.as_mut(),
                    args,
                    search_iters,
                    debug,
                );
                self.programs.insert((kernel_id, dev_info_id), program_id);
                self.optimizations.insert((kernel_id, dev_info_id), optimization);
            }
        }
        Ok(())
    }

    fn default_optimizations(kernel: &Kernel, dev_info: &DeviceInfo) -> Optimization {
        let mlws = dev_info.max_local_threads;
        let mlwd = dev_info.max_local_work_dims;

        //let mut reshapes = Vec::new();
        let num_loops = kernel.ops.iter().position(|op| !matches!(op, Op::Loop { .. })).unwrap();
        debug_assert_ne!(num_loops, 0);
        let mut gws = [1; 3];
        if num_loops < 3 {
            //reshapes.push((0, dims));
            let mut gws_i = 3 - num_loops;
            for d in &kernel.shape() {
                gws[gws_i] = *d;
                gws_i += 1;
            }
        } else {
            let sh = kernel.shape();
            for (gws_d, d) in gws.iter_mut().zip(sh[sh.len() - 3..].iter()) {
                *gws_d = *d;
            }
            gws[0] = sh[..sh.len() - 2].iter().product();
        }

        //let mrws = dev_info.num_registers;
        //let max_reg_split = 32;

        let lz = (1..=mlws.min(mlwd[2])).rev().filter(|lz| gws[2] % lz == 0).max().unwrap_or(1);
        let ly =
            (1..=(mlws / lz).min(mlwd[1])).rev().filter(|ly| gws[1] % ly == 0).max().unwrap_or(1);
        let lx = (1..=(mlws / (lz * ly)).min(mlwd[0]))
            .rev()
            .filter(|lx| gws[0] % lx == 0)
            .max()
            .unwrap_or(1);

        //println!("gws = {gws:?}, lws = [{lx}, {ly}, {lz}]");

        // Upcast is only possible if last local dimension (lz) is 1

        Optimization { local_work_size: [lx, ly, lz], upcast: Some((6, 32)) }
    }
}

// Optimize kernel further, search_iters times
fn optimize_kernel(
    kernel: &Kernel,
    device: &dyn Device,
    memory_pool: &dyn MemoryPool,
    args: &[Id],
    search_iters: usize,
    debug: DebugMask,
) -> (Optimization, u32) {
    let _ = kernel;
    let _ = device;
    let _ = device;
    let _ = memory_pool;
    let _ = args;
    let _ = search_iters;
    let _ = debug;

    // First ensure exactly 3 global work dimensions
    //let mgwd = dev_info.max_global_work_dims;
    /*let dev_info = device.info();
    let mlws = dev_info.max_local_threads;
    let mut mlwd = dev_info.max_local_work_dims;

    let mut reshapes = Vec::new();
    let num_loops = kernel.ops.iter().position(|op| !matches!(op, Op::Loop { .. })).unwrap();
    debug_assert_ne!(num_loops, 0);
    let mut gws = [1; 3];
    if num_loops < 3 {
        let dims: Vec<usize> =
            core::iter::repeat(1).take(3 - num_loops).chain([kernel.shape()[0]]).collect();
        reshapes.push((0, dims));
        let mut gws_i = 3 - num_loops;
        for d in &kernel.shape() {
            gws[gws_i] = *d;
            gws_i += 1;
        }
    } else {
        let sh = kernel.shape();
        for (gws_d, d) in gws.iter_mut().zip(sh[sh.len() - 3..].iter()) {
            *gws_d = *d;
        }
        gws[0] = sh[..sh.len() - 2].iter().product();
    }

    let mrws = dev_info.num_registers;
    let max_reg_split = 32;

    // Then find the best local work sizes (later including local tiles)

    // Then find the best register work sizes (loop splitting and unrolling)

    // Then possibly apply other optimizations

    //OptimizerProgress::Optimizing { best: Optimization { splits: Vec::new() }, done: BTreeMap::new(), }
    // list untried optimizations
    /*let mut opts = kernel.available_optimizations(device.info());
    debug_assert!(!opts.is_empty());
    let mut best_exec_time = Duration::MAX;
    // pick an optimization
    for _ in 0..search_iters.min(opts.len()) {
        if let Some(optimization) = opts.pop() {
            //println!("{optimization:?}");
            let ir_kernel = IRKernel::new(&kernel.ops, &optimization, debug.ir());
            let Ok(program_id) = device.compile(&ir_kernel, debug.asm()) else {
                done.insert(optimization, Duration::MAX);
                continue;
            };
            // Launch kernel and measure it's performance
            let begin = std::time::Instant::now();
            let Ok(_) = device.launch(program_id, memory_pool, args, Vec::new(), true) else {
                done.insert(optimization, Duration::MAX);
                continue;
            };
            let exec_time = begin.elapsed();
            done.insert(optimization, exec_time);
            if exec_time < best_exec_time {
                best_exec_time = exec_time;
            } else {
                let _ = device.release(program_id);
            }
        }
    }
    (
        done.iter().min_by_key(|x| x.1).unwrap().0.clone(),
        opts.is_empty(),
    )*/*/
    todo!()
}

#[allow(clippy::similar_names)]
fn print_perf(flop: u128, bytes_read: u128, bytes_written: u128, nanos: u128) {
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
        0..1_000 => (nanos*10, "ns"),
        1_000..1_000_000 => (nanos/100, "μs"),
        1_000_000..1_000_000_000 => (nanos/100_000, "ms"),
        1_000_000_000..1_000_000_000_000 => (nanos/100_000_000, "s"),
        1_000_000_000_000.. => (nanos/6_000_000_000, "min"),
    };

    let (fs, f_us) = value_unit(flop * 1_000_000_000 / nanos);
    let (brs, br_us) = value_unit(bytes_read * 1_000_000_000 / nanos);
    let (bws, bw_us) = value_unit(bytes_written * 1_000_000_000 / nanos);

    println!("        {}.{} {t_u} ~ {}.{} {f_us}FLOP/s, {}.{} {br_us}B/s read, {}.{} {bw_us}B/s write, {}.{} {f_u}FLOP, {}.{} {br_u}B read, {}.{} {bw_u}B write",
        t/10,
        t%10,
        fs/100,
        fs%100,
        brs/100,
        brs%100,
        bws/100,
        bws%100,
        f/100,
        f%100,
        br/100,
        br%100,
        bw/100,
        bw%100,
    );
}

#[test]
fn red1() {
    let x = crate::Tensor::rand([1024 * 1024], crate::DType::F32).unwrap();
    let y = x.sum([]).unwrap();
    println!("{y}");
}
