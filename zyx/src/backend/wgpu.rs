// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use super::{BackendError, ChunkId, DeviceInfo, ErrorStatus, GwsDim, LaunchArg, Pool, Shard, gws_from_kernel};
use crate::{
    DType, Set,
    backend::{DTypeCapability, DeviceProgramId},
    kernel::{Kernel, Op, ParamKind, RangeKind},
    shape::Dim,
    slab::Slab,
};
use nanoserde::DeJson;
use pollster::FutureExt;
use std::{
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};
use wgpu::{BindGroupLayout, BufferDescriptor, BufferUsages, ComputePipeline, PowerPreference, ShaderModule, wgt::PollType};

#[derive(DeJson, Debug)]
#[nserde(default)]
pub struct WGPUConfig {
    enabled: bool,
}

impl Default for WGPUConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug)]
pub struct WGPUBuffer {
    buffer: wgpu::Buffer,
    bytes: Dim,
}

#[derive(Debug)]
pub struct WGPUMemoryPool {
    free_bytes: Dim,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter: wgpu::Adapter,
    buffers: Slab<ChunkId, WGPUBuffer>,
    free_set: Set<ChunkId>,
    dev_info: DeviceInfo,
}

static WGPU_POOLS: OnceLock<Vec<Mutex<WGPUMemoryPool>>> = OnceLock::new();
static WGPU_DEVICES: OnceLock<Vec<Mutex<WGPUDevice>>> = OnceLock::new();

/// Single backend initializer: builds pools + devices together in one pass,
/// publishes both tables. Reads config directly; no init locks.
fn backend() -> Result<(&'static Vec<Mutex<WGPUMemoryPool>>, &'static Vec<Mutex<WGPUDevice>>), BackendError> {
    if let Some(pools) = WGPU_POOLS.get()
        && let Some(devs) = WGPU_DEVICES.get()
    {
        return Ok((pools, devs));
    }
    let config = super::config();
    let debug_dev = super::debug_backends();
    let pools = ensure_pool_table(&config.wgpu, debug_dev)?;
    let mut devs = Vec::with_capacity(pools.len());
    for (idx, pool) in pools.iter().enumerate() {
        let pool_id = Pool::WGPU(u16::try_from(idx).expect("So many WGPU devices..."));
        let guard = super::lock(pool_id, pool);
        devs.push(Mutex::new(WGPUDevice {
            dev_info: Arc::new(guard.dev_info.clone()),
            memory_pool: pool_id,
            device: guard.device.clone(),
            adapter: guard.adapter.clone(),
            programs: Slab::new(),
            queue: guard.queue.clone(),
            pending: Vec::new(),
        }));
        drop(guard);
    }
    let _ = WGPU_POOLS.set(pools);
    let _ = WGPU_DEVICES.set(devs);
    match (WGPU_POOLS.get(), WGPU_DEVICES.get()) {
        (Some(pools), Some(devs)) => Ok((pools, devs)),
        _ => Err(BackendError { status: ErrorStatus::Initialization, context: "WGPU init failed".into() }),
    }
}

pub(super) fn pool(id: u16) -> Result<&'static Mutex<WGPUMemoryPool>, BackendError> {
    backend()?.0.get(id as usize).ok_or_else(|| no_pool(id))
}

pub(super) fn pool_count() -> u16 {
    backend().map(|(pools, _)| pools.len() as u16).unwrap_or(0)
}

fn no_pool(id: u16) -> BackendError {
    BackendError { status: ErrorStatus::Initialization, context: format!("Pool::WGPU({id}) is not available").into() }
}

#[derive(Debug)]
pub struct WGPUDevice {
    dev_info: Arc<DeviceInfo>,
    memory_pool: Pool,
    device: Arc<wgpu::Device>,
    #[allow(unused)]
    adapter: wgpu::Adapter,
    programs: Slab<DeviceProgramId, WGPUProgram>,
    queue: Arc<wgpu::Queue>,
    /// Pending micro-batch window: launches accumulate here in program order
    /// until MICRO_BATCH_WINDOW is reached (or a sync point arrives), then
    /// the whole window is recorded into ONE command encoder and submitted
    /// with a single `queue.submit` (the single in-order queue preserves
    /// ordering — no waits are needed).
    pending: Vec<(DeviceProgramId, Vec<LaunchArg>)>,
}

/// Pending commands accumulate until the micro-batch window is flushed.
const MICRO_BATCH_WINDOW: usize = 100;

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct WGPUProgram {
    name: String,
    arg_ro_flags: Vec<bool>,
    shader: ShaderModule,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    gws: Vec<GwsDim>,
}

pub(super) fn ensure_pool_table(config: &WGPUConfig, debug_dev: bool) -> Result<Vec<Mutex<WGPUMemoryPool>>, BackendError> {
    let mut pools: Vec<Mutex<WGPUMemoryPool>> = Vec::new();
    if !config.enabled {
        if debug_dev {
            println!("[WGPU] configured out");
        }
        return Ok(pools);
    }

    let power_preference = PowerPreference::from_env().unwrap_or(wgpu::PowerPreference::HighPerformance);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds { for_resource_creation: None, for_device_loss: None },
        backend_options: wgpu::BackendOptions::from_env_or_default(),
        display: None,
    });

    if debug_dev {
        println!("[WGPU] requesting device with {power_preference:#?} power preference");
    }

    let (wgpu_adapter, wgpu_device, wgpu_queue, info) = async {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference, ..Default::default() })
            .await
            .expect("Failed at adapter creation.");
        let info = adapter.get_info();
        let mut features = wgpu::Features::empty();
        if adapter.features().contains(wgpu::Features::SHADER_F64) {
            features |= wgpu::Features::SHADER_F64;
        }
        if adapter.features().contains(wgpu::Features::SHADER_INT64) {
            features |= wgpu::Features::SHADER_INT64;
        }
        if adapter.features().contains(wgpu::Features::SHADER_F16) {
            features |= wgpu::Features::SHADER_F16;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: features,
                required_limits: adapter.limits(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed at device creation");
        (adapter, device, queue, info)
    }
    .block_on();

    if debug_dev {
        println!("[WGPU] {} ({}) — {:#?}", info.name, info.device, info.backend);
    }
    let device = Arc::new(wgpu_device);
    let queue = Arc::new(wgpu_queue);
    if debug_dev {
        println!("[WGPU] device total memory: {} MB", 1_000_000_000u64 / (1024 * 1024));
    }
    let limits = device.limits();
    let wgpu_features = wgpu_adapter.features();
    let dtype_capability = {
        let mut ops = [DTypeCapability::all(); DType::N_DTYPES];
        // Vulkan driver produces incorrect f64 results on some AMD GPUs (RADV).
        // Users who need reliable f64 should use the Vulkan backend directly.
        ops[DType::F64 as usize] = DTypeCapability::none();
        if !wgpu_features.contains(wgpu::Features::SHADER_INT64) {
            ops[DType::I64 as usize] = DTypeCapability::none();
            ops[DType::U64 as usize] = DTypeCapability::none();
        }
        if !wgpu_features.contains(wgpu::Features::SHADER_F16) {
            ops[DType::F16 as usize] = DTypeCapability::none();
        }
        ops[DType::BF16 as usize] = DTypeCapability::none();
        // naga validator does not support 8/16-bit integer types at all
        ops[DType::U8 as usize] = DTypeCapability::none();
        ops[DType::I8 as usize] = DTypeCapability::none();
        ops[DType::U16 as usize] = DTypeCapability::none();
        ops[DType::I16 as usize] = DTypeCapability::none();
        ops
    };
    pools.push(Mutex::new(WGPUMemoryPool {
        free_bytes: 1_000_000_000,
        device,
        queue,
        adapter: wgpu_adapter,
        buffers: Slab::new(),
        free_set: Set::default(),
        dev_info: DeviceInfo {
            compute: 1024 * 1024 * 1024 * 1024,
            max_global_work_dims: vec![100_000; 3],
            max_local_threads: limits.max_compute_invocations_per_workgroup,
            max_local_work_dims: vec![
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ],
            preferred_vector_size: 4,
            local_mem_size: 64 * 1024,
            max_register_bytes: 512,
            tensor_cores: false,
            warp_size: 32,
            cc: [0, 0],
            has_native_exp2: true,
            supported_vec_lens: vec![2, 3, 4],
            dtype_capability,
            tenstorrent: false,
            tile: [1, 1],
            tile_sizes: vec![],
            wmma_layouts: vec![],
            num_circular_buffers: 0,
            has_openmp: false,
        },
    }));

    Ok(pools)
}

pub(super) fn device(id: u16) -> Result<&'static Mutex<WGPUDevice>, BackendError> {
    backend()?.1.get(id as usize).ok_or_else(|| BackendError {
        status: ErrorStatus::Initialization,
        context: format!("Dev::WGPU({id}) is not available").into(),
    })
}

pub(super) fn device_count() -> u16 {
    backend().map(|(_, devs)| devs.len() as u16).unwrap_or(0)
}

impl WGPUMemoryPool {
    #[allow(clippy::unused_self)]
    pub const fn deinitialize(&mut self) {}

    pub const fn free_bytes(&self) -> Dim {
        self.free_bytes
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<ChunkId, BackendError> {
        let align = wgpu::COPY_BUFFER_ALIGNMENT as Dim;
        let bytes = (bytes + align - 1) / align * align;
        if bytes > self.free_bytes {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "".into() });
        }
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: bytes as u64,
            usage: BufferUsages::from_bits_truncate(
                BufferUsages::STORAGE.bits() | BufferUsages::COPY_SRC.bits() | BufferUsages::COPY_DST.bits(),
            ),
            mapped_at_creation: false,
        });
        self.free_bytes -= bytes;
        Ok(self.buffers.push(WGPUBuffer { buffer, bytes }))
    }

    /// Put a buffer into the free list for stable-address reuse. Frees
    /// nothing; VRAM is reclaimed only by [`WGPUMemoryPool::dispose`]. The
    /// mod.rs dispatch flushes the pending window before calling this, so no
    /// unsubmitted launch still uses the buffer.
    pub fn release(&mut self, buffer_id: ChunkId) {
        if !self.buffers.contains_id(buffer_id) {
            debug_assert!(false, "release of unknown WGPU buffer {buffer_id:?}");
            return;
        }
        debug_assert!(!self.free_set.contains(&buffer_id), "double release of WGPU buffer {buffer_id:?}");
        self.free_set.insert(buffer_id);
    }

    /// Tries to reuse existing allocations, all or nothing: every id must
    /// be on the free list (stable addresses) or nothing is claimed.
    pub fn try_reuse_allocations(&mut self, buffer_ids: &Set<ChunkId>) -> bool {
        let ok = buffer_ids.iter().all(|id| self.free_set.contains(id));
        if ok {
            for id in buffer_ids {
                self.free_set.remove(id);
            }
        }
        ok
    }

    /// Scratch for queue intermediaries: reuses the smallest fitting free
    /// buffer, else allocates fresh. Never fails on fragmentation.
    pub fn allocate_scratch(&mut self, bytes: Dim) -> Result<ChunkId, BackendError> {
        let best = self
            .free_set
            .iter()
            .filter_map(|id| {
                let len = self.buffers[*id].bytes;
                (len >= bytes).then_some((len, *id))
            })
            .min();
        if let Some((_, id)) = best {
            self.free_set.remove(&id);
            return Ok(id);
        }
        self.allocate(bytes)
    }

    /// Free all buffers in the free list at once: destroy every free buffer
    /// and give the bytes back. wgpu defers driver-level destruction behind
    /// in-flight work itself.
    pub fn dispose(&mut self) {
        for id in core::mem::take(&mut self.free_set) {
            if let Some(buffer) = self.buffers.get(id) {
                self.free_bytes += buffer.bytes;
            }
            if self.buffers.contains_id(id) {
                let entry = unsafe { self.buffers.remove_and_return(id) };
                entry.buffer.destroy();
            }
        }
    }

    /// Blocking read-back (sync point: drains the queue via poll(Wait)).
    #[allow(clippy::unnecessary_box_returns)]
    #[allow(clippy::unnecessary_wraps)]
    pub fn pool_to_host(&mut self, src: ChunkId, dst: &mut [u8]) -> Result<(), BackendError> {
        // Get the source buffer
        let src = &self.buffers[src].buffer;

        // Create a temporary download buffer to receive data from the GPU
        let download_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DownloadBuffer"),
            size: dst.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, // Ensure proper usage flags
            mapped_at_creation: false,
        });

        // Record a command to copy the data from the GPU buffer to the download buffer
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("CopyBufferEncoder") });

        // Copy data from the source buffer to the download buffer
        encoder.copy_buffer_to_buffer(
            src,
            0, // Start at the beginning of the source buffer
            &download_buffer,
            0,                // Start at the beginning of the destination buffer
            dst.len() as u64, // The number of bytes to copy
        );

        // Submit the command to the GPU
        let command_buffer = encoder.finish();
        self.queue.submit(Some(command_buffer));

        // Create a channel to notify when mapping is complete
        let (tx, rx) = std::sync::mpsc::channel();

        // Map the download buffer asynchronously
        download_buffer.map_async(wgpu::MapMode::Read, 0..download_buffer.size(), move |result| {
            // Notify the main thread when the mapping is done
            tx.send(result).unwrap();
        });

        // Poll the device to wait for the buffer mapping to complete
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap(); // Make sure polling completes

        // Wait for the map operation to complete
        let mapping_result = rx.recv().unwrap();
        mapping_result.unwrap(); // Ensure the mapping was successful

        // Now that the buffer is mapped, access the mapped data (entire buffer)
        let mapped_range = download_buffer.get_mapped_range(0..download_buffer.size());

        // Copy the data to the destination
        dst.copy_from_slice(&mapped_range);

        // Unmap the buffer after use. Make sure to drop the mapped view before unmapping.
        drop(mapped_range); // This drops the mapped range to release the view before unmapping the buffer.
        download_buffer.unmap();

        Ok(())
    }
}

impl WGPUDevice {
    #[allow(clippy::unused_self)]
    pub const fn deinitialize(&mut self) {}

    pub fn info(&self) -> Arc<DeviceInfo> {
        self.dev_info.clone()
    }

    pub const fn memory_pool(&self) -> Pool {
        self.memory_pool
    }

    pub fn free_compute(&self) -> u128 {
        self.dev_info.compute
    }

    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        let mut lws = [1u64; 3];
        let mut op_id = kernel.head;
        let mut steps_op_id = 0usize;
        while !op_id.is_null() {
            steps_op_id += 1;
            if steps_op_id > 10_000 {
                panic!("compile did not finish in 10000 steps");
            }
            if let Op::Range { axis, kind: scope } = kernel.ops[op_id].op {
                match scope {
                    RangeKind::Group(_) => {}
                    RangeKind::Local(len) => lws[axis as usize] = u64::from(len),
                    // A warp is a view over a local range — adds no threads.
                    RangeKind::Warp(_) => {}
                }
            }
            op_id = kernel.next_op(op_id);
        }

        let spirv_words = kernel.generate_spirv(debug_asm)?;

        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::SpirV(std::borrow::Cow::Owned(spirv_words)),
        });

        if lws.iter().product::<u64>() > u64::from(self.dev_info.max_local_threads) {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "Invalid local work size.".into() });
        }

        let name = format!("k_lws_{}", lws.iter().map(ToString::to_string).collect::<Vec<_>>().join("_"),);

        // Read only flags
        let mut arg_ro_flags = Vec::new();
        let mut op_id = kernel.head;
        let mut steps_op_id = 0usize;
        while !op_id.is_null() {
            steps_op_id += 1;
            if steps_op_id > 10_000 {
                panic!("compile did not finish in 10000 steps");
            }
            if let &Op::Param { kind, .. } = kernel.at(op_id)
                && matches!(kind, ParamKind::Global | ParamKind::GlobalMut)
            {
                arg_ro_flags.push(kind == ParamKind::Global);
            }
            op_id = kernel.next_op(op_id);
        }
        let bg_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = arg_ro_flags
            .iter()
            .enumerate()
            .map(|(bind_id, ro)| wgpu::BindGroupLayoutEntry {
                binding: u32::try_from(bind_id).unwrap(),
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    has_dynamic_offset: false,
                    min_binding_size: None,
                    ty: wgpu::BufferBindingType::Storage { read_only: *ro },
                },
                count: None,
            })
            .collect();

        let bind_group_layout =
            self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &bg_layout_entries });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            module: &shader_module,
            entry_point: Some(&name),
            layout: Some(&pipeline_layout),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let gws = gws_from_kernel(kernel, &self.dev_info.max_global_work_dims)?;
        let id = self.programs.push(WGPUProgram { name, arg_ro_flags, shader: shader_module, pipeline, bind_group_layout, gws });

        Ok(id)
    }

    pub fn release(&mut self, program_id: DeviceProgramId) {
        self.programs.remove(program_id);
    }

    /// Fire-and-forget launch: appended to the micro-batch window; the whole
    /// window is recorded into one command encoder and submitted with a
    /// single `queue.submit` when it flushes.
    #[allow(clippy::unnecessary_wraps)]
    pub fn launch(&mut self, program_id: DeviceProgramId, pool_handle: Pool, args: &[LaunchArg]) -> Result<(), BackendError> {
        debug_assert_eq!(pool_handle, self.memory_pool);
        self.pending.push((program_id, args.to_vec()));
        if self.pending.len() >= MICRO_BATCH_WINDOW {
            self.flush_window()?;
        }
        Ok(())
    }

    /// Timed launch for autotune: the pending window is submitted first, then
    /// the kernel runs solo and the wall-clock nanos are measured around
    /// submit-to-poll(Wait).
    pub fn launch_timed(&mut self, program_id: DeviceProgramId, args: &[LaunchArg]) -> Result<u64, BackendError> {
        self.flush_window()?;
        let start = Instant::now();
        let mut solo = vec![(program_id, args.to_vec())];
        self.record_and_submit(&mut solo);
        self.device
            .poll(PollType::Wait { submission_index: None, timeout: None })
            .map_err(|e| BackendError { status: ErrorStatus::KernelSync, context: format!("wgpu poll: {e:?}").into() })?;
        Ok(start.elapsed().as_nanos() as u64)
    }

    /// Records every pending launch into ONE command encoder (one compute
    /// pass each, program order) and submits with a single `queue.submit`.
    fn flush_window(&mut self) -> Result<(), BackendError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let mut pending = std::mem::take(&mut self.pending);
        self.record_and_submit(&mut pending);
        Ok(())
    }

    /// Locks this device's pool (device → pool, never the reverse) for the
    /// buffer handles, records and submits the window.
    fn record_and_submit(&mut self, pending: &mut Vec<(DeviceProgramId, Vec<LaunchArg>)>) {
        let Pool::WGPU(id) = self.memory_pool else {
            unreachable!("WGPU device with non-WGPU pool")
        };
        let pool_arc = pool(id).expect("flush on unavailable WGPU pool");
        let memory_pool = super::lock(self.memory_pool, &pool_arc);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Kernel::enqueue") });
        for (program_id, args) in pending.drain(..) {
            let program = &self.programs[program_id];
            let binds: Vec<wgpu::BindGroupEntry> = args
                .iter()
                .enumerate()
                .filter_map(|(bind_id, arg)| {
                    let LaunchArg::Buffer(placement) = arg else { return None };
                    let [Shard { chunk, .. }] = placement.shards.as_slice() else {
                        todo!("multi-shard placement in WGPU launch")
                    };
                    let buffer = &memory_pool.buffers[*chunk].buffer;
                    Some(wgpu::BindGroupEntry { binding: u32::try_from(bind_id).unwrap(), resource: buffer.as_entire_binding() })
                })
                .collect();

            let set = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &program.bind_group_layout,
                entries: &binds,
            });
            {
                let mut cpass = encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Kernel::enqueue"), timestamp_writes: None });
                cpass.set_pipeline(&program.pipeline);
                cpass.set_bind_group(0, &set, &[]);
                cpass.insert_debug_marker(&program.name);
                let default_gws = GwsDim::Const(1);
                let grid = |gdim: &GwsDim| -> u32 {
                    gdim.eval(&mut |ordinal| match &args[ordinal] {
                        LaunchArg::Variable(c) => c.as_dim().unwrap(),
                        LaunchArg::Buffer(_) => unreachable!("gws param must be a Variable launch arg"),
                    })
                    .try_into()
                    .unwrap()
                };
                cpass.dispatch_workgroups(
                    grid(program.gws.first().unwrap_or(&default_gws)),
                    grid(program.gws.get(1).unwrap_or(&default_gws)),
                    grid(program.gws.get(2).unwrap_or(&default_gws)),
                );
            }
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

/// Flushes the device's pending micro-batch window. Called by the `mod.rs`
/// dispatch before every WGPU sync point (pool_to_host / release), so pool
/// operations never race unsubmitted launches.
pub(super) fn flush_pending(id: u16) -> Result<(), BackendError> {
    let dev = device(id)?;
    let mut dev = dev.lock().unwrap_or_else(|_| panic!("WGPU device lock poisoned"));
    dev.flush_window()
}
