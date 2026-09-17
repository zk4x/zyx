// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use super::{
    BackendError, DeviceInfo, ErrorStatus, Event, GwsDim, LaunchArg, Pool, PoolBufferId, gws_from_kernel,
};
use crate::{
    DType,
    backend::{DTypeCapability, DeviceProgramId},
    kernel::{Kernel, MemScope, Op, ParamKind, RangeKind},
    shape::Dim,
    slab::Slab,
};
use nanoserde::DeJson;
use pollster::FutureExt;
use std::{
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};
use wgpu::{
    BindGroupLayout, BufferDescriptor, BufferUsages, ComputePipeline, PowerPreference, ShaderModule, SubmissionIndex,
    wgt::PollType,
};

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
pub struct WGPUMemoryPool {
    free_bytes: Dim,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter: wgpu::Adapter,
    buffers: Slab<PoolBufferId, wgpu::Buffer>,
    dev_info: DeviceInfo,
}

/// Process-wide per-device pools. Owned here — `mod.rs` only holds
/// `Pool::WGPU(i)` handles. `WGPU_INIT` serializes first construction only;
/// the alloc/free path never takes it.
static WGPU_POOLS: OnceLock<Vec<Arc<Mutex<WGPUMemoryPool>>>> = OnceLock::new();
static WGPU_INIT: Mutex<()> = Mutex::new(());

fn pools_with(config: &WGPUConfig, debug_dev: bool) -> Result<&'static Vec<Arc<Mutex<WGPUMemoryPool>>>, BackendError> {
    if let Some(pools) = WGPU_POOLS.get() {
        return Ok(pools);
    }
    let _init = WGPU_INIT.lock().unwrap_or_else(|_| panic!("wgpu pool init lock poisoned"));
    if let Some(pools) = WGPU_POOLS.get() {
        return Ok(pools);
    }
    let pools = ensure_pool_table(config, debug_dev)?;
    let _ = WGPU_POOLS.set(pools);
    WGPU_POOLS.get().ok_or_else(|| BackendError {
        status: ErrorStatus::Initialization,
        context: "WGPU pool init failed".into(),
    })
}

fn pools() -> Result<&'static Vec<Arc<Mutex<WGPUMemoryPool>>>, BackendError> {
    pools_with(&WGPUConfig::default(), false)
}

pub(super) fn pool(id: u16) -> Result<Arc<Mutex<WGPUMemoryPool>>, BackendError> {
    pools()?.get(id as usize).cloned().ok_or_else(|| no_pool(id))
}

pub(super) fn pool_count() -> u16 {
    pools().map(|pools| pools.len() as u16).unwrap_or(0)
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
}

#[derive(Debug, Clone)]
pub struct WGPUEvent {
    submission_index: Option<SubmissionIndex>,
}

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

pub(super) fn ensure_pool_table(
    config: &WGPUConfig,
    debug_dev: bool,
) -> Result<Vec<Arc<Mutex<WGPUMemoryPool>>>, BackendError> {
    let mut pools: Vec<Arc<Mutex<WGPUMemoryPool>>> = Vec::new();
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
    pools.push(Arc::new(Mutex::new(WGPUMemoryPool {
        free_bytes: 1_000_000_000,
        device,
        queue,
        adapter: wgpu_adapter,
        buffers: Slab::new(),
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
    })));

    Ok(pools)
}

/// Process-wide per-device WGPU devices. Owned here — `mod.rs` only holds
/// `Dev::WGPU(i)` handles. `WGPU_DEV_INIT` serializes first construction
/// only; compile/launch take the device lock, never the init lock.
static WGPU_DEVICES: OnceLock<Vec<Arc<Mutex<WGPUDevice>>>> = OnceLock::new();
static WGPU_DEV_INIT: Mutex<()> = Mutex::new(());

fn devices_with(config: &WGPUConfig, debug_dev: bool) -> Result<&'static Vec<Arc<Mutex<WGPUDevice>>>, BackendError> {
    if let Some(devs) = WGPU_DEVICES.get() {
        return Ok(devs);
    }
    let _init = WGPU_DEV_INIT.lock().unwrap_or_else(|_| panic!("wgpu device init lock poisoned"));
    if let Some(devs) = WGPU_DEVICES.get() {
        return Ok(devs);
    }
    let devs = ensure_device_table(config, debug_dev)?;
    let _ = WGPU_DEVICES.set(devs);
    WGPU_DEVICES.get().ok_or_else(|| BackendError {
        status: ErrorStatus::Initialization,
        context: "WGPU device init failed".into(),
    })
}

fn devices() -> Result<&'static Vec<Arc<Mutex<WGPUDevice>>>, BackendError> {
    devices_with(&super::load_config().wgpu, super::debug_backends())
}

pub(super) fn device(id: u16) -> Result<Arc<Mutex<WGPUDevice>>, BackendError> {
    devices()?.get(id as usize).cloned().ok_or_else(|| BackendError {
        status: ErrorStatus::Initialization,
        context: format!("Dev::WGPU({id}) is not available").into(),
    })
}

pub(super) fn device_count() -> u16 {
    devices().map(|devs| devs.len() as u16).unwrap_or(0)
}

fn ensure_device_table(
    config: &WGPUConfig,
    debug_dev: bool,
) -> Result<Vec<Arc<Mutex<WGPUDevice>>>, BackendError> {
    let pools = pools_with(config, debug_dev)?;
    let mut devs = Vec::with_capacity(pools.len());
    for (idx, pool_arc) in pools.iter().enumerate() {
        let pool_id = Pool::WGPU(u16::try_from(idx).expect("So many WGPU devices..."));
        let guard = super::lock(pool_id, pool_arc);
        devs.push(Arc::new(Mutex::new(WGPUDevice {
            dev_info: Arc::new(guard.dev_info.clone()),
            memory_pool: pool_id,
            device: guard.device.clone(),
            adapter: guard.adapter.clone(),
            programs: Slab::new(),
            queue: guard.queue.clone(),
        })));
    }
    Ok(devs)
}

impl WGPUMemoryPool {
    #[allow(clippy::unused_self)]
    pub const fn deinitialize(&mut self) {}

    pub const fn free_bytes(&self) -> Dim {
        self.free_bytes
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<(PoolBufferId, Event), BackendError> {
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
        let id = self.buffers.push(buffer);
        let event = Event::WGPU(WGPUEvent { submission_index: None });
        Ok((id, event))
    }

    pub fn deallocate(&mut self, buffer_id: PoolBufferId, event_wait_list: Vec<Event>) {
        drop(event_wait_list);
        let buffer = unsafe { self.buffers.remove_and_return(buffer_id) };
        buffer.destroy();
    }

    #[allow(clippy::unnecessary_wraps)]
    pub fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: PoolBufferId,
        event_wait_list: Vec<Event>,
    ) -> Result<super::Event, BackendError> {
        // wgpu requires writes to be multiples of 4 bytes
        const ALIGN: usize = wgpu::COPY_BUFFER_ALIGNMENT as usize;
        drop(event_wait_list);

        let dst = &self.buffers[dst];

        //let aligned_len = (src.len() + ALIGN - 1) / ALIGN * ALIGN;
        let aligned_len = src.len().div_ceil(ALIGN);

        // Use write_buffer for the aligned portion
        if aligned_len > src.len() {
            // If src.len() is not divisible by 4, we need a tiny slice with padding
            // Here we can safely use `write_buffer` with padding without allocating a new Vec
            // by creating a small stack buffer for the extra bytes
            let mut padded: [u8; ALIGN] = [0; ALIGN];
            let full_chunks = src.len() / ALIGN;
            let remaining = src.len() % ALIGN;

            // Write full 4-byte chunks directly
            if full_chunks > 0 {
                self.queue.write_buffer(dst, 0, &src[..full_chunks * ALIGN]);
            }

            // Write the remaining bytes padded with zeros
            if remaining > 0 {
                padded[..remaining].copy_from_slice(&src[full_chunks * ALIGN..]);
                self.queue.write_buffer(dst, (full_chunks * ALIGN) as u64, &padded);
            }
        } else {
            // Already aligned
            self.queue.write_buffer(dst, 0, src);
        }

        let encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("GpuBuffer::write") });
        self.queue.submit(Some(encoder.finish()));

        Ok(Event::WGPU(WGPUEvent { submission_index: None }))
    }

    /*pub fn pool_to_host(
        &mut self,
        src: PoolBufferId,
        dst: &mut [u8],
        event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError> {
        let _ = event_wait_list;
        let src = &self.buffers[src];
        async {
            let (tx, rx) = futures::channel::oneshot::channel();
            DownloadBuffer::read_buffer(&self.device, &self.queue, &src.slice(..), move |result| {
                tx.send(result).unwrap_or_else(|_| panic!("Failed to download buffer."));
            });
            self.device.poll(PollType::Wait { submission_index: None, timeout: None }).unwrap();
            let download = rx.await.unwrap().unwrap();
            dst.copy_from_slice(&download);
        }
        .block_on();
        Ok(())
    }*/

    #[allow(clippy::unnecessary_box_returns)]
    #[allow(clippy::unnecessary_wraps)]
    pub fn pool_to_host(&mut self, src: PoolBufferId, dst: &mut [u8], event_wait_list: Vec<Event>) -> Result<(), BackendError> {
        drop(event_wait_list); // You can eventually use events if needed

        // Get the source buffer
        let src = &self.buffers[src];

        // Create a temporary download buffer to receive data from the GPU
        let download_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DownloadBuffer"), // You can try removing or adjusting the label if needed
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

    #[allow(clippy::unnecessary_box_returns)]
    #[allow(clippy::unnecessary_wraps)]
    pub fn sync_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        for event in events {
            if let Event::WGPU(event) = event {
                _ = self
                    .device
                    .poll(PollType::Wait { submission_index: event.submission_index, timeout: Some(Duration::from_secs(300)) });
            }
        }
        Ok(())
    }

    pub fn pool_to_pool(
        &mut self,
        src: Pool,
        src_buf: PoolBufferId,
        dst_buf: PoolBufferId,
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        match src {
            Pool::Host => {
                let src_pool = super::host::pool();
                let src_pool = super::lock(src, &src_pool);
                self.host_to_pool(src_pool.get_buffer(src_buf), dst_buf, event_wait_list)
            }
            _ => todo!("pool_to_pool from {src:?} to WGPU"),
        }
    }

    pub fn release_events(&mut self, events: Vec<Event>) {
        drop(events);
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

    pub const fn free_compute(&self) -> u128 {
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

    #[allow(clippy::unnecessary_wraps)]
    pub fn launch(
        &mut self,
        program_id: DeviceProgramId,
        pool_handle: Pool,
        args: &[LaunchArg],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        debug_assert_eq!(pool_handle, self.memory_pool);
        drop(event_wait_list);
        let Pool::WGPU(id) = pool_handle else { unreachable!("WGPU launch with non-WGPU pool") };
        let pool_arc = pool(id).expect("launch on unavailable WGPU pool");
        let memory_pool = super::lock(pool_handle, &pool_arc);
        let program = &self.programs[program_id];
        let binds: Vec<wgpu::BindGroupEntry> = args
            .iter()
            .enumerate()
            .filter_map(|(bind_id, arg)| {
                let LaunchArg::Buffer(buffer_id) = arg else { return None };
                let buffer = &memory_pool.buffers[*buffer_id];
                Some(wgpu::BindGroupEntry { binding: u32::try_from(bind_id).unwrap(), resource: buffer.as_entire_binding() })
            })
            .collect();

        let set = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &program.bind_group_layout,
            entries: &binds,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Kernel::enqueue") });
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
        let submission_index = Some(self.queue.submit(Some(encoder.finish())));
        Ok(Event::WGPU(WGPUEvent { submission_index }))
    }
}
