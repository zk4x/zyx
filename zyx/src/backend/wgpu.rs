// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use super::{BackendError, Device, DeviceId, DeviceInfo, ErrorStatus, Event, MemoryPool, PoolId};
use crate::{
    DType, Map,
    backend::{DeviceProgramId, PoolBufferId},
    dtype::Constant,
    kernel::{BOp, IDX_T, Kernel, Op, OpId, Scope, UOp},
    runtime::Pool,
    shape::Dim,
    slab::Slab,
};
use nanoserde::DeJson;
use pollster::FutureExt;
use std::{fmt::Write, hash::BuildHasherDefault, sync::Arc, time::Duration};
use wgpu::{
    BufferDescriptor, BufferUsages, PowerPreference, ShaderModule, ShaderModuleDescriptor, ShaderSource, SubmissionIndex,
    wgt::PollType,
};

#[derive(DeJson, Debug)]
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
    buffers: Slab<PoolBufferId, wgpu::Buffer>,
}

#[derive(Debug)]
pub struct WGPUDevice {
    dev_info: DeviceInfo,
    memory_pool_id: PoolId,
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
pub(super) struct WGPUProgram {
    name: String,
    gws: Vec<u64>,
    //local_work_size: [usize; 3],
    arg_ro_flags: Vec<bool>,
    shader: ShaderModule,
}

pub(super) fn initialize_device(
    config: &WGPUConfig,
    memory_pools: &mut Slab<PoolId, Pool>,
    devices: &mut Slab<DeviceId, Device>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    if !config.enabled {
        return Err(BackendError { status: super::ErrorStatus::Initialization, context: "WGPU configured out.".into() });
    }

    let power_preference = PowerPreference::from_env().unwrap_or(wgpu::PowerPreference::HighPerformance);
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), ..Default::default() });

    if debug_dev {
        println!("WGPU Requesting device with {power_preference:#?} power preference");
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
                required_limits: wgpu::Limits { max_storage_buffers_per_shader_stage: 8, ..Default::default() },
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
        println!("Using {} ({}) - {:#?}.", info.name, info.device, info.backend);
    }
    let device = Arc::new(wgpu_device);
    let queue = Arc::new(wgpu_queue);
    let pool = MemoryPool::WGPU(WGPUMemoryPool {
        free_bytes: 1_000_000_000,
        device: device.clone(),
        queue: queue.clone(),
        buffers: Slab::new(),
    });
    memory_pools.push(Pool::new(pool));
    let limits = device.limits();
    devices.push(Device::WGPU(WGPUDevice {
        device,
        adapter: wgpu_adapter,
        dev_info: DeviceInfo {
            compute: 1024 * 1024 * 1024 * 1024,
            max_global_work_dims: vec![100_000; 3],
            max_local_threads: Dim::from(limits.max_compute_invocations_per_workgroup),
            max_local_work_dims: vec![
                Dim::from(limits.max_compute_workgroup_size_x),
                Dim::from(limits.max_compute_workgroup_size_y),
                Dim::from(limits.max_compute_workgroup_size_z),
            ],
            preferred_vector_size: 4,
            local_mem_size: 64 * 1024,
            max_register_bytes: 1024,
            tensor_cores: false,
            warp_size: 32,
        },
        memory_pool_id: PoolId::from(usize::from(memory_pools.len()) - 1),
        programs: Slab::new(),
        queue,
    }));

    Ok(())
}

impl WGPUMemoryPool {
    #[allow(clippy::unused_self)]
    pub const fn deinitialize(&mut self) {}

    pub const fn free_bytes(&self) -> Dim {
        self.free_bytes
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<(PoolBufferId, Event), BackendError> {
        const ALIGN: Dim = wgpu::COPY_BUFFER_ALIGNMENT;
        //let bytes = (bytes + ALIGN - 1) / ALIGN * ALIGN;
        let bytes = bytes.div_ceil(ALIGN);
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

        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("GpuBuffer::write") });
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("CopyBufferEncoder") });

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
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .unwrap(); // Make sure polling completes

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
                    .poll(PollType::Wait { submission_index: event.submission_index, timeout: Some(Duration::from_mins(5)) });
            }
        }
        Ok(())
    }

    #[allow(clippy::unused_self)]
    pub fn release_events(&mut self, events: Vec<Event>) {
        drop(events);
    }
}

impl WGPUDevice {
    #[allow(clippy::unused_self)]
    pub const fn deinitialize(&mut self) {}

    pub const fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    pub const fn memory_pool_id(&self) -> PoolId {
        self.memory_pool_id
    }

    pub const fn free_compute(&self) -> u128 {
        self.dev_info.compute
    }

    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        let mut gws: Vec<u64> = Vec::new();
        let mut lws: Vec<u64> = Vec::new();
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            if let &Op::Index { len, scope, axis: _ } = kernel.at(op_id) {
                match scope {
                    Scope::Global => {
                        gws.push(len);
                    }
                    Scope::Local => {
                        lws.push(len);
                    }
                    Scope::Register => {}
                }
            }
            op_id = kernel.next_op(op_id);
        }

        if lws.iter().product::<u64>() > self.dev_info.max_local_threads as u64 {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "Invalid local work size.".into() });
        }

        let mut arg_ro_flags = Vec::new();
        let mut global_args = String::new();
        let mut workgroup_args = String::new();
        let mut max_p = 0;
        for (op_id, op) in kernel.iter_unordered() {
            if let &Op::Define { dtype, scope, ro, len } = op {
                if scope == Scope::Global {
                    writeln!(
                        global_args,
                        "@group(0) @binding({max_p}) var<storage, {}> p{op_id}: array<{}>;",
                        if ro { "read" } else { "read_write" },
                        dtype.wgsl()
                    )
                    .unwrap();
                    max_p += 1;
                    arg_ro_flags.push(ro);
                }
                if scope == Scope::Local {
                    writeln!(workgroup_args, "var<workgroup> p{op_id}: array<{}, {len}>;", dtype.wgsl()).unwrap();
                }
            }
        }

        let mut dtypes: Map<OpId, DType> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
        let mut n_global_ids = 0;
        let mut loop_id = 0;
        let mut indent = String::from("  ");
        let mut source = String::with_capacity(1000);

        let mut op_id = kernel.head;
        while !op_id.is_null() {
            //println!("{i} -> {op:?}");
            match kernel.at(op_id) {
                Op::Wmma { .. }
                | Op::ConstView { .. }
                | Op::LoadView { .. }
                | Op::StoreView { .. }
                | Op::Move { .. }
                | Op::Reduce { .. } => {
                    unreachable!()
                }
                &Op::Const(x) => {
                    dtypes.insert(op_id, x.dtype());
                    writeln!(source, "{indent}const r{op_id}: {} = {};", x.dtype().wgsl(), x.wgsl()).unwrap();
                }
                &Op::Define { dtype, scope, ro: _, len: _ } => {
                    dtypes.insert(op_id, dtype);
                    match scope {
                        Scope::Register => {
                            writeln!(source, "{indent}var p{op_id}: array<{}, 1>;", dtype.wgsl()).unwrap();
                        }
                        Scope::Local | Scope::Global => {}
                    }
                }
                &Op::Load { src, index, .. } => {
                    dtypes.insert(op_id, dtypes[&src]);
                    writeln!(source, "{indent}let r{op_id} = p{src}[r{index}];").unwrap();
                }
                &Op::Store { dst, x: src, index, vlen: _ } => {
                    writeln!(source, "{indent}p{dst}[r{index}] = r{src};").unwrap();
                }
                &Op::Cast { x, dtype } => {
                    dtypes.insert(op_id, dtype);
                    if dtype == DType::F16 {
                        writeln!(source, "{indent}let r{op_id} = {}(f32(r{x}));", dtype.wgsl()).unwrap();
                    } else {
                        writeln!(source, "{indent}let r{op_id} = {}(r{x});", dtype.wgsl()).unwrap();
                    }
                }
                &Op::Unary { x, uop } => {
                    dtypes.insert(op_id, dtypes[&x]);
                    let dtype = dtypes[&x];
                    match uop {
                        UOp::BitNot => {
                            todo!();
                        }
                        UOp::Neg => writeln!(source, "{indent}let r{op_id} = -r{x};").unwrap(),
                        UOp::Exp2 => {
                            //writeln!(source, "{indent}printf(\"%d\\n\", r{reg});").unwrap();
                            writeln!(source, "{indent}let r{op_id} = exp2(r{x});").unwrap();
                        }
                        UOp::Log2 => writeln!(source, "{indent}let r{op_id} = log2(r{x});").unwrap(),
                        UOp::Reciprocal => {
                            writeln!(source, "{indent}let r{op_id} = {}/r{x};", dtype.one_constant().wgsl()).unwrap();
                        }
                        UOp::Sqrt => writeln!(source, "{indent}let r{op_id} = sqrt(r{x});").unwrap(),
                        UOp::Sin => writeln!(source, "{indent}let r{op_id} = sin(r{x});").unwrap(),
                        UOp::Cos => writeln!(source, "{indent}let r{op_id} = cos(r{x});").unwrap(),
                        UOp::Floor => writeln!(source, "{indent}let r{op_id} = floor(r{x});").unwrap(),
                        UOp::Trunc => writeln!(source, "{indent}let r{op_id} = trunc(r{x});").unwrap(),
                    }
                }
                &Op::Binary { x, y, bop } => {
                    if matches!(bop, BOp::Cmpgt | BOp::Cmplt | BOp::NotEq | BOp::Eq | BOp::And | BOp::Or) {
                        dtypes.insert(op_id, DType::Bool);
                    } else {
                        dtypes.insert(op_id, dtypes[&x]);
                    }
                    match bop {
                        BOp::Add => writeln!(source, "{indent}let r{op_id} = r{x} + r{y};").unwrap(),
                        BOp::Sub => writeln!(source, "{indent}let r{op_id} = r{x} - r{y};").unwrap(),
                        BOp::Mul => writeln!(source, "{indent}let r{op_id} = r{x} * r{y};").unwrap(),
                        BOp::Div => writeln!(source, "{indent}let r{op_id} = r{x} / r{y};").unwrap(),
                        BOp::Pow => writeln!(source, "{indent}let r{op_id} = pow(r{x}, r{y});").unwrap(),
                        BOp::Mod => writeln!(source, "{indent}let r{op_id} = r{x} % r{y};").unwrap(),
                        BOp::Cmplt => writeln!(source, "{indent}let r{op_id} = r{x} < r{y};").unwrap(),
                        BOp::Cmpgt => writeln!(source, "{indent}let r{op_id} = r{x} > r{y};").unwrap(),
                        BOp::Max => writeln!(source, "{indent}let r{op_id} = max(r{x}, r{y});").unwrap(),
                        BOp::Or => writeln!(source, "{indent}let r{op_id} = r{x} || r{y};").unwrap(),
                        BOp::And => writeln!(source, "{indent}let r{op_id} = r{x} && r{y};").unwrap(),
                        BOp::BitXor => writeln!(source, "{indent}let r{op_id} = r{x} ^ r{y};").unwrap(),
                        BOp::BitOr => writeln!(source, "{indent}let r{op_id} = r{x} | r{y};").unwrap(),
                        BOp::BitAnd => writeln!(source, "{indent}let r{op_id} = r{x} & r{y};").unwrap(),
                        BOp::BitShiftLeft => writeln!(source, "{indent}let r{op_id} = r{x} << r{y};").unwrap(),
                        BOp::BitShiftRight => writeln!(source, "{indent}let r{op_id} = r{x} >> r{y};").unwrap(),
                        BOp::NotEq => writeln!(source, "{indent}let r{op_id} = r{x} != r{y};").unwrap(),
                        BOp::Eq => writeln!(source, "{indent}let r{op_id} = r{x} == r{y};").unwrap(),
                    }
                }
                &Op::Mad { x, y, z } => {
                    dtypes.insert(op_id, dtypes[&x]);
                    writeln!(source, "{indent}let r{op_id} = r{x} * r{y} + r{z};").unwrap();
                }
                Op::Vectorize { .. } => todo!(),
                Op::Devectorize { .. } => todo!(),
                &Op::Index { len, scope, axis: _ } => {
                    dtypes.insert(op_id, IDX_T);
                    match scope {
                        Scope::Global => {
                            writeln!(
                                source,
                                "{indent}let r{op_id} = {}(gidx[{loop_id}]); // 0..={}",
                                IDX_T.wgsl(),
                                len - 1
                            )
                            .unwrap();
                            n_global_ids += 1;
                        }
                        Scope::Local => {
                            writeln!(
                                source,
                                "{indent}let r{op_id} = {}(lidx[{}]); // 0..={}",
                                IDX_T.wgsl(),
                                loop_id - n_global_ids,
                                len - 1
                            )
                            .unwrap();
                        }
                        Scope::Register => {}
                    }
                    loop_id += 1;
                }
                &Op::Loop { len } => {
                    dtypes.insert(op_id, IDX_T);
                    writeln!(
                        source,
                        "{indent}for (var r{op_id}: {} = 0; r{op_id} < {len}; r{op_id} += 1) {{",
                        IDX_T.wgsl()
                    )
                    .unwrap();
                    indent += "  ";
                    loop_id += 1;
                }
                Op::EndLoop => {
                    if loop_id as usize > lws.len() + gws.len() {
                        indent.pop();
                        indent.pop();
                        writeln!(source, "{indent}}}").unwrap();
                        loop_id -= 1;
                    }
                }
                &Op::If { condition } => {
                    _ = writeln!(source, "{indent}if (r{condition}) {{");
                    indent += "  ";
                }
                Op::EndIf => {
                    indent.pop();
                    indent.pop();
                    _ = writeln!(source, "{indent}}}");
                }
                Op::Barrier { scope } => match scope {
                    Scope::Global | Scope::Register => unreachable!(),
                    Scope::Local => _ = writeln!(source, "{indent}workgroupBarrier();"),
                },
            }
            op_id = kernel.next_op(op_id);
        }

        let mut pragma = String::new();
        if dtypes.values().any(|&v| v == DType::F16) {
            pragma += "enable f16;\n";
        }

        let name = format!(
            "k_{}__{}",
            gws.iter().map(ToString::to_string).collect::<Vec<_>>().join("_"),
            lws.iter().map(ToString::to_string).collect::<Vec<_>>().join("_"),
        );

        let workgroup_size = if lws.is_empty() {
            "1".to_string()
        } else {
            lws.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
        };
        let source = format!(
            "{pragma}{global_args}{workgroup_args}@compute @workgroup_size({workgroup_size}) fn {name}(
  @builtin(workgroup_id) gidx: vec3<u32>,
  @builtin(local_invocation_id) lidx: vec3<u32>
) {{\n{source}}}\n",
        );
        if debug_asm {
            println!();
            println!("{source}");
        }

        let shader_module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(source)),
        });
        let id = self.programs.push(WGPUProgram {
            name,
            gws,
            //local_work_size,
            arg_ro_flags,
            shader: shader_module,
        });

        Ok(id)
    }

    pub fn release(&mut self, program_id: DeviceProgramId) {
        self.programs.remove(program_id);
    }

    #[allow(clippy::unnecessary_wraps)]
    pub fn launch(
        &mut self,
        program_id: DeviceProgramId,
        memory_pool: &mut WGPUMemoryPool,
        args: &[PoolBufferId],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        drop(event_wait_list);
        let program = &self.programs[program_id];
        let mut set_layout: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();
        let mut binds: Vec<wgpu::BindGroupEntry> = Vec::new();
        for (bind_id, &arg) in args.iter().enumerate() {
            let bind_entry = wgpu::BindGroupLayoutEntry {
                binding: u32::try_from(bind_id).unwrap(),
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    has_dynamic_offset: false,
                    min_binding_size: None,
                    ty: wgpu::BufferBindingType::Storage { read_only: program.arg_ro_flags[bind_id] },
                },
                count: None,
            };
            let buffer = &memory_pool.buffers[arg];
            let bind = wgpu::BindGroupEntry { binding: u32::try_from(bind_id).unwrap(), resource: buffer.as_entire_binding() };
            set_layout.push(bind_entry);
            binds.push(bind);
        }
        // Program
        // shader
        // entry point - function name
        // descriptors
        let mut layouts = Vec::new();
        let mut sets = Vec::new();
        // Unwraping of descriptors from program
        let set_layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &set_layout });
        let set = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &set_layout, entries: &binds });
        layouts.push(set_layout);
        sets.push(set);
        // Compute pipeline bindings
        let group_layouts = layouts.iter().collect::<Vec<_>>();
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &group_layouts,
            immediate_size: 0,
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            module: &program.shader,
            entry_point: Some(&program.name),
            layout: Some(&pipeline_layout),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Kernel::enqueue") });
        {
            let mut cpass = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Kernel::enqueue"), timestamp_writes: None });
            cpass.set_pipeline(&pipeline);
            for (id_set, set) in sets.iter().enumerate() {
                cpass.set_bind_group(u32::try_from(id_set).unwrap(), set, &[]);
            }
            cpass.insert_debug_marker(&program.name);
            cpass.dispatch_workgroups(
                u32::try_from(program.gws.first().copied().unwrap_or(1)).unwrap(),
                u32::try_from(program.gws.get(1).copied().unwrap_or(1)).unwrap(),
                u32::try_from(program.gws.get(2).copied().unwrap_or(1)).unwrap(),
            );
        }
        let submission_index = Some(self.queue.submit(Some(encoder.finish())));
        Ok(Event::WGPU(WGPUEvent { submission_index }))
    }
}

impl DType {
    const fn wgsl(&self) -> &str {
        match self {
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::U16 => "u16",
            DType::U32 => "u32",
            DType::U64 => "u64",
            DType::Bool => "bool",
        }
    }
}

impl Constant {
    fn wgsl(self) -> String {
        match self {
            Constant::F16(x) => format!("f16({})", half::f16::from_le_bytes(x)),
            Constant::BF16(x) => format!("bf16({})", half::bf16::from_le_bytes(x)),
            Constant::F32(x) => format!("f32({:.16})", f32::from_le_bytes(x)),
            Constant::F64(x) => format!("f64({:.16})", f64::from_le_bytes(x)),
            Constant::I8(x) => format!("i8({x})"),
            Constant::I16(x) => format!("i16({x})"),
            Constant::I32(x) => format!("i32({x})"),
            Constant::I64(x) => format!("i64({})", i64::from_le_bytes(x)),
            Constant::U8(x) => format!("u8({x})"),
            Constant::U16(x) => format!("u16({x})"),
            Constant::U32(x) => format!("{x}"),
            Constant::U64(x) => format!("u64({})", u64::from_le_bytes(x)),
            Constant::Bool(x) => format!("{x}"),
        }
    }
}
