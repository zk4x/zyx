use super::{BackendError, Device, DeviceInfo, ErrorStatus, Event, MemoryPool};
use crate::{
    DType, Map,
    backend::{BufferId, ProgramId},
    dtype::Constant,
    graph::{BOp, UOp},
    kernel::{IDX_T, Kernel, Op, OpId, Scope},
    runtime::Pool,
    slab::Slab,
};
use nanoserde::DeJson;
use pollster::FutureExt;
use std::{fmt::Write, hash::BuildHasherDefault, sync::Arc};
use wgpu::{BufferDescriptor, BufferUsages, PowerPreference, ShaderModule, ShaderModuleDescriptor, ShaderSource};

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
    free_bytes: usize,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffers: Slab<BufferId, wgpu::Buffer>,
}

#[derive(Debug)]
pub struct WGPUDevice {
    dev_info: DeviceInfo,
    memory_pool_id: u32,
    device: Arc<wgpu::Device>,
    #[allow(unused)]
    adapter: wgpu::Adapter,
    programs: Slab<ProgramId, WGPUProgram>,
    queue: Arc<wgpu::Queue>,
}

#[derive(Debug, Clone)]
pub struct WGPUEvent {}

#[derive(Debug)]
pub(super) struct WGPUProgram {
    name: String,
    gws: Vec<usize>,
    //local_work_size: [usize; 3],
    arg_ro_flags: Vec<bool>,
    shader: ShaderModule,
}

pub(super) fn initialize_device(
    config: &WGPUConfig,
    memory_pools: &mut Vec<Pool>,
    devices: &mut Vec<Device>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    if !config.enabled {
        return Err(BackendError {
            status: super::ErrorStatus::Initialization,
            context: "WGPU configured out.".into(),
        });
    }

    let power_preference = PowerPreference::from_env().unwrap_or(wgpu::PowerPreference::HighPerformance);
    let instance =
        wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), ..Default::default() });

    if debug_dev {
        println!("WGPU Requesting device with {power_preference:#?} power preference");
    }

    let (adapter, device, queue) = async {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference, ..Default::default() })
            .await
            .expect("Failed at adapter creation.");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::SHADER_INT64
                    | wgpu::Features::SHADER_F64
                    | wgpu::Features::SHADER_F16,
                required_limits: wgpu::Limits { max_storage_buffers_per_shader_stage: 32, ..Default::default() },
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed at device creation");
        (adapter, device, queue)
    }
    .block_on();

    let info = adapter.get_info();
    if debug_dev {
        println!("Using {} ({}) - {:#?}.", info.name, info.device, info.backend);
    }
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let pool = MemoryPool::WGPU(WGPUMemoryPool {
        free_bytes: 1_000_000_000,
        device: device.clone(),
        queue: queue.clone(),
        buffers: Slab::new(),
    });
    memory_pools.push(Pool::new(pool));
    devices.push(Device::WGPU(WGPUDevice {
        device: device.clone(),
        adapter,
        dev_info: DeviceInfo {
            compute: 1024 * 1024 * 1024 * 1024,
            max_global_work_dims: vec![1024, 1024, 1024],
            max_local_threads: 256,
            max_local_work_dims: vec![256, 256, 256],
            preferred_vector_size: 4,
            local_mem_size: 64 * 1024,
            num_registers: 96,
            tensor_cores: false,
        },
        memory_pool_id: (memory_pools.len() - 1).try_into().unwrap(),
        programs: Slab::new(),
        queue: queue.clone(),
    }));

    Ok(())
}

impl WGPUMemoryPool {
    pub fn deinitialize(&mut self) {}

    pub const fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    pub fn allocate(&mut self, bytes: usize) -> Result<(BufferId, Event), BackendError> {
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
        let event = Event::WGPU(WGPUEvent {});
        Ok((id, event))
    }

    pub fn deallocate(&mut self, buffer_id: BufferId, event_wait_list: Vec<Event>) {
        let _ = event_wait_list;
        let buffer = unsafe { self.buffers.remove_and_return(buffer_id) };
        buffer.destroy();
    }

    pub fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: BufferId,
        event_wait_list: Vec<Event>,
    ) -> Result<super::Event, BackendError> {
        let _ = event_wait_list;
        let dst = &self.buffers[dst];
        self.queue.write_buffer(&dst, 0, src);
        let encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("GpuBuffer::write") });
        self.queue.submit(Some(encoder.finish()));
        Ok(Event::WGPU(WGPUEvent {}))
    }

    /*pub fn pool_to_host(
        &mut self,
        src: BufferId,
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

    pub fn pool_to_host(
        &mut self,
        src: BufferId,
        dst: &mut [u8],
        event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError> {
        let _ = event_wait_list; // You can eventually use events if needed

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
            &src,
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

    pub fn sync_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        let _ = events;
        Ok(())
    }

    pub fn release_events(&mut self, events: Vec<Event>) {
        let _ = events;
    }
}

impl WGPUDevice {
    pub const fn deinitialize(&mut self) {}

    pub const fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    pub const fn memory_pool_id(&self) -> u32 {
        self.memory_pool_id
    }

    pub const fn free_compute(&self) -> u128 {
        self.dev_info.compute
    }

    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<ProgramId, BackendError> {
        let mut gws = Vec::new();
        let mut lws = Vec::new();
        for &op_id in &kernel.order {
            if let &Op::Loop { dim, scope } = &kernel.ops[op_id] {
                match scope {
                    Scope::Global => {
                        gws.push(dim);
                    }
                    Scope::Local => {
                        lws.push(dim);
                    }
                    Scope::Register => {}
                }
            }
        }

        if lws.iter().product::<usize>() > self.dev_info.max_local_threads {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "Invalid local work size.".into(),
            });
        }

        let mut arg_ro_flags = Vec::new();
        let mut global_args = String::new();
        let mut max_p = 0;
        for (op_id, op) in kernel.ops.iter() {
            if let &Op::Define { dtype, scope, ro, .. } = op
                && scope == Scope::Global
            {
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
        }

        let mut dtypes: Map<OpId, DType> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
        let mut n_global_ids = 0;
        let mut loop_id = 0;
        let mut indent = String::from("  ");
        let mut source = String::with_capacity(1000);

        for &op_id in &kernel.order {
            //println!("{i} -> {op:?}");
            match &kernel.ops[op_id] {
                Op::ConstView { .. } | Op::LoadView { .. } | Op::StoreView { .. } | Op::Reduce { .. } => {
                    unreachable!()
                }
                &Op::Const(x) => {
                    dtypes.insert(op_id, x.dtype());
                    writeln!(source, "{indent}const r{op_id}: {} = {};", x.dtype().wgsl(), x.wgsl()).unwrap();
                }
                &Op::Define { dtype, scope, ro: _, len } => {
                    dtypes.insert(op_id, dtype);
                    if scope == Scope::Register {
                        writeln!(source, "{indent}var p{op_id}: array<{}, {len}>;", dtype.wgsl(),).unwrap();
                    }
                }
                &Op::Load { src, index } => {
                    dtypes.insert(op_id, dtypes[&src]);
                    writeln!(source, "{indent}let r{op_id} = p{src}[r{index}];").unwrap();
                }
                &Op::Store { dst, x: src, index } => {
                    writeln!(source, "{indent}p{dst}[r{}] = r{};", index, src,).unwrap();
                }
                &Op::Cast { x, dtype } => {
                    dtypes.insert(op_id, dtype);
                    writeln!(source, "{indent}let r{op_id} = {}(r{x});", dtype.wgsl()).unwrap();
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
                        BOp::Maximum => writeln!(source, "{indent}let r{op_id} = max(r{x}, r{y});").unwrap(),
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
                &Op::Loop { dim, scope } => {
                    dtypes.insert(op_id, IDX_T);
                    match scope {
                        Scope::Global => {
                            writeln!(
                                source,
                                "{indent}let r{op_id} = {}(gidx[{loop_id}]); // 0..={}",
                                IDX_T.wgsl(),
                                dim - 1
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
                                dim - 1
                            )
                            .unwrap();
                        }
                        Scope::Register => {
                            writeln!(
                                source,
                                "{indent}for (var r{op_id}: {} = 0; r{op_id} < {dim}; r{op_id} += 1) {{",
                                IDX_T.wgsl()
                            )
                            .unwrap();
                            indent += "  ";
                        }
                    }
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
            }
        }

        let mut pragma = String::new();
        if source.contains("f16") {
            pragma += "enable f16;\n";
        }

        let name = format!(
            "k_{}__{}",
            gws.iter().map(ToString::to_string).collect::<Vec<_>>().join("_"),
            lws.iter().map(ToString::to_string).collect::<Vec<_>>().join("_"),
        );

        let source = format!(
            "{pragma}{global_args}@compute @workgroup_size({}) fn {name}(
  @builtin(workgroup_id) gidx: vec3<u32>,
  @builtin(local_invocation_id) lidx: vec3<u32>\n) {{\n{source}}}\n",
            lws.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
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

    pub fn release(&mut self, program_id: ProgramId) {
        self.programs.remove(program_id);
    }

    pub fn launch(
        &mut self,
        program_id: ProgramId,
        memory_pool: &mut WGPUMemoryPool,
        args: &[BufferId],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let _ = event_wait_list;
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
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: program.arg_ro_flags[bind_id], // TODO make this work properly with read only args
                    },
                },
                count: None,
            };
            let buffer = &memory_pool.buffers[arg];
            let bind =
                wgpu::BindGroupEntry { binding: u32::try_from(bind_id).unwrap(), resource: buffer.as_entire_binding() };
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
        let set = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &set_layout,
            entries: &binds,
        });
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
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Kernel::enqueue") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Kernel::enqueue"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            for (id_set, set) in sets.iter().enumerate() {
                cpass.set_bind_group(u32::try_from(id_set).unwrap(), set, &[]);
            }
            cpass.insert_debug_marker(&program.name);
            cpass.dispatch_workgroups(
                u32::try_from(program.gws.get(0).copied().unwrap_or(1)).unwrap(),
                u32::try_from(program.gws.get(1).copied().unwrap_or(1)).unwrap(),
                u32::try_from(program.gws.get(2).copied().unwrap_or(1)).unwrap(),
            );
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(Event::WGPU(WGPUEvent {}))
    }
}

impl DType {
    fn wgsl(&self) -> &str {
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
