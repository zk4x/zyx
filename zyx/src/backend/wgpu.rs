use super::{BackendError, Device, DeviceInfo, ErrorStatus, Event, MemoryPool};
use crate::{
    DType, Map,
    backend::{BufferId, ProgramId},
    dtype::Constant,
    graph::{BOp, UOp},
    kernel::{Kernel, Op, OpId, Scope},
    runtime::Pool,
    slab::Slab,
};
use nanoserde::DeJson;
use pollster::FutureExt;
use std::{fmt::Write, hash::BuildHasherDefault, sync::Arc};
use wgpu::{
    BufferDescriptor, BufferUsages, PollType, PowerPreference, ShaderModule, ShaderModuleDescriptor, ShaderSource,
    util::DownloadBuffer,
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
                trace: wgpu::Trace::Off,
                required_features: adapter.features(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::default(),
            })
            .await
            .expect("Failed at device creation.");
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

    pub fn pool_to_host(
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
            self.device.poll(PollType::Wait).unwrap();
            let download = rx.await.unwrap().unwrap();
            dst.copy_from_slice(&download);
        }
        .block_on();
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
        for op in &kernel.ops {
            if let &Op::Loop { dim, scope } = op {
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
        for (i, op) in kernel.ops.iter().enumerate() {
            if let &Op::Define { dtype, scope, ro, .. } = op
                && scope == Scope::Global
            {
                writeln!(
                    global_args,
                    "@group(0) @binding({i}) var<storage, {}> p{i}: array<{}>;",
                    if ro { "read" } else { "read_write" },
                    dtype.wgsl()
                )
                .unwrap();
                arg_ro_flags.push(ro);
            }
        }

        let mut dtypes: Map<OpId, DType> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
        let mut n_global_ids = 0;
        let mut loop_id = 0;
        let mut indent = String::from("  ");
        let mut source = String::with_capacity(1000);

        for (i, op) in kernel.ops.iter().enumerate() {
            //println!("{i} -> {op:?}");
            match op {
                Op::ConstView { .. } | Op::LoadView { .. } | Op::StoreView { .. } | Op::Reduce { .. } => unreachable!(),
                &Op::Const(x) => {
                    dtypes.insert(i, x.dtype());
                    writeln!(source, "{indent}const r{i}: {} = {};", x.dtype().wgsl(), x.wgsl()).unwrap();
                }
                &Op::Define { dtype, scope, ro: _, len } => {
                    dtypes.insert(i, dtype);
                    if scope == Scope::Register {
                        writeln!(source, "{indent}var p{i}: array<{}, {len}>;", dtype.wgsl(),).unwrap();
                    }
                }
                &Op::Load { src, index } => {
                    dtypes.insert(i, dtypes[&src]);
                    writeln!(source, "{indent}let r{i} = p{src}[r{index}];").unwrap();
                }
                &Op::Store { dst, x: src, index } => {
                    writeln!(source, "{indent}p{dst}[r{}] = r{};", index, src,).unwrap();
                }
                &Op::Cast { x, dtype } => {
                    dtypes.insert(i, dtype);
                    writeln!(source, "{indent}let r{i} = {}(r{x});", dtype.wgsl()).unwrap();
                }
                &Op::Unary { x, uop } => {
                    dtypes.insert(i, dtypes[&x]);
                    let dtype = dtypes[&x];
                    match uop {
                        UOp::ReLU => {
                            writeln!(
                                source,
                                "{indent}let r{i} = max(r{x}, {});",
                                dtype.zero_constant().wgsl()
                            )
                            .unwrap();
                        }
                        UOp::Neg => writeln!(source, "{indent}let r{i} = -r{x};").unwrap(),
                        UOp::Exp2 => {
                            //writeln!(source, "{indent}printf(\"%d\\n\", r{reg});").unwrap();
                            writeln!(source, "{indent}let r{i} = exp2(r{x});").unwrap();
                        }
                        UOp::Log2 => writeln!(source, "{indent}let r{i} = log2(r{x});").unwrap(),
                        UOp::Reciprocal => {
                            writeln!(source, "{indent}let r{i} = {}/r{x};", dtype.one_constant().wgsl()).unwrap();
                        }
                        UOp::Sqrt => writeln!(source, "{indent}let r{i} = sqrt(r{x});").unwrap(),
                        UOp::Sin => writeln!(source, "{indent}let r{i} = sin(r{x});").unwrap(),
                        UOp::Cos => writeln!(source, "{indent}let r{i} = cos(r{x});").unwrap(),
                        UOp::Floor => writeln!(source, "{indent}let r{i} = floor(r{x});").unwrap(),
                    }
                }
                &Op::Binary { x, y, bop } => {
                    if matches!(bop, BOp::Cmpgt | BOp::Cmplt | BOp::NotEq | BOp::Eq | BOp::And | BOp::Or) {
                        dtypes.insert(i, DType::Bool);
                    } else {
                        dtypes.insert(i, dtypes[&x]);
                    }
                    match bop {
                        BOp::Add => writeln!(source, "{indent}let r{i} = r{x} + r{y};").unwrap(),
                        BOp::Sub => writeln!(source, "{indent}let r{i} = r{x} - r{y};").unwrap(),
                        BOp::Mul => writeln!(source, "{indent}let r{i} = r{x} * r{y};").unwrap(),
                        BOp::Div => writeln!(source, "{indent}let r{i} = r{x} / r{y};").unwrap(),
                        BOp::Pow => writeln!(source, "{indent}let r{i} = pow(r{x}, r{y});").unwrap(),
                        BOp::Mod => writeln!(source, "{indent}let r{i} = r{x} % r{y};").unwrap(),
                        BOp::Cmplt => writeln!(source, "{indent}let r{i} = r{x} < r{y};").unwrap(),
                        BOp::Cmpgt => writeln!(source, "{indent}let r{i} = r{x} > r{y};").unwrap(),
                        BOp::Maximum => writeln!(source, "{indent}let r{i} = max(r{x}, r{y});").unwrap(),
                        BOp::Or => writeln!(source, "{indent}let r{i} = r{x} || r{y};").unwrap(),
                        BOp::And => writeln!(source, "{indent}let r{i} = r{x} && r{y};").unwrap(),
                        BOp::BitXor => writeln!(source, "{indent}let r{i} = r{x} ^ r{y};").unwrap(),
                        BOp::BitOr => writeln!(source, "{indent}let r{i} = r{x} | r{y};").unwrap(),
                        BOp::BitAnd => writeln!(source, "{indent}let r{i} = r{x} & r{y};").unwrap(),
                        BOp::BitShiftLeft => writeln!(source, "{indent}let r{i} = r{x} << r{y};").unwrap(),
                        BOp::BitShiftRight => writeln!(source, "{indent}let r{i} = r{x} >> r{y};").unwrap(),
                        BOp::NotEq => writeln!(source, "{indent}let r{i} = r{x} != r{y};").unwrap(),
                        BOp::Eq => writeln!(source, "{indent}let r{i} = r{x} == r{y};").unwrap(),
                    }
                }
                &Op::Loop { dim, scope } => {
                    dtypes.insert(i, DType::U32);
                    match scope {
                        Scope::Global => {
                            writeln!(source, "{indent}let r{i} = gidx[{loop_id}]; // 0..{dim}").unwrap();
                            n_global_ids += 1;
                        }
                        Scope::Local => {
                            writeln!(
                                source,
                                "{indent}let r{i} = lidx[{}]; // 0..{dim}",
                                loop_id - n_global_ids
                            )
                            .unwrap();
                        }
                        Scope::Register => {
                            writeln!(source, "{indent}for (var r{i}: u32 = 0; r{i} < {dim}; r{i} += 1) {{").unwrap();
                            indent += "  ";
                        }
                    }
                    loop_id += 1;
                }
                Op::EndLoop => {
                    indent.pop();
                    indent.pop();
                    writeln!(source, "{indent}}}").unwrap();
                    loop_id -= 1;
                }
            }
        }

        while loop_id as usize > lws.len() + gws.len() {
            indent.pop();
            indent.pop();
            loop_id -= 1;
            writeln!(source, "{indent}}}").unwrap();
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
            push_constant_ranges: &[],
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
