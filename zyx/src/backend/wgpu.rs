use std::{borrow::Cow, collections::BTreeMap, sync::Arc};

use nanoserde::DeJson;
use pollster::FutureExt;
use wgpu::{
    util::DownloadBuffer, BufferDescriptor, BufferUsages, Maintain, ShaderModule,
    ShaderModuleDescriptor, ShaderSource,
};

use crate::{
    dtype::Constant,
    ir::{IRKernel, IROp, Reg, Scope},
    node::{BOp, UOp},
    runtime::Pool,
    slab::{Id, Slab},
    DType,
};

use super::{BackendError, BufferMut, Device, DeviceInfo, ErrorStatus, Event, MemoryPool};

#[derive(DeJson, Debug, Default)]
pub struct WGPUConfig {
    enabled: bool,
}

#[derive(Debug)]
pub(super) struct WGPUMemoryPool {
    free_bytes: usize,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffers: Slab<wgpu::Buffer>,
}

#[derive(Debug)]
pub(super) struct WGPUDevice {
    dev_info: DeviceInfo,
    memory_pool_id: u32,
    device: Arc<wgpu::Device>,
    #[allow(unused)]
    adapter: wgpu::Adapter,
    programs: Slab<WGPUProgram>,
    queue: Arc<wgpu::Queue>,
}

pub(super) type WGPUBuffer = wgpu::Buffer;

#[derive(Debug)]
pub struct WGPUEvent {}

#[derive(Debug)]
pub(super) struct WGPUProgram {
    name: String,
    global_work_size: [usize; 3],
    //local_work_size: [usize; 3],
    read_only_args: Vec<bool>,
    shader: ShaderModule,
}

pub(super) fn initialize_device(
    config: &WGPUConfig,
    memory_pools: &mut Vec<Pool>,
    devices: &mut Vec<Box<dyn Device>>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    if !config.enabled {
        return Err(BackendError {
            status: super::ErrorStatus::Initialization,
            context: "WGPU configured out.".into(),
        });
    }

    let power_preference =
        wgpu::util::power_preference_from_env().unwrap_or(wgpu::PowerPreference::HighPerformance);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    if debug_dev {
        println!("Requesting device with {power_preference:#?}");
    }

    let (adapter, device, queue) = async {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                ..Default::default()
            })
            .await
            .expect("Failed at adapter creation.");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: adapter.features(),
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("Failed at device creation.");
        (adapter, device, queue)
    }
    .block_on();

    let info = adapter.get_info();
    if debug_dev {
        println!(
            "Using {} ({}) - {:#?}.",
            info.name, info.device, info.backend
        );
    }
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let pool = Box::new(WGPUMemoryPool {
        free_bytes: 1_000_000_000,
        device: device.clone(),
        queue: queue.clone(),
        buffers: Slab::new(),
    });
    memory_pools.push(Pool { pool, events: BTreeMap::new(), buffer_map: BTreeMap::new() });
    devices.push(Box::new(WGPUDevice {
        device: device.clone(),
        adapter,
        dev_info: DeviceInfo {
            compute: 1024 * 1024 * 1024 * 1024,
            max_global_work_dims: [1024, 1024, 1024],
            max_local_threads: 256,
            max_local_work_dims: [256, 256, 256],
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

impl MemoryPool for WGPUMemoryPool {
    fn deinitialize(&mut self) -> Result<(), BackendError> {
        Ok(())
    }

    fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    fn get_buffer(&self, buffer: Id) -> BufferMut {
        BufferMut::WGPU(&self.buffers[buffer])
    }

    fn allocate(&mut self, bytes: usize) -> Result<(Id, Event), BackendError> {
        if bytes > self.free_bytes {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "".into() });
        }
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: bytes as u64,
            usage: BufferUsages::from_bits_truncate(
                BufferUsages::STORAGE.bits()
                    | BufferUsages::COPY_SRC.bits()
                    | BufferUsages::COPY_DST.bits(),
            ),
            mapped_at_creation: false,
        });
        let id = self.buffers.push(buffer);
        let event = Event::WGPU(WGPUEvent {});
        Ok((id, event))
    }

    fn deallocate(
        &mut self,
        buffer_id: Id,
        event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError> {
        let _ = event_wait_list;
        if let Some(buffer) = self.buffers.remove(buffer_id) {
            buffer.destroy();
        }
        Ok(())
    }

    fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: Id,
        event_wait_list: Vec<Event>,
    ) -> Result<super::Event, BackendError> {
        let _ = event_wait_list;
        let dst = &self.buffers[dst];
        self.queue.write_buffer(&dst, 0, src);
        let encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GpuBuffer::write"),
        });
        self.queue.submit(Some(encoder.finish()));
        Ok(Event::WGPU(WGPUEvent {}))
    }

    fn pool_to_host(
        &mut self,
        src: Id,
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
            self.device.poll(Maintain::Wait);
            let download = rx.await.unwrap().unwrap();
            dst.copy_from_slice(&download);
        }
        .block_on();
        Ok(())
    }
}

impl Device for WGPUDevice {
    fn deinitialize(&mut self) -> Result<(), BackendError> {
        Ok(())
    }

    fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    fn memory_pool_id(&self) -> u32 {
        self.memory_pool_id
    }

    fn compute(&self) -> u128 {
        self.dev_info.compute
    }

    fn compile(&mut self, kernel: &IRKernel, debug_asm: bool) -> Result<Id, BackendError> {
        let mut source = String::new();
        let mut indent = String::from("  ");

        let mut global_work_size = [0; 3];
        let mut local_work_size = [0; 3];

        let mut loops = [0; 6];
        for (i, op) in kernel.ops[..6].iter().enumerate() {
            if let &IROp::Loop { id, len } = op {
                if i % 2 == 0 {
                    global_work_size[i / 2] = len;
                } else {
                    local_work_size[i / 2] = len;
                }
                loops[i] = id;
            } else {
                unreachable!()
            }
        }

        // Declare global variables
        let mut read_only_args = Vec::new();
        for (id, (scope, dtype, _, read_only)) in kernel.addressables.iter().enumerate() {
            if *scope == Scope::Global {
                source += &format!(
                    "@group(0) @binding({id}) var<storage, {}> p{id}: array<{}>;\n",
                    if *read_only { "read" } else { "read_write" },
                    dtype.wgsl(),
                );
            }
            read_only_args.push(*read_only);
        }
        // Declare local variables
        for (id, (scope, dtype, len, _)) in kernel.addressables.iter().enumerate() {
            if *scope == Scope::Local {
                source += &format!("var<workgroup> p{id}: array<{}, {len}>;\n", dtype.wgsl(),);
            }
        }

        // Function declaration and name
        source += &format!(
            "@compute @workgroup_size({}, {}, {})\n",
            local_work_size[0], local_work_size[1], local_work_size[2]
        );
        let name = format!(
            "k_{}_{}_{}__{}_{}_{}",
            global_work_size[0],
            global_work_size[1],
            global_work_size[2],
            local_work_size[0],
            local_work_size[1],
            local_work_size[2],
        );
        source += &format!("fn {name}(\n{indent}@builtin(workgroup_id) gid: vec3<u32>,\n{indent}@builtin(local_invocation_id) lid: vec3<u32>\n) {{\n");

        // Declare register variables
        for (id, dtype) in kernel.registers.iter().enumerate() {
            source += &format!("{indent}var r{id}: {};\n", dtype.wgsl());
        }

        // Add indices for global and local loops
        source += &format!(
            "  r{} = u64(gid.x);   /* 0..{} */\n",
            loops[0], global_work_size[0]
        );
        source += &format!(
            "  r{} = u64(lid.x);   /* 0..{} */\n",
            loops[1], local_work_size[0]
        );
        source += &format!(
            "  r{} = u64(gid.y);   /* 0..{} */\n",
            loops[2], global_work_size[1]
        );
        source += &format!(
            "  r{} = u64(lid.y);   /* 0..{} */\n",
            loops[3], local_work_size[1]
        );
        source += &format!(
            "  r{} = u64(gid.z);   /* 0..{} */\n",
            loops[4], global_work_size[2]
        );
        source += &format!(
            "  r{} = u64(lid.z);   /* 0..{} */\n",
            loops[5], local_work_size[2]
        );

        for op in kernel.ops[6..kernel.ops.len() - 6].iter().copied() {
            match op {
                IROp::Load { z, address, offset } => {
                    source += &format!("{indent}r{z} = p{address}[{}];\n", offset.wgsl());
                }
                IROp::Store { address, x, offset } => {
                    source += &format!("{indent}p{address}[{}] = {};\n", offset.wgsl(), x.wgsl());
                }
                IROp::Set { z, value } => {
                    source += &format!("{indent}r{z} = {};\n", value.wgsl());
                }
                IROp::Unary { z, x, uop } => {
                    let dtype = kernel.registers[z as usize];
                    source += &match uop {
                        UOp::Cast(dtype) => {
                            format!("{indent}r{z} = {}(r{x});\n", dtype.wgsl(),)
                        }
                        UOp::ReLU => format!(
                            "{indent}r{z} = max(r{x}, {});\n",
                            dtype.zero_constant().wgsl()
                        ),
                        UOp::Neg => format!("{indent}r{z} = -r{x};\n"),
                        UOp::Exp2 => format!("{indent}r{z} = exp2(r{x});\n"),
                        UOp::Log2 => format!("{indent}r{z} = log2(r{x});\n"),
                        UOp::Inv => format!("{indent}r{z} = 1/r{x};\n"),
                        UOp::Sqrt => format!("{indent}r{z} = sqrt(r{x});\n"),
                        UOp::Sin => format!("{indent}r{z} = sin(r{x});\n"),
                        UOp::Cos => format!("{indent}r{z} = cos(r{x});\n"),
                        UOp::Not => {
                            format!(
                                "{indent}r{z} = {}(r{x} == 0);\n",
                                kernel.registers[z as usize].wgsl(),
                            )
                        }
                    };
                }
                IROp::Binary { z, x, y, bop } => {
                    source += &format!(
                        "{indent}r{z} = {};\n",
                        match bop {
                            BOp::Add => format!("{} + {}", x.wgsl(), y.wgsl()),
                            BOp::Sub => format!("{} - {}", x.wgsl(), y.wgsl()),
                            BOp::Mul => format!("{} * {}", x.wgsl(), y.wgsl()),
                            BOp::Div => format!("{} / {}", x.wgsl(), y.wgsl()),
                            BOp::Mod => format!("{} % {}", x.wgsl(), y.wgsl()),
                            BOp::Pow => format!("pow({}, {})", x.wgsl(), y.wgsl()),
                            BOp::Max => format!("max({}, {})", x.wgsl(), y.wgsl()),
                            BOp::BitOr => format!("{} | {}", x.wgsl(), y.wgsl()),
                            BOp::BitXor => format!("{} ^ {}", x.wgsl(), y.wgsl()),
                            BOp::BitAnd => format!("{} & {}", x.wgsl(), y.wgsl()),
                            BOp::Cmplt => format!("{} < {}", x.wgsl(), y.wgsl()),
                            BOp::Cmpgt => format!("{} > {}", x.wgsl(), y.wgsl()),
                            BOp::NotEq => format!("{} != {}", x.wgsl(), y.wgsl()),
                            BOp::Or => format!("{} || {}", x.wgsl(), y.wgsl()),
                            BOp::And => format!("{} && {}", x.wgsl(), y.wgsl()),
                            BOp::BitShiftLeft => format!(
                                "{} << {}",
                                x.wgsl(),
                                if let Reg::Const(y) = y {
                                    Reg::Const(y.unary(UOp::Cast(DType::U32)))
                                } else {
                                    y
                                }
                                .wgsl()
                            ),
                            BOp::BitShiftRight => format!(
                                "{} >> {}",
                                x.wgsl(),
                                if let Reg::Const(y) = y {
                                    Reg::Const(y.unary(UOp::Cast(DType::U32)))
                                } else {
                                    y
                                }
                                .wgsl()
                            ),
                        }
                    );
                }
                IROp::MAdd { z, a, b, c } => {
                    source += &format!(
                        "{indent}r{z} = {} * {} + {};\n",
                        a.wgsl(),
                        b.wgsl(),
                        c.wgsl()
                    );
                }
                IROp::Loop { id, len } => {
                    source += &format!(
                        "{indent}for (var r{id}: u64 = 0; r{id} < {len}; r{id} = r{id} + 1) {{\n"
                    );
                    indent += "  ";
                }
                IROp::EndLoop { .. } => {
                    indent.pop();
                    indent.pop();
                    source += &format!("{indent}}}\n");
                }
                IROp::Barrier { scope } => match scope {
                    Scope::Global => source += &format!("{indent}storageBarrier();\n"),
                    Scope::Local => source += &format!("{indent}workgroupBarrier();\n"),
                    Scope::Register => unreachable!(),
                },
            }
        }
        source += "}\n";
        //println!("{source}");
        if debug_asm {
            println!("{source}");
        }
        if source.contains("f16") {
            source.insert_str(0, "enable f16;\n");
        }

        let shader_module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Owned(source)),
        });
        let id = self.programs.push(WGPUProgram {
            name,
            global_work_size,
            //local_work_size,
            read_only_args,
            shader: shader_module,
        });

        Ok(id)
    }

    fn release(&mut self, program_id: Id) -> Result<(), BackendError> {
        self.programs.remove(program_id);
        Ok(())
    }

    fn launch(
        &mut self,
        program_id: Id,
        memory_pool: &mut dyn MemoryPool,
        args: &[Id],
        event_wait_list: Vec<Event>,
        // Immediatelly synchronize?
        sync: bool,
    ) -> Result<Event, BackendError> {
        let _ = event_wait_list;
        let _ = sync;
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
                        read_only: program.read_only_args[bind_id], // TODO make this work properly with read only args
                    },
                },
                count: None,
            };
            let BufferMut::WGPU(buffer) = memory_pool.get_buffer(arg) else { unreachable!() };
            let bind = wgpu::BindGroupEntry {
                binding: u32::try_from(bind_id).unwrap(),
                resource: buffer.as_entire_binding(),
            };
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
        let set_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &set_layout,
        });
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
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Kernel::enqueue"),
        });
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
                u32::try_from(program.global_work_size[0]).unwrap(),
                u32::try_from(program.global_work_size[1]).unwrap(),
                u32::try_from(program.global_work_size[2]).unwrap(),
            );
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(Event::WGPU(WGPUEvent {}))
    }

    fn sync(&mut self, event_wait_list: Vec<super::Event>) -> Result<(), BackendError> {
        let _ = event_wait_list;
        Ok(())
    }
}

impl DType {
    fn wgsl(&self) -> &str {
        match self {
            DType::BF16 => todo!("WIP"),
            DType::F8 => "f8",
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
    fn wgsl(&self) -> String {
        match self {
            &Constant::F8(x) => format!("f8({})", float8::F8E4M3::from_bits(x)),
            &Constant::F16(x) => format!("f16({})", half::f16::from_bits(x)),
            &Constant::BF16(x) => format!("bf16({})", half::bf16::from_bits(x)),
            &Constant::F32(x) => format!("f32({:.16})", f32::from_bits(x)),
            &Constant::F64(x) => format!("f64({:.16})", f64::from_bits(x)),
            Constant::I8(x) => format!("i8({x})"),
            Constant::I16(x) => format!("i16({x})"),
            Constant::I32(x) => format!("i32({x})"),
            Constant::I64(x) => format!("i64({x})"),
            Constant::U8(x) => format!("u8({x})"),
            Constant::U16(x) => format!("u16({x})"),
            Constant::U32(x) => format!("u32({x})"),
            Constant::U64(x) => format!("u64({x})"),
            Constant::Bool(x) => format!("{x}"),
        }
    }
}

impl Reg {
    fn wgsl(&self) -> String {
        match self {
            Reg::Var(id) => format!("r{id}"),
            Reg::Const(value) => value.wgsl(),
        }
    }
}
