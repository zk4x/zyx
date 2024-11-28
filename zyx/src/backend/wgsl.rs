//! WGSL backend

use pollster::FutureExt;
use std::{borrow::Cow, sync::Arc};
use wgpu::{
    util::DownloadBuffer, BufferDescriptor, BufferUsages, Maintain, ShaderModule,
    ShaderModuleDescriptor, ShaderSource,
};

use super::DeviceInfo;
use crate::{
    dtype::Constant,
    index_map::{Id, IndexMap},
    runtime::{
        ir::{IRDType, IRKernel, IROp, Reg, Scope},
        node::{BOp, UOp},
    },
};

#[derive(serde::Deserialize, Debug, Default)]
pub struct WGSLConfig {
    use_wgsl: bool,
}

#[derive(Debug)]
pub struct WGSLError {}

#[derive(Debug)]
pub(super) struct WGSLMemoryPool {
    free_bytes: usize,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

#[derive(Debug)]
pub(super) struct WGSLBuffer {
    buffer: wgpu::Buffer,
}

#[derive(Debug)]
pub(super) struct WGSLDevice {
    dev_info: DeviceInfo,
    memory_pool_id: u32,
    device: Arc<wgpu::Device>,
    #[allow(unused)]
    adapter: wgpu::Adapter,
}

#[derive(Debug)]
pub(super) struct WGSLProgram {
    name: String,
    global_work_size: [usize; 3],
    //local_work_size: [usize; 3],
    read_only_args: Vec<bool>,
    shader: ShaderModule,
}

#[derive(Debug)]
pub(super) struct WGSLQueue {
    load: usize,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

type WGSLQueuePool = Vec<(WGSLDevice, Vec<WGSLQueue>)>;

pub(super) fn initialize_backend(
    config: &WGSLConfig,
    debug_dev: bool,
) -> Result<(Vec<WGSLMemoryPool>, WGSLQueuePool), WGSLError> {
    if !config.use_wgsl {
        return Err(WGSLError {});
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
    let mut memory_pools = Vec::new();
    let mut devices = Vec::new();
    memory_pools.push(WGSLMemoryPool {
        free_bytes: 1_000_000_000,
        device: device.clone(),
        queue: queue.clone(),
    });
    devices.push((
        WGSLDevice {
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
            memory_pool_id: 0,
        },
        vec![WGSLQueue {
            load: 0,
            device,
            queue,
        }],
    ));

    Ok((memory_pools, devices))
}

impl WGSLMemoryPool {
    pub(super) const fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::unnecessary_wraps)]
    pub(super) fn deinitialize(self) -> Result<(), WGSLError> {
        Ok(())
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(super) fn allocate(&mut self, bytes: usize) -> Result<WGSLBuffer, WGSLError> {
        if self.free_bytes < bytes {
            return Err(WGSLError {});
        }
        Ok(WGSLBuffer {
            buffer: self.device.create_buffer(&BufferDescriptor {
                label: None,
                size: bytes as u64,
                usage: BufferUsages::from_bits_truncate(
                    BufferUsages::STORAGE.bits()
                        | BufferUsages::COPY_SRC.bits()
                        | BufferUsages::COPY_DST.bits(),
                ),
                mapped_at_creation: false,
            }),
        })
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::unnecessary_wraps)]
    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(super) fn deallocate(&mut self, buffer: WGSLBuffer) -> Result<(), WGSLError> {
        buffer.buffer.destroy();
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(super) fn host_to_pool(&mut self, src: &[u8], dst: &WGSLBuffer) -> Result<(), WGSLError> {
        self.queue.write_buffer(&dst.buffer, 0, src);
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuBuffer::write"),
            });
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(super) fn pool_to_host(
        &mut self,
        src: &WGSLBuffer,
        dst: &mut [u8],
    ) -> Result<(), WGSLError> {
        async {
            let (tx, rx) = futures::channel::oneshot::channel();
            DownloadBuffer::read_buffer(
                &self.device,
                &self.queue,
                &src.buffer.slice(..),
                move |result| {
                    tx.send(result)
                        .unwrap_or_else(|_| panic!("Failed to download buffer."));
                },
            );
            self.device.poll(Maintain::Wait);
            let download = rx.await.unwrap().unwrap();
            dst.copy_from_slice(&download);
        }
        .block_on();
        Ok(())
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(super) fn pool_to_pool(
        &mut self,
        src: &WGSLBuffer,
        dst: &WGSLBuffer,
    ) -> Result<(), WGSLError> {
        let _ = src;
        let _ = dst;
        todo!()
    }
}

impl WGSLDevice {
    pub(super) const fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    // Memory pool id out of OpenCLMemoryPools
    pub(super) const fn memory_pool_id(&self) -> u32 {
        self.memory_pool_id
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::unnecessary_wraps)]
    pub(super) fn release_program(&self, program: WGSLProgram) -> Result<(), WGSLError> {
        drop(program);
        Ok(())
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unnecessary_wraps)]
    pub(super) fn release_queue(&self, queue: WGSLQueue) -> Result<(), WGSLError> {
        drop(queue);
        Ok(())
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::unnecessary_wraps)]
    pub(super) fn deinitialize(self) -> Result<(), WGSLError> {
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(super) fn compile(
        &mut self,
        kernel: &IRKernel,
        debug_asm: bool,
    ) -> Result<WGSLProgram, WGSLError> {
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
        source += "\n";
        // Declare local variables
        for (id, (scope, dtype, len, _)) in kernel.addressables.iter().enumerate() {
            if *scope == Scope::Local {
                source += &format!("var<workgroup> p{id}: array<{}, {len}>;\n", dtype.wgsl(),);
            }
        }

        // Function declaration and name
        source += &format!(
            "\n@compute @workgroup_size({}, {}, {})\n",
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

        // Declare acuumulators
        for (id, (scope, dtype, len, read_only)) in kernel.addressables.iter().enumerate() {
            if *scope == Scope::RegTile {
                source += &format!(
                    "{indent}{} p{id}: array<{}, {len}>;\n",
                    if *read_only { "let" } else { "var" },
                    dtype.wgsl(),
                );
            }
        }

        // Declare register variables
        for (id, dtype) in kernel.registers.iter().enumerate() {
            source += &format!("{indent}var r{id}: {};\n", dtype.wgsl());
        }

        // Add indices for global and local loops
        source += &format!(
            "  r{} = gid.x;   /* 0..{} */\n",
            loops[0], global_work_size[0]
        );
        source += &format!(
            "  r{} = lid.x;   /* 0..{} */\n",
            loops[1], local_work_size[0]
        );
        source += &format!(
            "  r{} = gid.y;   /* 0..{} */\n",
            loops[2], global_work_size[1]
        );
        source += &format!(
            "  r{} = lid.y;   /* 0..{} */\n",
            loops[3], local_work_size[1]
        );
        source += &format!(
            "  r{} = gid.z;   /* 0..{} */\n",
            loops[4], global_work_size[2]
        );
        source += &format!(
            "  r{} = lid.z;   /* 0..{} */\n",
            loops[5], local_work_size[2]
        );

        for op in kernel.ops[6..kernel.ops.len() - 6].iter().copied() {
            match op {
                /*IROp::Set { z, value } => {
                    source += &format!("{indent}r{z} = {};\n", value.wgsl());
                }*/
                IROp::Load { z, address, offset } => {
                    source += &format!("{indent}r{z} = p{address}[{}];\n", offset.wgsl());
                }
                IROp::Store { address, x, offset } => {
                    source += &format!("{indent}p{address}[{}] = {};\n", offset.wgsl(), x.wgsl());
                }
                IROp::Unary { z, x, uop } => {
                    let dtype = kernel.registers[z as usize];
                    source += &match uop {
                        UOp::Cast(dtype) => {
                            format!("{indent}r{z} = {}(r{x});\n", dtype.ir_dtype().wgsl(),)
                        }
                        UOp::ReLU => format!(
                            "{indent}r{z} = max(r{x}, {});\n",
                            dtype.dtype().zero_constant().wgsl()
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
                        "{indent}for (var r{id}: u32 = 0; r{id} < {len}; r{id} = r{id} + 1) {{\n"
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
                    Scope::Register | Scope::RegTile => unreachable!(),
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

        Ok(WGSLProgram {
            name,
            global_work_size,
            //local_work_size,
            read_only_args,
            shader: shader_module,
        })
    }
}

impl WGSLQueue {
    #[allow(clippy::unnecessary_wraps)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(super) fn launch(
        &mut self,
        program: &mut WGSLProgram,
        buffers: &mut IndexMap<WGSLBuffer>,
        args: &[Id],
    ) -> Result<(), WGSLError> {
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
            let bind = wgpu::BindGroupEntry {
                binding: u32::try_from(bind_id).unwrap(),
                resource: buffers[arg].buffer.as_entire_binding(),
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
        let set_layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &group_layouts,
                push_constant_ranges: &[],
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                module: &program.shader,
                entry_point: Some(&program.name),
                layout: Some(&pipeline_layout),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    pub(super) fn sync(&mut self) -> Result<(), WGSLError> {
        self.device.poll(Maintain::Wait);
        self.load = 0;
        Ok(())
    }

    pub(super) const fn load(&self) -> usize {
        self.load
    }
}

impl IRDType {
    fn wgsl(&self) -> &str {
        match self {
            IRDType::BF16(_) => todo!("WIP"),
            IRDType::F8(_) => "f8",
            IRDType::F16(_) => "f16",
            IRDType::F32(_) => "f32",
            IRDType::F64(_) => "f64",
            IRDType::U8(_) => "u8",
            IRDType::I8(_) => "i8",
            IRDType::I16(_) => "i16",
            IRDType::I32(_) => "i32",
            IRDType::I64(_) => "i64",
            IRDType::U32(_) => "u32",
            IRDType::U64(_) => "u64",
            IRDType::Bool => "bool",
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
            Constant::U8(x) => format!("{x}"),
            Constant::I8(x) => format!("{x}"),
            Constant::I16(x) => format!("{x}"),
            Constant::U32(x) => format!("{x}u"),
            Constant::U64(x) => format!("u64({x})"),
            Constant::I32(x) => format!("{x}"),
            Constant::I64(x) => format!("{x}"),
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
