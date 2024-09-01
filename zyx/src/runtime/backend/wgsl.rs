use std::sync::Arc;

use wgpu::{util::DownloadBuffer, BufferDescriptor, BufferUsages};

use super::DeviceInfo;
use crate::{
    dtype::Constant,
    index_map::IndexMap,
    runtime::{
        ir::{IRDType, IRKernel, IROp, Scope, Var},
        node::{BOp, UOp},
    },
};

#[derive(serde::Deserialize, Debug, Default)]
pub struct WGSLConfig {}

#[derive(Debug)]
pub(crate) enum WGSLError {}

#[derive(Debug)]
pub(crate) struct WGSLMemoryPool {
    free_bytes: usize,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

#[derive(Debug)]
pub(crate) struct WGSLBuffer {
    buffer: wgpu::Buffer,
}

#[derive(Debug)]
pub(crate) struct WGSLDevice {
    dev_info: DeviceInfo,
    memory_pool_id: usize,
    device: Arc<wgpu::Device>,
    adapter: wgpu::Adapter,
}

#[derive(Debug)]
pub(crate) struct WGSLProgram {}

#[derive(Debug)]
pub(crate) struct WGSLQueue {
    load: usize,
    queue: Arc<wgpu::Queue>,
}

pub(crate) fn initialize_wgsl_backend(
    config: &WGSLConfig,
    debug_dev: bool,
) -> Result<(Vec<WGSLMemoryPool>, Vec<(WGSLDevice, Vec<WGSLQueue>)>), WGSLError> {
    let power_preference =
        wgpu::util::power_preference_from_env().unwrap_or(wgpu::PowerPreference::HighPerformance);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    println!("Requesting device with {:#?}", power_preference);

    let (adapter, device, queue) = futures::executor::block_on(async {
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
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .expect("Failed at device creation.");
        (adapter, device, queue)
    });

    let info = adapter.get_info();
    println!(
        "Using {} ({}) - {:#?}.",
        info.name, info.device, info.backend
    );
    let device = Arc::new(device);
    /*let polling_device = Arc::clone(&device);
    let handle = std::thread::spawn(move || loop {
        polling_device.poll(wgpu::Maintain::Poll);
        std::thread::sleep(std::time::Duration::from_millis(10));
    });
    handle.join().unwrap();*/

    let queue = Arc::new(queue);
    let mut memory_pools = Vec::new();
    let mut devices = Vec::new();
    memory_pools.push(WGSLMemoryPool {
        free_bytes: 1000000000,
        device: device.clone(),
        queue: queue.clone(),
    });
    devices.push((
        WGSLDevice {
            device,
            adapter,
            dev_info: DeviceInfo::default(),
            memory_pool_id: 0,
        },
        vec![WGSLQueue { load: 0, queue }],
    ));

    Ok((memory_pools, devices))
}

impl WGSLMemoryPool {
    pub(crate) fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    pub(crate) fn allocate(&mut self, bytes: usize) -> Result<WGSLBuffer, WGSLError> {
        Ok(WGSLBuffer {
            buffer: self.device.create_buffer(&BufferDescriptor {
                label: None,
                size: bytes as u64,
                usage: BufferUsages::from_bits_truncate(BufferUsages::STORAGE.bits() | BufferUsages::COPY_SRC.bits() | BufferUsages::COPY_DST.bits()),
                mapped_at_creation: false,
            }),
        })
    }

    pub(crate) fn deallocate(&mut self, buffer: WGSLBuffer) -> Result<(), WGSLError> {
        todo!()
    }

    pub(crate) fn host_to_pool(&mut self, src: &[u8], dst: &WGSLBuffer) -> Result<(), WGSLError> {
        self.queue.write_buffer(&dst.buffer, 0, src);
        let encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GpuBuffer::write"),
        });
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    pub(crate) fn pool_to_host(
        &mut self,
        src: &WGSLBuffer,
        dst: &mut [u8],
    ) -> Result<(), WGSLError> {
        futures::executor::block_on(async {
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
            let download = rx.await.unwrap().unwrap();
            dst.copy_from_slice(&download);
        });
        Ok(())
    }

    pub(crate) fn pool_to_pool(
        &mut self,
        src: &WGSLBuffer,
        dst: &WGSLBuffer,
    ) -> Result<(), WGSLError> {
        todo!()
    }
}

impl WGSLDevice {
    pub(crate) fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    // Memory pool id out of OpenCLMemoryPools
    pub(crate) fn memory_pool_id(&self) -> usize {
        self.memory_pool_id
    }

    pub(crate) fn compile(
        &mut self,
        kernel: &IRKernel,
        debug_asm: bool,
    ) -> Result<WGSLProgram, WGSLError> {
        let mut source = String::new();
        let mut indent = String::from("  ");

        let mut global_work_size = [0; 3];
        let mut local_work_size = [0; 3];

        /*
        @group(0) @binding(0) var<storage, read> g0: array<f32>;
        @group(0) @binding(1) var<storage, read> g1: array<f32>;
        @group(0) @binding(2) var<storage, read_write> g2: array<f32>;
        */

        for op in &kernel.ops[..6] {
            if let &IROp::Loop { id, len } = op {
                if id % 2 == 0 {
                    global_work_size[id as usize / 2] = len;
                } else {
                    local_work_size[id as usize / 2] = len;
                }
            } else {
                panic!()
            }
        }

        // Declare global variables
        for (id, (_, dtype, read_only)) in kernel.addressables.iter().enumerate() {
            source += &format!(
                "@group(0) @binding({id}) var<storage, {}> g{id}: array<{}>;\n",
                if *read_only { "read" } else { "read_write" },
                dtype.wgsl(),
            );
        }

        // NOTE Just gonna assume wgsl workgroup size is local work size
        source += &format!(
            "@compute @workgroup_size({}, {}, {})",
            local_work_size[0], local_work_size[1], local_work_size[2]
        );
        source += &format!("fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{\n");

        // Declare register variables
        for (id, (dtype, read_only)) in kernel.registers.iter().enumerate() {
            source += &format!(
                "{indent}{} r{id}: {};\n",
                if *read_only { "let" } else { "var" },
                dtype.wgsl()
            );
        }

        // Add indices for global and local loops
        source += &format!("  r0 = gid.x;   /* 0..{} */\n", global_work_size[0]);
        source += &format!("  r1 = lid.x;   /* 0..{} */\n", local_work_size[0]);
        source += &format!("  r2 = gid.y;   /* 0..{} */\n", global_work_size[1]);
        source += &format!("  r3 = lid.y;   /* 0..{} */\n", local_work_size[1]);
        source += &format!("  r4 = gid.z;   /* 0..{} */\n", global_work_size[2]);
        source += &format!("  r5 = lid.z;   /* 0..{} */\n", local_work_size[2]);

        for op in kernel.ops[6..kernel.ops.len() - 6].iter().copied() {
            match op {
                IROp::Set { z, len: _, value } => {
                    source += &format!("{indent}r{z} = {value};\n");
                }
                IROp::Load { z, x, at, dtype: _ } => {
                    source += &format!("{indent}{} = {}[{}];\n", z.wgsl(), x.wgsl(), at.wgsl());
                }
                IROp::Store { z, x, at, dtype: _ } => {
                    source += &format!("{indent}{}[{}] = {};\n", z.wgsl(), at.wgsl(), x.wgsl());
                }
                IROp::Unary { z, x, uop, dtype } => {
                    source += &match uop {
                        UOp::Cast(_) => {
                            format!("{indent}{} = ({}){};\n", z.wgsl(), dtype.wgsl(), x.wgsl())
                        }
                        UOp::ReLU => format!("{indent}{} = max({}, 0);\n", z.wgsl(), x.wgsl()),
                        UOp::Neg => format!("{indent}{} = -{};\n", z.wgsl(), x.wgsl()),
                        UOp::Exp2 => format!("{indent}{} = exp2({});\n", z.wgsl(), x.wgsl()),
                        UOp::Log2 => format!("{indent}{} = log2({});\n", z.wgsl(), x.wgsl()),
                        UOp::Inv => format!("{indent}{} = 1/{};\n", z.wgsl(), x.wgsl()),
                        UOp::Sqrt => format!("{indent}{} = sqrt({});\n", z.wgsl(), x.wgsl()),
                        UOp::Sin => format!("{indent}{} = sin({});\n", z.wgsl(), x.wgsl()),
                        UOp::Cos => format!("{indent}{} = cos({});\n", z.wgsl(), x.wgsl()),
                        UOp::Not => format!("{indent}{} = !{};\n", z.wgsl(), x.wgsl()),
                        UOp::Nonzero => format!("{indent}{} = {} != 0;\n", z.wgsl(), x.wgsl()),
                    };
                }
                IROp::Binary {
                    z,
                    x,
                    y,
                    bop,
                    dtype: _,
                } => {
                    source += &format!(
                        "{indent}{} = {};\n",
                        z.wgsl(),
                        match bop {
                            BOp::Add => format!("{} + {}", x.wgsl(), y.wgsl()),
                            BOp::Sub => format!("{} - {}", x.wgsl(), y.wgsl()),
                            BOp::Mul => format!("{} * {}", x.wgsl(), y.wgsl()),
                            BOp::Div => format!("{} / {}", x.wgsl(), y.wgsl()),
                            BOp::Pow => format!("pow({}, {})", x.wgsl(), y.wgsl()),
                            BOp::Cmplt => format!("{} < {}", x.wgsl(), y.wgsl()),
                            BOp::Cmpgt => format!("{} > {}", x.wgsl(), y.wgsl()),
                            BOp::Max => format!("max({}, {})", x.wgsl(), y.wgsl()),
                            BOp::Or => format!("{} || {}", x.wgsl(), y.wgsl()),
                        }
                    );
                }
                IROp::MAdd {
                    z,
                    a,
                    b,
                    c,
                    dtype: _,
                } => {
                    source += &format!(
                        "{indent}{} = {} * {} + {};\n",
                        z.wgsl(),
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
                IROp::Barrier { scope } => {
                    source += &format!(
                        "{indent}barrier(CLK_{}AL_MEM_FENCE);\n",
                        match scope {
                            Scope::Global => "GLOB",
                            Scope::Local => "LOC",
                            Scope::Register => panic!(),
                        }
                    );
                }
            }
        }
        source += "}\n";

        let mut global_work_size = global_work_size;
        let local_work_size = local_work_size;
        let name = format!(
            "k__{}_{}__{}_{}__{}_{}",
            global_work_size[0],
            local_work_size[0],
            global_work_size[1],
            local_work_size[1],
            global_work_size[2],
            local_work_size[2],
        );
        for (i, lwd) in local_work_size.iter().enumerate() {
            global_work_size[i] *= lwd;
        }
        let mut pragma = format!("");
        if source.contains("double") {
            pragma += &"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        }
        let source = format!("{pragma}__kernel void {name}{source}");
        if debug_asm {
            println!("{source}");
        }

        todo!()
    }
}

impl IRDType {
    fn wgsl(&self) -> &str {
        return match self {
            #[cfg(feature = "half")]
            IRDType::BF16 => todo!("WIP"),
            #[cfg(feature = "half")]
            IRDType::F16 => "f16",
            IRDType::F32 => "f32",
            IRDType::F64 => "f64",
            #[cfg(feature = "complex")]
            IRDType::CF32 => todo!("WIP"),
            #[cfg(feature = "complex")]
            IRDType::CF64 => todo!("WIP"),
            IRDType::U8 => "u8",
            IRDType::I8 => "i8",
            IRDType::I16 => "i16",
            IRDType::I32 => "int",
            IRDType::I64 => "i64",
            IRDType::Bool => "bool",
            IRDType::U32 => "u32",
        };
    }
}

impl WGSLQueue {
    pub(crate) fn launch(
        &mut self,
        program: &mut WGSLProgram,
        buffers: &mut IndexMap<WGSLBuffer>,
        args: &[usize],
    ) -> Result<(), WGSLError> {
        todo!()
    }

    pub(crate) fn sync(&mut self) -> Result<(), WGSLError> {
        self.load = 0;
        todo!()
    }

    pub(crate) fn load(&self) -> usize {
        self.load
    }
}

impl Constant {
    fn wgsl(&self) -> String {
        use core::mem::transmute as t;
        match self {
            #[cfg(feature = "half")]
            Constant::F16(x) => format!("{}f", unsafe { t::<_, half::f16>(*x) }),
            #[cfg(feature = "half")]
            Constant::BF16(x) => format!("{}f", unsafe { t::<_, half::bf16>(*x) }),
            Constant::F32(x) => format!("{}f", unsafe { t::<_, f32>(*x) }),
            Constant::F64(x) => format!("{}f", unsafe { t::<_, f64>(*x) }),
            #[cfg(feature = "complex")]
            Constant::CF32(..) => todo!("Complex numbers are currently not supported for OpenCL"),
            #[cfg(feature = "complex")]
            Constant::CF64(..) => todo!("Complex numbers are currently not supported for OpenCL"),
            Constant::U8(x) => format!("{x}"),
            Constant::I8(x) => format!("{x}"),
            Constant::I16(x) => format!("{x}"),
            Constant::U32(x) => format!("{x}"),
            Constant::I32(x) => format!("{x}"),
            Constant::I64(x) => format!("{x}"),
            Constant::Bool(x) => format!("{x}"),
        }
    }
}

impl Var {
    fn wgsl(&self) -> String {
        match self {
            Var::Id(id, scope) => format!("{scope}{id}"),
            Var::Const(value) => format!("{}", value.wgsl()),
        }
    }
}
