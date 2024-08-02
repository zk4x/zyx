use crate::DType;
use crate::Scalar;

use super::{Compiler, HWInfo};
use alloc::{boxed::Box, vec::Vec, vec, string::String};
use wgpu::{
    Backends, BufferDescriptor, BufferSize, BufferUsages, Device, DeviceDescriptor, Dx12Compiler,
    Features, Gles3MinorVersion, Instance, InstanceDescriptor, InstanceFlags, Limits, Queue,
};

use super::IRKernel;

#[derive(Debug)]
pub struct WGSLError {
    info: String,
}

pub(crate) struct WGSLBuffer {
    memory: wgpu::Buffer,
}

pub(crate) struct WGSLProgram {}

pub(crate) struct WGSLRuntime {
    instance: Instance,
    //adapter: Adapter,
    device: Device,
    queues: Box<[Queue]>,
    queue_size: Box<[u8]>,
    queue_id: usize,
}

impl WGSLRuntime {
    fn queue(&mut self) -> Result<&Queue, WGSLError> {
        let res = &self.queues[self.queue_id];
        self.queue_size[self.queue_id] += 1;
        // Blocks and waits for queue to finish execution so that
        // we do not overwhelm the device with tasks.
        // Up to two events per queue.
        /*if self.queue_size[self.queue_id] == 2 {
        queu
        let status = unsafe { clFinish(res) };
        handle_status(
            status,
            "Unable to finish execution of command queue.",
            &[-36, -5, -6],
        )?;
        self.queue_size[self.queue_id] = 0;
        }*/
        self.queue_id = (self.queue_id + 1) % self.queues.len();
        return Ok(res);
    }
}

impl Compiler for WGSLRuntime {
    type Buffer = WGSLBuffer;
    type Program = WGSLProgram;
    type Error = WGSLError;

    fn initialize() -> Result<Self, WGSLError> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::VULKAN,
            flags: InstanceFlags::empty(),
            dx12_shader_compiler: Dx12Compiler::Fxc,
            gles_minor_version: Gles3MinorVersion::Automatic,
        });

        // For now we will support only single device
        let adapter = instance.enumerate_adapters(Backends::VULKAN).pop().ok_or(
            WGSLError { info: "No adapters found in WGSL backend.".into() },
        )?;
        let runtime = nostd_async::Runtime::new();
        let mut task = nostd_async::Task::new(adapter.request_device(
            &DeviceDescriptor {
                label: None,
                required_features: Features::default(),
                required_limits: Limits::downlevel_defaults(),
            },
            None,
        ));
        let handle = task.spawn(&runtime);
        let (device, queue) = handle.join().map_err(|_| {
            WGSLError { info: "No devices found in WGSL backend.".into() }
        })?;

        return Ok(Self {
            instance,
            //adapter,
            device,
            queue_size: alloc::vec![0; 1].into_boxed_slice(),
            queues: [queue].into(),
            queue_id: 0,
        });
    }

    fn hardware_information(&mut self) -> Result<HWInfo, WGSLError> {
        return Ok(HWInfo {
            max_work_item_sizes: vec![1024, 1024, 1024],
            max_work_group_size: 256,
            preferred_vector_size: 4,
            f16_support: true,
            f64_support: true,
            fmadd: true,
            global_mem_size: 2 * 1024 * 1024 * 1024,
            max_mem_alloc: 512 * 1024 * 1024,
            mem_align: 1024,
            page_size: 1024,
            local_mem_size: 1024 * 1024,
            num_registers: 96,
            native_mm16x16_support: false,
        });
    }

    fn allocate_memory(&mut self, byte_size: usize) -> Result<Self::Buffer, WGSLError> {
        Ok(WGSLBuffer {
            memory: self.device.create_buffer(&BufferDescriptor {
                label: None,
                size: byte_size as u64,
                usage: BufferUsages::MAP_READ | BufferUsages::MAP_WRITE,
                mapped_at_creation: false,
            }),
        })
    }

    fn store_memory<T: Scalar>(
        &mut self,
        buffer: &mut Self::Buffer,
        data: Vec<T>,
    ) -> Result<(), WGSLError> {
        let queue = self.queue()?;
        let mut view = queue
            .write_buffer_with(
                &buffer.memory,
                0,
                BufferSize::new((data.len() * T::byte_size()) as u64).unwrap(),
            )
            .unwrap();
        match T::dtype() {
            DType::F32 => {
                for i in 0..data.len() {
                    let [a, b, c, d] = data[i].cast::<f32>().to_ne_bytes();
                    view.as_mut()[i * 4] = a;
                    view.as_mut()[i * 4 + 1] = b;
                    view.as_mut()[i * 4 + 2] = c;
                    view.as_mut()[i * 4 + 3] = d;
                }
            }
            _ => {}
        }
        //queue.submit(command_buffers);
        Ok(())
    }

    fn load_memory<T: Scalar>(
        &mut self,
        buffer: &Self::Buffer,
        length: usize,
    ) -> Result<Vec<T>, WGSLError> {
        let _ = buffer;
        let _ = length;
        todo!()
    }

    fn deallocate_memory(&mut self, buffer: Self::Buffer) -> Result<(), WGSLError> {
        let _ = buffer;
        todo!()
    }

    fn compile_program(&mut self, kernel: &IRKernel) -> Result<Self::Program, WGSLError> {
        let _ = kernel;
        todo!()
    }

    fn launch_program(
        &mut self,
        program: &Self::Program,
        args: &mut [Self::Buffer],
    ) -> Result<(), WGSLError> {
        let _ = program;
        let _ = args;
        todo!()
    }

    fn release_program(&mut self, program: Self::Program) -> Result<(), WGSLError> {
        let _ = program;
        todo!()
    }
}
