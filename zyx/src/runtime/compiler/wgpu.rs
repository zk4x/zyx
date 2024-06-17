use crate::runtime::compiler::ir::IRKernel;
use crate::runtime::compiler::{Compiler, CompilerError, HWInfo};
use alloc::vec::Vec;
use wgpu::{
    Adapter, Backends, Device, DeviceDescriptor, Dx12Compiler, Gles3MinorVersion, Instance,
    InstanceDescriptor, InstanceFlags, RequestAdapterOptions,
};

pub(crate) struct WGPU {
    instance: Instance,
    adapter: Adapter,
    device: Device,
}

impl Compiler for WGPU {
    type Buffer = ();
    type Program = ();

    fn initialize() -> Result<Self, CompilerError> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::VULKAN,
            flags: InstanceFlags::empty(),
            dx12_shader_compiler: Dx12Compiler::Fxc,
            gles_minor_version: Gles3MinorVersion::Automatic,
        });

        // For now we will support only single device
        let adapter = instance.enumerate_adapters(Backends::VULKAN).pop().ok_or(
            CompilerError::InitializationFailure("No devices found in WGPU backend."),
        )?;

        let device = adapter.create_device_from_hal();

        return Ok(Self { instance, adapter, device });
    }

    fn hardware_information(&mut self) -> Result<HWInfo, CompilerError> {
        todo!()
    }

    fn allocate_memory(&mut self, byte_size: usize) -> Result<Self::Buffer, CompilerError> {
        let _ = byte_size;
        todo!()
    }

    fn store_memory<T>(
        &mut self,
        buffer: &mut Self::Buffer,
        data: &[T],
    ) -> Result<(), CompilerError> {
        let _ = buffer;
        let _ = data;
        todo!()
    }

    fn load_memory<T>(
        &mut self,
        buffer: &Self::Buffer,
        length: usize,
    ) -> Result<Vec<T>, CompilerError> {
        let _ = buffer;
        let _ = length;
        todo!()
    }

    fn deallocate_memory(&mut self, buffer: Self::Buffer) -> Result<(), CompilerError> {
        let _ = buffer;
        todo!()
    }

    fn compile_program(&mut self, kernel: &IRKernel) -> Result<Self::Program, CompilerError> {
        let _ = kernel;
        todo!()
    }

    fn launch_program(
        &mut self,
        program: &Self::Program,
        args: &mut [Self::Buffer],
    ) -> Result<(), CompilerError> {
        let _ = program;
        let _ = args;
        todo!()
    }

    fn release_program(&mut self, program: Self::Program) -> Result<(), CompilerError> {
        let _ = program;
        todo!()
    }
}
