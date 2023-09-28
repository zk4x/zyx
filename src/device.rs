use crate::{
    node_id::NodeId, graph::Node,
{dtype::DType, shape::Shape, OutOfMemoryError}};
extern crate alloc;
use alloc::{
    boxed::Box,
collections::{BTreeMap, BTreeSet}};

pub(super) mod cpu;
#[cfg(feature = "opencl")]
pub(super) mod opencl;

#[derive(Debug)]
pub(crate) enum Storage {
    None,
    CPUF32(Box<[f32]>, Shape),
    CPUI32(Box<[i32]>, Shape),
    #[cfg(feature = "opencl")]
    OpenCLF32(Shape, opencl::ClStorage), // shape, buffer, event
    #[cfg(feature = "opencl")]
    OpenCLI32(Shape, opencl::ClStorage),
}

impl Storage {
    pub(super) fn dtype(&self) -> DType {
        match self {
            Self::None => panic!(),
            Self::CPUF32(..) => DType::F32,
            Self::CPUI32(..) => DType::I32,
            #[cfg(feature = "opencl")]
            Self::OpenCLF32(..) => DType::F32,
            #[cfg(feature = "opencl")]
            Self::OpenCLI32(..) => DType::I32,
        }
    }
}

impl Storage {
    pub(super) fn shape(&self) -> &Shape {
        match self {
            Self::None => panic!(),
            Self::CPUF32(_, shape, ..) => shape,
            Self::CPUI32(_, shape, ..) => shape,
            #[cfg(feature = "opencl")]
            Self::OpenCLF32(shape, ..) => shape,
            #[cfg(feature = "opencl")]
            Self::OpenCLI32(shape, ..) => shape,
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub(crate) enum Device {
    CPU,
    #[cfg(feature = "opencl")]
    OpenCL(opencl::OpenCLDev),
}

impl Device {
    #[cfg(feature = "opencl")]
    pub(crate) fn opencl() -> Result<Self, cl3::error_codes::ClError> {
        Ok(Self::OpenCL(opencl::OpenCLDev::new()?))
    }

    #[allow(clippy::unused_self)]
    pub(crate) fn load_f32(&mut self, storage: &Storage) -> Box<[f32]> {
        match storage {
            Storage::None => panic!(),
            Storage::CPUF32(data, ..) => data.clone(),
            Storage::CPUI32(..) => panic!("Trying to load i32 tensor as if it was f32 tensor"),
            #[cfg(feature = "opencl")]
            Storage::OpenCLF32(shape, storage) => {
                let mut data: Box<[f32]> = core::iter::repeat(0f32).take(shape.numel()).collect();
                if let Device::OpenCL(dev) = self {
                    let queue = dev.queue();
                    let event = storage.event();
                    // TODO this should be in the event wait list
                    cl3::event::wait_for_events(&[event]).unwrap();
                    let event = unsafe {
                        cl3::command_queue::enqueue_read_buffer(
                            queue,
                            storage.buffer(),
                            cl3::types::CL_NON_BLOCKING,
                            0,
                            shape.numel() * core::mem::size_of::<f32>(),
                            data.as_mut_ptr().cast(),
                            0,
                            core::ptr::null_mut(),
                        )
                    }
                    .unwrap();
                    //cl3::command_queue::finish(queue).unwrap();
                    cl3::event::wait_for_events(&[event]).unwrap();
                    data
                } else {
                    panic!("Trying to access OpenCL tensor using {:?} device", self);
                }
            }
            #[cfg(feature = "opencl")]
            Storage::OpenCLI32(..) => panic!("Trying to load i32 tensor as if it was f32 tensor"),
        }
    }

    #[allow(clippy::unused_self)]
    pub(crate) fn load_i32(&mut self, storage: &Storage) -> Box<[i32]> {
        match storage {
            Storage::None => panic!(),
            Storage::CPUF32(..) => panic!("Trying to load f32 tensor as if it was i32 tensor"),
            Storage::CPUI32(data, ..) => data.clone(),
            #[cfg(feature = "opencl")]
            Storage::OpenCLF32(..) => panic!("Trying to load f32 tensor as if it was i32 tensor"),
            #[cfg(feature = "opencl")]
            Storage::OpenCLI32(shape, storage) => {
                let mut data: Box<[i32]> = core::iter::repeat(0i32).take(shape.numel()).collect();
                if let Device::OpenCL(dev) = self {
                    let queue = dev.queue();
                    let event = storage.event();
                    // TODO this should be in the event wait list
                    cl3::event::wait_for_events(&[event]).unwrap();
                    let event = unsafe {
                        cl3::command_queue::enqueue_read_buffer(
                            queue,
                            storage.buffer(),
                            cl3::types::CL_NON_BLOCKING,
                            0,
                            shape.numel() * core::mem::size_of::<i32>(),
                            data.as_mut_ptr().cast(),
                            0,
                            core::ptr::null_mut(),
                        )
                    }
                    .unwrap();
                    //cl3::command_queue::finish(queue).unwrap();
                    cl3::event::wait_for_events(&[event]).unwrap();
                    data
                } else {
                    panic!("Trying to access OpenCL tensor using {:?} device", self);
                }
            }
        }
    }

    pub(super) fn realize(
        &mut self,
        graph: &mut BTreeMap<NodeId, (usize, Node)>,
        order: &[NodeId],
        nodes: &BTreeSet<NodeId>,
    ) -> Result<(), OutOfMemoryError> {
        match self {
            Device::CPU => cpu::realize(graph, order, nodes)?,
            #[cfg(feature = "opencl")]
            Device::OpenCL(dev) => dev.realize(graph, order, nodes)?,
        }
        Ok(())
    }
}
