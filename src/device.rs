use crate::{
    node_id::NodeId, graph::Node,
{dtype::DType, shape::Shape, OutOfMemoryError}};
extern crate alloc;
use alloc::{
    boxed::Box,
collections::{BTreeMap, BTreeSet}};

use self::{cpu::CpuStorage, torch::TorchStorage};

pub(super) mod cpu;
#[cfg(feature = "opencl")]
pub(super) mod opencl;
pub(super) mod torch;

#[derive(Debug)]
pub(crate) enum Storage {
    CPUF32(CpuStorage<f32>),
    CPUI32(CpuStorage<i32>),
    #[cfg(feature = "opencl")]
    OpenCLF32(Shape, opencl::ClStorage), // shape, buffer, event
    #[cfg(feature = "opencl")]
    OpenCLI32(Shape, opencl::ClStorage),
    TorchF32(TorchStorage<i32>),
}

impl Storage {
    pub(super) fn dtype(&self) -> DType {
        match self {
            Self::CPUF32(..) => DType::F32,
            Self::CPUI32(..) => DType::I32,
            #[cfg(feature = "opencl")]
            Self::OpenCLF32(..) => DType::F32,
            #[cfg(feature = "opencl")]
            Self::OpenCLI32(..) => DType::I32,
            Self::TorchF32(..) => DType::F32,
        }
    }
}

impl Storage {
    pub(super) fn shape(&self) -> &Shape {
        match self {
            Self::CPUF32(data) => data.shape(),
            Self::CPUI32(data) => data.shape(),
            #[cfg(feature = "opencl")]
            Self::OpenCLF32(shape, ..) => shape,
            #[cfg(feature = "opencl")]
            Self::OpenCLI32(shape, ..) => shape,
            Self::TorchF32(data) => data.shape(),
        }
    }
}

#[derive(Debug)]
pub(crate) enum Device {
    Cpu(cpu::CpuDev),
    #[cfg(feature = "opencl")]
    OpenCL(opencl::OpenCLDev),
    Torch(torch::TorchDev),
}

impl Device {
    #[cfg(feature = "opencl")]
    pub(crate) fn opencl() -> Result<Self, cl3::error_codes::ClError> {
        Ok(Self::OpenCL(opencl::OpenCLDev::new()?))
    }

    // TODO join these two together
    #[allow(clippy::unused_self)]
    pub(crate) fn load_f32(&mut self, storage: &Storage) -> Box<[f32]> {
        match storage {
            Storage::CPUF32(data) => (0..data.shape().numel()).map(|i| data.at(i)).collect(),
            Storage::CPUI32(..) => panic!("Trying to load i32 tensor as if it was f32 tensor"),
            #[cfg(feature = "opencl")]
            Storage::OpenCLF32(shape, storage) => {
                if let Device::OpenCL(dev) = self {
                    dev.load(storage, shape)
                } else {
                    panic!("Trying to access OpenCL tensor using {:?} device", self);
                }
            }
            #[cfg(feature = "opencl")]
            Storage::OpenCLI32(..) => panic!("Trying to load i32 tensor as if it was f32 tensor"),
            Storage::TorchF32(data) => todo!(),
        }
    }

    #[allow(clippy::unused_self)]
    pub(crate) fn load_i32(&mut self, storage: &Storage) -> Box<[i32]> {
        match storage {
            Storage::CPUF32(..) => panic!("Trying to load f32 tensor as if it was i32 tensor"),
            Storage::CPUI32(data) => (0..data.shape().numel()).map(|i| data.at(i)).collect(),
            #[cfg(feature = "opencl")]
            Storage::OpenCLF32(..) => panic!("Trying to load f32 tensor as if it was i32 tensor"),
            #[cfg(feature = "opencl")]
            Storage::OpenCLI32(shape, storage) => {
                if let Device::OpenCL(dev) = self {
                    dev.load(storage, shape)
                } else {
                    panic!("Trying to access OpenCL tensor using {:?} device", self);
                }
            }
            Storage::TorchF32(..) => panic!("Trying to load f32 tensor as if it was i32 tensor"),
        }
    }

    pub(super) fn realize(
        &mut self,
        graph: &mut BTreeMap<NodeId, (usize, Node)>,
        order: &[NodeId],
        nodes: &BTreeSet<NodeId>,
    ) -> Result<(), OutOfMemoryError> {
        match self {
            Device::Cpu(dev) => dev.realize(graph, order, nodes)?,
            #[cfg(feature = "opencl")]
            Device::OpenCL(dev) => dev.realize(graph, order, nodes)?,
            Device::Torch(dev) => dev.realize(graph, order, nodes)?,
        }
        Ok(())
    }
}

trait Dtype:
    Clone
    + core::fmt::Debug
    + core::fmt::Display
    + core::ops::Add<Output = Self>
    + core::ops::Mul<Output = Self>
    + Sync
    + Send
    + core::iter::Sum
{
    fn dtype() -> DType;
    fn zero() -> Self;
}

impl Dtype for f32 {
    fn dtype() -> DType {
        DType::F32
    }

    fn zero() -> Self {
        0.
    }
}

impl Dtype for i32 {
    fn dtype() -> DType {
        DType::I32
    }

    fn zero() -> Self {
        0
    }
}
