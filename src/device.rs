use crate::{
    graph::Node,
    node_id::NodeId,
    {dtype::DType, shape::Shape, OutOfMemoryError},
};
extern crate alloc;
use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
};

use self::cpu::CpuStorage;

pub(super) mod cpu;
#[cfg(feature = "opencl")]
pub(super) mod opencl;
#[cfg(feature = "torch")]
pub(super) mod torch;

#[derive(Debug)]
pub(crate) enum Storage {
    CPUF32(CpuStorage<f32>),
    CPUI32(CpuStorage<i32>),
    #[cfg(feature = "opencl")]
    OpenCLF32(Shape, opencl::ClStorage), // shape, buffer, event
    #[cfg(feature = "opencl")]
    OpenCLI32(Shape, opencl::ClStorage),
    #[cfg(feature = "torch")]
    TorchF32(tch::Tensor),
    #[cfg(feature = "torch")]
    TorchI32(tch::Tensor),
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
            #[cfg(feature = "torch")]
            Self::TorchF32(..) => DType::F32,
            #[cfg(feature = "torch")]
            Self::TorchI32(..) => DType::I32,
        }
    }
}

impl Storage {
    pub(super) fn shape(&self) -> Shape {
        match self {
            Self::CPUF32(data) => data.shape().clone(),
            Self::CPUI32(data) => data.shape().clone(),
            #[cfg(feature = "opencl")]
            Self::OpenCLF32(shape, ..) | Self::OpenCLI32(shape, ..) => shape.clone(),
            #[cfg(feature = "torch")]
            Self::TorchF32(data) | Self::TorchI32(data) => data.size().into_iter().map(|x| x as usize).collect::<Box<[usize]>>().into(),
        }
    }
}

#[derive(Debug)]
pub(crate) enum Device {
    Cpu(cpu::CpuDev),
    #[cfg(feature = "opencl")]
    OpenCL(opencl::OpenCLDev),
    #[cfg(feature = "torch")]
    Torch(torch::TorchDev),
}

impl Device {
    #[cfg(feature = "opencl")]
    pub(crate) fn opencl() -> Result<Self, cl3::error_codes::ClError> {
        Ok(Self::OpenCL(opencl::OpenCLDev::new()?))
    }

    #[cfg(feature = "torch")]
    pub(crate) fn torch() -> Self {
        Self::Torch(torch::TorchDev::new())
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
            #[cfg(feature = "torch")]
            Storage::TorchF32(data) =>
                if let Device::Torch(dev) = self {
                    dev.load_f32(data)
                } else {
                    panic!("Trying to access Torch tensor using {:?} device", self);
                }
            #[cfg(feature = "torch")]
            Storage::TorchI32(..) => panic!("Trying to load i32 tensor as if it was f32 tensor"),
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
            #[cfg(feature = "torch")]
            Storage::TorchF32(..) => panic!("Trying to load f32 tensor as if it was i32 tensor"),
            #[cfg(feature = "torch")]
            Storage::TorchI32(data) =>
                if let Device::Torch(dev) = self {
                    dev.load_i32(data)
                } else {
                    panic!("Trying to access Torch tensor using {:?} device", self);
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
            Device::Cpu(dev) => dev.realize(graph, order, nodes)?,
            #[cfg(feature = "opencl")]
            Device::OpenCL(dev) => dev.realize(graph, order, nodes)?,
            #[cfg(feature = "torch")]
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
