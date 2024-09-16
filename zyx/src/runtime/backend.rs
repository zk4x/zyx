//! This file creates backend agnostic API to backends
//! That is it contains enums that dispatch function calls to appropriate backends.

// Because I don't want to write struct and inner enum for MemoryPool and Device
#![allow(private_interfaces)]

use super::{ir::IRKernel, DeviceConfig, Runtime, ZyxError};
use crate::{index_map::IndexMap, Scalar};
use cuda::{CUDABuffer, CUDADevice, CUDAMemoryPool, CUDAProgram, CUDAQueue};
use hip::{HIPBuffer, HIPDevice, HIPMemoryPool, HIPProgram, HIPQueue};
use opencl::{OpenCLBuffer, OpenCLDevice, OpenCLMemoryPool, OpenCLProgram, OpenCLQueue};

#[cfg(feature = "wgsl")]
use wgsl::{WGSLBuffer, WGSLDevice, WGSLMemoryPool, WGSLProgram, WGSLQueue};

mod cuda;
mod hip;
mod opencl;
#[cfg(feature = "wgsl")]
mod wgsl;

// Export configs and errors, nothing more
pub use cuda::{CUDAConfig, CUDAError};
pub use hip::{HIPConfig, HIPError};
pub use opencl::{OpenCLConfig, OpenCLError};
#[cfg(feature = "wgsl")]
pub use wgsl::{WGSLConfig, WGSLError};

/// Hardware information needed for applying optimizations
#[derive(
    Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, bitcode::Encode, bitcode::Decode,
)]
pub(super) struct DeviceInfo {
    /// Device compute in flops
    pub compute: u128,
    /// Biggest kernel dimensions
    pub max_global_work_dims: [usize; 3],
    /// Maximum local work size threads
    pub max_local_threads: usize,
    pub max_local_work_dims: [usize; 3],
    /// Preferred vector size in bytes
    pub preferred_vector_size: usize,
    /// Local memory size in bytes
    pub local_mem_size: usize,
    /// Number of registers per thread
    pub num_registers: usize,
    /// Does this hardware have tensor cores?
    pub tensor_cores: bool,
}

pub(super) type MemoryPoolId = usize;
pub(super) type DeviceId = usize;

pub(super) enum MemoryPool {
    CUDA {
        memory_pool: CUDAMemoryPool,
        buffers: IndexMap<CUDABuffer>,
    },
    HIP {
        memory_pool: HIPMemoryPool,
        buffers: IndexMap<HIPBuffer>,
    },
    OpenCL {
        memory_pool: OpenCLMemoryPool,
        buffers: IndexMap<OpenCLBuffer>,
    },
    #[cfg(feature = "wgsl")]
    WGSL {
        memory_pool: WGSLMemoryPool,
        buffers: IndexMap<WGSLBuffer>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct BufferId {
    pub(super) memory_pool_id: usize,
    pub(super) buffer_id: usize,
}

#[derive(Debug)]
pub(super) enum Device {
    CUDA {
        memory_pool_id: MemoryPoolId,
        device: CUDADevice,
        programs: IndexMap<CUDAProgram>,
        queues: Vec<CUDAQueue>,
    },
    HIP {
        memory_pool_id: MemoryPoolId,
        device: HIPDevice,
        programs: IndexMap<HIPProgram>,
        queues: Vec<HIPQueue>,
    },
    OpenCL {
        memory_pool_id: MemoryPoolId,
        device: OpenCLDevice,
        programs: IndexMap<OpenCLProgram>,
        queues: Vec<OpenCLQueue>,
    },
    #[cfg(feature = "wgsl")]
    WGSL {
        memory_pool_id: MemoryPoolId,
        device: WGSLDevice,
        programs: IndexMap<WGSLProgram>,
        queues: Vec<WGSLQueue>,
    },
}

impl Runtime {
    // Initializes all available devices, creating a device for each compute
    // device and a memory pool for each physical memory.
    // Does nothing if devices were already initialized.
    // Returns error if all devices failed to initialize
    // DeviceParameters allows to disable some devices if requested
    pub(super) fn initialize_devices(&mut self) -> Result<(), ZyxError> {
        if !self.devices.is_empty() {
            return Ok(());
        }

        // Set env vars
        if let Ok(x) = std::env::var("ZYX_DEBUG") {
            if let Ok(x) = x.parse::<u32>() {
                self.debug = x;
            }
        }

        // ZYX_SEARCH is number of variations of one kernel that will be tried
        // during each run of the program. Timings are cached to disk,
        // so rerunning the same kernels will continue the search where it left of.
        if let Ok(x) = std::env::var("ZYX_SEARCH") {
            if let Ok(x) = x.parse() {
                self.search_iterations = x;
            }
        }

        // Search through config directories and find zyx/backend_config.json
        // If not found or failed to parse, use defaults.
        let device_config = xdg::BaseDirectories::new()
            .map_err(|e| {
                if self.debug_dev() {
                    println!("Failed to find config directories for backend_config.json, {e}");
                }
            })
            .ok()
            .map(|bd| {
                let mut dirs = bd.get_config_dirs();
                dirs.push(bd.get_config_home());
                dirs
            })
            .map(|paths| {
                paths.into_iter().find_map(|mut path| {
                    path.push("zyx/backend_config.json");
                    if let Ok(file) = std::fs::read_to_string(&path) {
                        path.pop();
                        self.config_dir = Some(path);
                        Some(file)
                    } else {
                        None
                    }
                })
            })
            .flatten()
            .map(|file| {
                serde_json::from_str(&file)
                    .map_err(|e| {
                        if self.debug_dev() {
                            println!("Failed to parse backend_config.json, {e}");
                        }
                    })
                    .ok()
            })
            .flatten()
            .map(|x| {
                if self.debug_dev() {
                    println!("Backend config successfully read and parsed.");
                }
                x
            })
            .unwrap_or_else(|| {
                if self.debug_dev() {
                    println!("Failed to get backend config, using defaults.");
                }
                DeviceConfig::default()
            });

        if let Ok((memory_pools, devices)) =
            cuda::initialize_devices(&device_config.cuda, self.debug_dev())
        {
            let n = self.memory_pools.len();
            self.memory_pools
                .extend(memory_pools.into_iter().map(|m| MemoryPool::CUDA {
                    memory_pool: m,
                    buffers: IndexMap::new(),
                }));
            self.devices
                .extend(devices.into_iter().map(|(device, queues)| Device::CUDA {
                    memory_pool_id: device.memory_pool_id() + n,
                    device,
                    programs: IndexMap::new(),
                    queues,
                }));
        }
        if let Ok((memory_pools, devices)) =
            hip::initialize_device(&device_config.hip, self.debug_dev())
        {
            let n = self.memory_pools.len();
            self.memory_pools
                .extend(memory_pools.into_iter().map(|m| MemoryPool::HIP {
                    memory_pool: m,
                    buffers: IndexMap::new(),
                }));
            self.devices
                .extend(devices.into_iter().map(|(device, queues)| Device::HIP {
                    memory_pool_id: device.memory_pool_id() + n,
                    device,
                    programs: IndexMap::new(),
                    queues,
                }));
        }
        if let Ok((memory_pools, devices)) =
            opencl::initialize_devices(&device_config.opencl, self.debug_dev())
        {
            let n = self.memory_pools.len();
            self.memory_pools
                .extend(memory_pools.into_iter().map(|m| MemoryPool::OpenCL {
                    memory_pool: m,
                    buffers: IndexMap::new(),
                }));
            self.devices
                .extend(devices.into_iter().map(|(device, queues)| Device::OpenCL {
                    memory_pool_id: device.memory_pool_id() + n,
                    device,
                    programs: IndexMap::new(),
                    queues,
                }));
        }
        #[cfg(feature = "wgsl")]
        if let Ok((memory_pools, devices)) =
            wgsl::initialize_backend(&device_config.wgsl, self.debug_dev())
        {
            let n = self.memory_pools.len();
            self.memory_pools
                .extend(memory_pools.into_iter().map(|m| MemoryPool::WGSL {
                    memory_pool: m,
                    buffers: IndexMap::new(),
                }));
            self.devices
                .extend(devices.into_iter().map(|(device, queues)| Device::WGSL {
                    memory_pool_id: device.memory_pool_id() + n,
                    device,
                    programs: IndexMap::new(),
                    queues,
                }));
        }
        if self.devices.is_empty() {
            return Err(ZyxError::NoBackendAvailable);
        }
        Ok(())
    }
}

impl MemoryPool {
    pub(super) fn free_bytes(&self) -> usize {
        match self {
            MemoryPool::CUDA { memory_pool, .. } => memory_pool.free_bytes(),
            MemoryPool::HIP { memory_pool, .. } => memory_pool.free_bytes(),
            MemoryPool::OpenCL { memory_pool, .. } => memory_pool.free_bytes(),
            #[cfg(feature = "wgsl")]
            MemoryPool::WGSL { memory_pool, .. } => memory_pool.free_bytes(),
        }
    }

    // Allocates bytes on memory pool and returns buffer id
    pub(super) fn allocate(&mut self, bytes: usize) -> Result<usize, ZyxError> {
        let id = match self {
            MemoryPool::CUDA {
                memory_pool,
                buffers,
            } => buffers.push(memory_pool.allocate(bytes)?),
            MemoryPool::HIP {
                memory_pool,
                buffers,
            } => buffers.push(memory_pool.allocate(bytes)?),
            MemoryPool::OpenCL {
                memory_pool,
                buffers,
            } => buffers.push(memory_pool.allocate(bytes)?),
            #[cfg(feature = "wgsl")]
            MemoryPool::WGSL {
                memory_pool,
                buffers,
            } => buffers.push(memory_pool.allocate(bytes)?),
        };
        //println!("Allocate {bytes} bytes into buffer id {id}");
        Ok(id)
    }

    pub(super) fn deallocate(&mut self, buffer_id: usize) -> Result<(), ZyxError> {
        //println!("Deallocate buffer id {buffer_id}");
        match self {
            MemoryPool::CUDA {
                memory_pool,
                buffers,
            } => {
                let buffer = buffers.remove(buffer_id).unwrap();
                memory_pool.deallocate(buffer)?;
            }
            MemoryPool::HIP {
                memory_pool,
                buffers,
            } => {
                let buffer = buffers.remove(buffer_id).unwrap();
                memory_pool.deallocate(buffer)?;
            }
            MemoryPool::OpenCL {
                memory_pool,
                buffers,
            } => {
                let buffer = buffers.remove(buffer_id).unwrap();
                memory_pool.deallocate(buffer)?;
            }
            #[cfg(feature = "wgsl")]
            MemoryPool::WGSL {
                memory_pool,
                buffers,
            } => {
                let buffer = buffers.remove(buffer_id).unwrap();
                memory_pool.deallocate(buffer)?;
            }
        }
        Ok(())
    }

    pub(super) fn host_to_pool<T: Scalar>(
        &mut self,
        data: &[T],
        buffer_id: usize,
    ) -> Result<(), ZyxError> {
        let bytes = data.len() * T::byte_size();
        match self {
            MemoryPool::CUDA {
                memory_pool,
                buffers,
            } => {
                let ptr: *const u8 = data.as_ptr().cast();
                memory_pool.host_to_pool(
                    unsafe { std::slice::from_raw_parts(ptr, bytes) },
                    &mut buffers[buffer_id],
                )?;
            }
            MemoryPool::HIP {
                memory_pool,
                buffers,
            } => {
                let ptr: *const u8 = data.as_ptr().cast();
                memory_pool.host_to_pool(
                    unsafe { std::slice::from_raw_parts(ptr, bytes) },
                    &mut buffers[buffer_id],
                )?;
            }
            MemoryPool::OpenCL {
                memory_pool,
                buffers,
            } => {
                let ptr: *const u8 = data.as_ptr().cast();
                memory_pool.host_to_pool(
                    unsafe { std::slice::from_raw_parts(ptr, bytes) },
                    &mut buffers[buffer_id],
                )?;
            }
            #[cfg(feature = "wgsl")]
            MemoryPool::WGSL {
                memory_pool,
                buffers,
            } => {
                let ptr: *const u8 = data.as_ptr().cast();
                memory_pool.host_to_pool(
                    unsafe { std::slice::from_raw_parts(ptr, bytes) },
                    &mut buffers[buffer_id],
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn pool_to_host<T: Scalar>(
        &mut self,
        buffer_id: usize,
        data: &mut [T],
    ) -> Result<(), ZyxError> {
        let slice = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), data.len() * T::byte_size())
        };
        match self {
            MemoryPool::CUDA {
                memory_pool,
                buffers,
            } => {
                memory_pool.pool_to_host(&buffers[buffer_id], slice)?;
            }
            MemoryPool::HIP {
                memory_pool,
                buffers,
            } => {
                memory_pool.pool_to_host(&buffers[buffer_id], slice)?;
            }
            MemoryPool::OpenCL {
                memory_pool,
                buffers,
            } => {
                memory_pool.pool_to_host(&buffers[buffer_id], slice)?;
            }
            #[cfg(feature = "wgsl")]
            MemoryPool::WGSL {
                memory_pool,
                buffers,
            } => {
                memory_pool.pool_to_host(&buffers[buffer_id], slice)?;
            }
        }
        Ok(())
    }

    #[rustfmt::skip]
    pub(super) fn pool_to_pool(&mut self, sbid: usize, dst_mp: &mut MemoryPool, dbid: usize, bytes: usize) -> Result<(), ZyxError> {
        macro_rules! cross_backend {
            ($sm: expr, $sb: expr, $dm: expr, $db: expr) => {{
                let mut data: Vec<u8> = Vec::with_capacity(bytes);
                unsafe { data.set_len(bytes) };
                $sm.pool_to_host(&$sb[sbid], &mut data)?;
                $dm.host_to_pool(&data, &$db[dbid])?;
            }};
        }
        match (self, dst_mp) {
            #[rustfmt::skip]
            (MemoryPool::CUDA { buffers: sb, .. }, MemoryPool::CUDA { memory_pool: dm, buffers: db }) => { dm.pool_to_pool(&sb[sbid], &db[dbid])?; }
            #[rustfmt::skip]
            (MemoryPool::CUDA { memory_pool: sm, buffers: sb }, MemoryPool::HIP { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[rustfmt::skip]
            (MemoryPool::CUDA { memory_pool: sm, buffers: sb }, MemoryPool::OpenCL { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "wgsl")]
            #[rustfmt::skip]
            (MemoryPool::CUDA { memory_pool: sm, buffers: sb }, MemoryPool::WGSL { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[rustfmt::skip]
            (MemoryPool::HIP { memory_pool: sm, buffers: sb }, MemoryPool::CUDA { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[rustfmt::skip]
            (MemoryPool::HIP { buffers: sb, .. }, MemoryPool::HIP { memory_pool: dm, buffers: db }) => { dm.pool_to_pool(&sb[sbid], &db[dbid])?; }
            #[rustfmt::skip]
            (MemoryPool::HIP { memory_pool: sm, buffers: sb }, MemoryPool::OpenCL { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "wgsl")]
            #[rustfmt::skip]
            (MemoryPool::HIP { memory_pool: sm, buffers: sb }, MemoryPool::WGSL { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[rustfmt::skip]
            (MemoryPool::OpenCL { memory_pool: sm, buffers: sb }, MemoryPool::CUDA { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[rustfmt::skip]
            (MemoryPool::OpenCL { memory_pool: sm, buffers: sb }, MemoryPool::HIP { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[rustfmt::skip]
            (MemoryPool::OpenCL { buffers: sb, .. }, MemoryPool::OpenCL { memory_pool: dm, buffers: db }) => { dm.pool_to_pool(&sb[sbid], &db[dbid])?; }
            #[cfg(feature = "wgsl")]
            #[rustfmt::skip]
            (MemoryPool::OpenCL { memory_pool: sm, buffers: sb }, MemoryPool::WGSL { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "wgsl")]
            #[rustfmt::skip]
            (MemoryPool::WGSL { memory_pool: sm, buffers: sb }, MemoryPool::CUDA { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "wgsl")]
            #[rustfmt::skip]
            (MemoryPool::WGSL { memory_pool: sm, buffers: sb }, MemoryPool::HIP { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "wgsl")]
            #[rustfmt::skip]
            (MemoryPool::WGSL { memory_pool: sm, buffers: sb }, MemoryPool::OpenCL { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "wgsl")]
            #[rustfmt::skip]
            (MemoryPool::WGSL { buffers: sb, .. }, MemoryPool::WGSL { memory_pool: dm, buffers: db }) => { dm.pool_to_pool(&sb[sbid], &db[dbid])?; }
        }
        Ok(())
    }
}

impl Device {
    // NOTE returns memory pool id out of runtime memory pools
    pub(super) fn memory_pool_id(&self) -> MemoryPoolId {
        match self {
            Device::CUDA { memory_pool_id, .. } => *memory_pool_id,
            Device::HIP { memory_pool_id, .. } => *memory_pool_id,
            Device::OpenCL { memory_pool_id, .. } => *memory_pool_id,
            #[cfg(feature = "wgsl")]
            Device::WGSL { memory_pool_id, .. } => *memory_pool_id,
        }
    }

    pub(super) fn info(&self) -> &DeviceInfo {
        match self {
            Device::CUDA { device, .. } => device.info(),
            Device::HIP { device, .. } => device.info(),
            Device::OpenCL { device, .. } => device.info(),
            #[cfg(feature = "wgsl")]
            Device::WGSL { device, .. } => device.info(),
        }
    }

    pub(super) fn compute(&self) -> u128 {
        self.info().compute
    }

    pub(super) fn sync(&mut self, queue_id: usize) -> Result<(), ZyxError> {
        match self {
            Device::CUDA { queues, .. } => queues[queue_id].sync()?,
            Device::HIP { queues, .. } => queues[queue_id].sync()?,
            Device::OpenCL { queues, .. } => queues[queue_id].sync()?,
            #[cfg(feature = "wgsl")]
            Device::WGSL { queues, .. } => queues[queue_id].sync()?,
        }
        Ok(())
    }

    pub(super) fn release_program(&mut self, program_id: usize) -> Result<(), ZyxError> {
        //println!("Release program {program_id}");
        match self {
            Device::CUDA {
                device, programs, ..
            } => device.release_program(programs.remove(program_id).unwrap())?,
            Device::HIP {
                device, programs, ..
            } => device.release_program(programs.remove(program_id).unwrap())?,
            Device::OpenCL {
                device, programs, ..
            } => device.release_program(programs.remove(program_id).unwrap())?,
            #[cfg(feature = "wgsl")]
            Device::WGSL {
                device, programs, ..
            } => device.release_program(programs.remove(program_id).unwrap())?,
        }
        Ok(())
    }

    pub(super) fn compile(
        &mut self,
        ir_kernel: &IRKernel,
        debug_asm: bool,
    ) -> Result<usize, ZyxError> {
        let id = match self {
            Device::CUDA {
                device, programs, ..
            } => programs.push(device.compile(&ir_kernel, debug_asm)?),
            Device::HIP {
                device, programs, ..
            } => programs.push(device.compile(&ir_kernel, debug_asm)?),
            Device::OpenCL {
                device, programs, ..
            } => programs.push(device.compile(&ir_kernel, debug_asm)?),
            #[cfg(feature = "wgsl")]
            Device::WGSL {
                device, programs, ..
            } => programs.push(device.compile(&ir_kernel, debug_asm)?),
        };
        //println!("Compile program {id}");
        Ok(id)
    }

    pub(super) fn launch(
        &mut self,
        program_id: usize,
        memory_pool: &mut MemoryPool,
        buffer_ids: &[usize],
    ) -> Result<usize, ZyxError> {
        Ok(match self {
            Device::CUDA {
                programs, queues, ..
            } => {
                let (mut id, mut queue) = queues
                    .iter_mut()
                    .enumerate()
                    .min_by_key(|(_, queue)| queue.load())
                    .unwrap();
                if queue.load() > 10 {
                    (id, queue) = queues
                        .iter_mut()
                        .enumerate()
                        .max_by_key(|(_, queue)| queue.load())
                        .unwrap();
                    queue.sync()?;
                }
                let MemoryPool::CUDA { buffers, .. } = memory_pool else {
                    panic!()
                };
                queue.launch(&mut programs[program_id], buffers, &buffer_ids)?;
                id
            }
            Device::HIP {
                programs, queues, ..
            } => {
                let (mut id, mut queue) = queues
                    .iter_mut()
                    .enumerate()
                    .min_by_key(|(_, queue)| queue.load())
                    .unwrap();
                if queue.load() > 10 {
                    (id, queue) = queues
                        .iter_mut()
                        .enumerate()
                        .max_by_key(|(_, queue)| queue.load())
                        .unwrap();
                    queue.sync()?;
                }
                let MemoryPool::HIP { buffers, .. } = memory_pool else {
                    panic!()
                };
                queue.launch(&mut programs[program_id], buffers, &buffer_ids)?;
                id
            }
            Device::OpenCL {
                programs, queues, ..
            } => {
                let (mut id, mut queue) = queues
                    .iter_mut()
                    .enumerate()
                    .min_by_key(|(_, queue)| queue.load())
                    .unwrap();
                if queue.load() > 10 {
                    (id, queue) = queues
                        .iter_mut()
                        .enumerate()
                        .max_by_key(|(_, queue)| queue.load())
                        .unwrap();
                    queue.sync()?;
                }
                let MemoryPool::OpenCL { buffers, .. } = memory_pool else {
                    panic!()
                };
                queue.launch(&mut programs[program_id], buffers, &buffer_ids)?;
                id
            }
            #[cfg(feature = "wgsl")]
            Device::WGSL {
                programs, queues, ..
            } => {
                let (mut id, mut queue) = queues
                    .iter_mut()
                    .enumerate()
                    .min_by_key(|(_, queue)| queue.load())
                    .unwrap();
                if queue.load() > 10 {
                    (id, queue) = queues
                        .iter_mut()
                        .enumerate()
                        .max_by_key(|(_, queue)| queue.load())
                        .unwrap();
                    queue.sync()?;
                }
                let MemoryPool::WGSL { buffers, .. } = memory_pool else {
                    panic!()
                };
                queue.launch(&mut programs[program_id], buffers, &buffer_ids)?;
                id
            }
        })
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::CUDA { memory_pool_id, .. } => f.write_fmt(format_args!(
                "Device {{ memory_pool_id: {memory_pool_id} }})"
            )),
            Device::HIP { memory_pool_id, .. } => f.write_fmt(format_args!(
                "Device {{ memory_pool_id: {memory_pool_id} }})"
            )),
            Device::OpenCL { memory_pool_id, .. } => f.write_fmt(format_args!(
                "Device {{ memory_pool_id: {memory_pool_id} }})"
            )),
            #[cfg(feature = "wgsl")]
            Device::WGSL { memory_pool_id, .. } => f.write_fmt(format_args!(
                "Device {{ memory_pool_id: {memory_pool_id} }})"
            )),
        }
    }
}

/*pub(super) struct Timer {
    begin: std::time::Instant,
}

impl Timer {
    pub(crate) fn new() -> Timer {
        Timer {
            begin: std::time::Instant::now(),
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        println!("Timer took {}us", self.begin.elapsed().as_micros());
    }
}*/
