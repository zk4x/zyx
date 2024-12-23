use std::sync::Arc;

use nanoserde::DeJson;
use pollster::FutureExt;
use wgpu::ShaderModule;

use crate::runtime::Pool;

use super::{BackendError, Device, DeviceInfo};

#[derive(DeJson, Debug, Default)]
pub struct WGPUConfig {
    enabled: bool,
}

#[derive(Debug)]
pub(super) struct WGPUMemoryPool {
    free_bytes: usize,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

#[derive(Debug)]
pub(super) struct WGPUDevice {
    dev_info: DeviceInfo,
    memory_pool_id: u32,
    device: Arc<wgpu::Device>,
    #[allow(unused)]
    adapter: wgpu::Adapter,
}

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
    let mut memory_pools = Vec::new();
    let mut devices = Vec::new();
    memory_pools.push(WGPUMemoryPool {
        free_bytes: 1_000_000_000,
        device: device.clone(),
        queue: queue.clone(),
    });
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
        memory_pool_id: 0,
    }));

    Ok(())
}
