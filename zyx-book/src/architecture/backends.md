# Backend System

Zyx supports multiple hardware backends through an enum dispatch system. All backends are compiled into the library and selected at runtime.

## Enum Dispatch

Backends use enums instead of trait objects (`dyn Backend`). Trait objects would require downcasting to access backend-specific functionality, which is ugly in Rust.

```rust,ignore
pub enum Device {
    C(CDevice),
    CUDA(CUDADevice),
    OpenCL(OpenCLDevice),
    Vulkan(VulkanDevice),
    WGPU(WGPUDevice),
    HIP(HIPDevice),
    Dummy(DummyDevice),
}

pub enum MemoryPool {
    Host(HostMemoryPool),
    Disk(DiskMemoryPool),
    C(CMemoryPool),
    CUDA(CUDAMemoryPool),
    OpenCL(OpenCLMemoryPool),
    Vulkan(VulkanMemoryPool),
    WGPU(WGPUMemoryPool),
    HIP(HIPMemoryPool),
    Dummy(DummyMemoryPool),
}
```

Each method matches on the variant and delegates:

```rust,ignore
impl Device {
    pub fn compile(&self, kernel: &Kernel, debug: DebugMask) -> Result<ProgramId, ZyxError> {
        match self {
            Device::C(dev) => dev.compile(kernel, debug),
            Device::CUDA(dev) => dev.compile(kernel, debug),
            // ...
        }
    }
}
```

## Backend Codegen is Trivial

The optimization passes do the hard work. Backend codegen is:

1. **deSSA** — resolve SSA references to physical registers or memory locations
2. **Linear pass** — walk the op linked list once, emitting target instructions

No further optimizations, no complex backend-specific lowering. The IR emits directly to the target language.

## Initialization

Backends are initialized at startup via `initialize_backends()`:

```rust,ignore
pub fn initialize_backends(config, memory_pools, devices, debug) {
    host::initialize_pool(memory_pools, debug);
    disk::initialize_pool(memory_pools, debug);
    dummy::initialize_device(&config.dummy, ...);
    c::initialize_device(&config.c, ...);
    cuda::initialize_device(&config.cuda, ...);
    opencl::initialize_device(&config.opencl, ...);
    hip::initialize_device(&config.hip, ...);
    vulkan::initialize_device(&config.vulkan, ...);
    wgpu::initialize_device(&config.wgpu, ...);
    #[cfg(feature = "tenstorrent")]
    tenstorrent::initialize_device(&config.tenstorrent, ...);
}
```

Each backend tries to initialize. Failure (missing driver, no hardware) causes it to be skipped silently. If all backends fail, the program exits with an error.

## Current Backends

| Backend | Source | Target | Runtime |
|---------|--------|--------|---------|
| C | `c.rs` | C99 (compiled to .so) | Clang/GCC |
| CUDA | `cuda.rs` | CUDA C (compiled to SASS) | CUDA driver via `libloading` |
| HIP | `hip.rs` | HIP | ROCm via `libloading` |
| OpenCL | `opencl.rs` | OpenCL C | OpenCL runtime via `libloading` |
| Vulkan | `vulkan.rs` | SPIR-V | Vulkan via `ash` crate |
| WGPU | `wgpu.rs` | SPIR-V | WGPU (feature: `wgpu`) |
| Dummy | `dummy.rs` | — | No hardware needed (fake device) |

All backends except WGPU and Tenstorrent are compiled in by default. WGPU requires `--features wgpu`. Tenstorrent requires `--features tenstorrent`.

## Device Configuration in Config File

Each backend can be enabled/disabled and configured:

```json
{
    "c": { "enabled": true },
    "cuda": { "device_ids": [0] },
    "opencl": { "platform_ids": [] },
    "dummy": { "enabled": false }
}
```

If a section is missing or the config file doesn't exist, defaults are used (most backends enabled).

## Device Selection

The scheduler picks a device at realize time:

1. If `DeviceId::AUTO`, sort devices by free compute capacity (descending)
2. If a specific device is requested, try it first
3. Pick the first device with enough free memory for all required tensors
4. If no device has enough memory, return an allocation error
