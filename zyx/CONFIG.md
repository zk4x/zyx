# Configuration

zyx is configured via a JSON file. The config file is searched for at:

1. `$XDG_CONFIG_HOME/zyx/config.json` (if `XDG_CONFIG_HOME` is set and is absolute)
2. `$HOME/.config/zyx/config.json`

If neither exists, all defaults are used.

The same directory is also used for the kernel cache (file named `cached_kernels`).

## Example `config.json`

```json
{
  "autotune": {
    "save_to_disk": true,
    "n_launches": 1,
    "n_seeds": 100,
    "n_added_per_step": 10,
    "n_removed_per_step": 5,
    "n_total_opts": 1000
  },
  "dummy": {
    "enabled": false
  },
  "c": {
    "enabled": false
  },
  "cuda": {
    "device_ids": null
  },
  "hip": {
    "device_ids": null
  },
  "opencl": {
    "platform_ids": null
  },
  "wgpu": {
    "enabled": true
  },
  "vulkan": {
    "device_ids": []
  }
}
```

Any key can be omitted — the missing section falls back to defaults.

## Options Reference

### `autotune` — Kernel autotuning

Controls how zyx searches for optimal kernel configurations.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `save_to_disk` | `bool` | `true` | Cache autotuned kernels to disk |
| `n_launches` | `usize` | `20` | Max kernel launches during autotune search |
| `n_seeds` | `usize` | `100` | Number of initial optimization seeds |
| `n_added_per_step` | `usize` | `10` | Optimizations to try per iteration |
| `n_removed_per_step` | `usize` | `5` | Optimizations to remove per iteration |
| `n_total_opts` | `usize` | `1000` | Max total optimizations to try |

### `dummy` — Dummy test backend

A fake device with ~1 TB memory used for testing. All operations succeed without actual computation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable the dummy backend |

### `c` — C/Clang CPU backend

Compiles kernel IR to C, then compiles with clang and loads via `dlopen`. Uses the host memory pool.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable the C/Clang CPU backend |

### `cuda` — CUDA backend

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device_ids` | `Option<Vec<i32>>` | `null` | Which CUDA devices to use. `null` = all available devices. `[]` (empty) = disable CUDA. `[0, 1]` = use first two devices. |

### `hip` — HIP/AMD backend

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device_ids` | `Option<Vec<i32>>` | `null` | Which HIP devices to use. `null` = all available devices. |

### `opencl` — OpenCL backend

OpenCL supports multiple platforms (e.g., Intel GPU, POCL CPU). Each platform may have multiple devices.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `platform_ids` | `Option<Vec<usize>>` | `null` | Which OpenCL platforms to use. `null` = all available. `[]` = disable OpenCL. `[0]` = use first platform only. |

### `wgpu` — WGPU backend (feature `wgpu`)

Requires `--features wgpu` at compile time. Supports Vulkan, Metal, DX12, and OpenGL ES.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable the WGPU backend |

### `tenstorrent` — Tenstorrent backend (feature `tenstorrent`)

Requires `--features tenstorrent` at compile time.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device_ids` | `Option<Vec<i32>>` | `null` | Which Tenstorrent devices to use. `null` = all available. `[]` = disable Tenstorrent. |

### `vulkan` — Vulkan backend

Uses the vulkano crate for Vulkan compute operations.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device_ids` | `Option<Vec<i32>>` | `null` | Which Vulkan devices to use. `null` = all available. `[]` = disable Vulkan. `[0]` = use first device only. |

## Backend selection rules

- **dummy**, **c**: enabled only when `"enabled": true`
- **cuda**: disabled when `"device_ids": []`; uses all devices when `null`
- **hip**: currently always tries to initialize (ignores config); disabled only if `libamdhip64.so` is not found
- **opencl**: disabled when `"platform_ids": []`; uses all platforms when `null`
- **tenstorrent**: disabled when `"device_ids": []`; uses all devices when `null`
- **vulkan**: disabled when `"device_ids": []`; uses all devices when `null`
- **wgpu**: enabled by default; disabled when `"enabled": false`

CUDA, HIP, OpenCL, and Tenstorrent backends are always compiled into zyx (cannot be disabled by cargo features). They search for required `.so` files at runtime. WGPU requires the `wgpu` cargo feature.

The Vulkan backend is compiled by default. It requires the vulkano crate (vulkan-sys, ash, etc.).

If all backends fail to initialize or are configured out, initialization returns an error.
