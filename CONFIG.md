# Configuration

zyx is configured through a JSON file and environment variables. The file is
read **once** per process, on the first backend access; changing it requires a
restart. Missing sections and missing keys fall back to defaults, and a missing
or unparsable file means “all defaults”.

The config file is searched for at:

1. `$XDG_CONFIG_HOME/zyx/config.json` (when `XDG_CONFIG_HOME` is an absolute path)
2. `$HOME/.config/zyx/config.json`

Run `ZYX_DEBUG=1` to see whether the file was found and parsed.

## Example `config.json`

Every section is optional:

```json
{
  "autotune": {
    "n_launches": 1,
    "n_seeds": 200,
    "n_added_per_step": 200,
    "n_removed_per_step": 0,
    "n_total_opts": 1000
  },
  "c": { "enabled": true },
  "cuda": { "device_ids": null, "cudnn": false, "queues": 8 },
  "opencl": { "platform_ids": null, "queues": 8 },
  "vulkan": { "device_ids": null },
  "tenstorrent": { "device_ids": null },
  "wgpu": { "enabled": true }
}
```

`wgpu` exists only when the crate is built with the `wgpu` feature.

## Options Reference

### `autotune` — kernel autotuning

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_launches` | `usize` | `1` | Candidates actually launched and timed |
| `n_seeds` | `usize` | `200` | Initial optimization samples |
| `n_added_per_step` | `usize` | `200` | New candidates per expansion |
| `n_removed_per_step` | `usize` | `0` | Candidates dropped per expansion |
| `n_total_opts` | `usize` | `1000` | Expansion iterations |

The cost function only ranks candidates; measured launch times decide the
winner.

### `c` — C/Clang CPU backend

Compiles kernel IR to C, then compiles with clang/gcc and loads the shared
object. Uses the host memory pool.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable the C backend |

### `cuda` — CUDA backend

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device_ids` | `Option<Vec<i32>>` | `null` | `null` = all devices; `[]` = disable CUDA; `[0, 1]` = those driver ordinals |
| `cudnn` | `bool` | `false` | Use cuDNN for AOT matmul kernels |
| `queues` | `Option<usize>` | `8` | Hardware queues (streams) per device |

### `opencl` — OpenCL backend

OpenCL supports multiple platforms (e.g. Intel GPU, POCL CPU); each platform
may have multiple devices.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `platform_ids` | `Option<Vec<usize>>` | `null` | `null` = all platforms; `[]` = disable OpenCL; `[0]` = first platform only |
| `queues` | `Option<usize>` | `8` | In-order command queues per device |

### `vulkan` — Vulkan backend

Uses the `ash` crate for Vulkan compute.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device_ids` | `Option<Vec<i32>>` | `null` | `null` = all devices; `[]` = disable Vulkan; `[0]` = first device only |

### `tenstorrent` — Tenstorrent backend

Always compiled in; it runs through the bundled simulator, which is found
through `TT_TTSIM_LIB` (default `vendor/libttsim_bh.so`). A machine without a
usable simulator simply contributes no devices.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device_ids` | `Option<Vec<i32>>` | `null` | `null` = all devices; `[]` = disable Tenstorrent |

### `wgpu` — WGPU backend (feature `wgpu`)

Requires `--features wgpu`. Supports Vulkan, Metal, DX12, and OpenGL ES.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable the WGPU backend |

## Backend Selection Rules

| Backend | Enabled by default | Disable with |
|---------|--------------------|--------------|
| `c` | yes | `"enabled": false` |
| `cuda` | yes | `"device_ids": []` |
| `opencl` | yes | `"platform_ids": []` |
| `vulkan` | yes | `"device_ids": []` |
| `tenstorrent` | yes | `"device_ids": []` |
| `wgpu` | yes (with the feature) | `"enabled": false` |

`null`/absent means “every device the backend finds”; a list selects exactly
those ordinals.

C, CUDA, OpenCL, Vulkan, and Tenstorrent are compiled in
unconditionally (they locate their runtimes at load time); only WGPU is behind a
cargo feature. There is no global “initialize all backends” step: `Dev::all()`
probes each compiled-in backend lazily, and backends whose driver, device, or
simulator is missing contribute nothing.

## `ZYX_DEV` — pin one device

`ZYX_DEV` is applied on top of the config file and leaves exactly one device
available (every other backend is configured out):

```bash
ZYX_DEV=cpu          # C backend
ZYX_DEV=cuda:0        # CUDA driver ordinal 0
ZYX_DEV=tt            # Tenstorrent, device 0
ZYX_DEV=vulkan        # Vulkan device 0
```

Names are case-insensitive: `auto`, `c`/`cpu`, `host`, `cuda`,
`tt`/`tenstorrent`, `vulkan`/`vk`, `opencl`/`cl`, `wgpu`. An index on an
indexless device (`c:1`) is an error, and a bad value panics — silently falling
back would run on the wrong hardware. `ZYX_DEV=auto` narrows nothing.
