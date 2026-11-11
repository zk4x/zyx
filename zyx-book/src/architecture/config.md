# Configuration

Zyx is configured through a JSON file and environment variables. The file is
read **once** per process, on the first backend access; changing it requires a
restart. Missing sections and missing keys fall back to defaults, and a missing
or unparsable file simply means "all defaults".

## Config File

The runtime reads `$XDG_CONFIG_HOME/zyx/config.json` when `XDG_CONFIG_HOME` is
an absolute path, otherwise `~/.config/zyx/config.json`. `ZYX_DEBUG=1` prints
whether the file was found and parsed.

A complete example (every section is optional):

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
    "cuda": { "device_ids": [0], "cudnn": true, "queues": 8 },
    "opencl": { "platform_ids": [0], "queues": 8 },
    "vulkan": { "device_ids": [0] },
    "tenstorrent": { "device_ids": [0] }
}
```

`wgpu` (`"wgpu": { "enabled": true }`) exists only when the crate is built with
the `wgpu` feature; the `tenstorrent` section only has an effect when the crate
is built with the `tenstorrent` feature.

### Backend Sections

| Section | Keys | Default | Disable with |
|---------|------|---------|--------------|
| `c` | `enabled` | `true` | `"enabled": false` |
| `cuda` | `device_ids`, `cudnn`, `queues` | all GPUs, cuDNN off, 8 queues | `"device_ids": []` |
| `opencl` | `platform_ids`, `queues` | all platforms, 8 queues | `"platform_ids": []` |
| `vulkan` | `device_ids` | all devices | `"device_ids": []` |
| `tenstorrent` | `device_ids` | all devices (with the feature) | `"device_ids": []` |
| `wgpu` | `enabled` | `true` | `"enabled": false` |

`device_ids` / `platform_ids`: `null`/absent means "every device the backend
finds"; a list selects exactly those ordinals; an **empty list disables the
backend**. Ordinals are the backend's own (CUDA driver order, Vulkan device
index, OpenCL platform index).

`cuda.cudnn` enables the cuDNN AOT matmul path (the struct default is `false`).
`cuda.queues` and `opencl.queues` set the number of hardware command
queues/streams per device used by the submission workers (default 8).

### Autotune Section

| Key | Default | Meaning |
|-----|---------|---------|
| `n_launches` | `1` | candidates actually launched and timed |
| `n_seeds` | `200` | initial optimization samples |
| `n_added_per_step` | `200` | new candidates per expansion |
| `n_removed_per_step` | `0` | candidates dropped per expansion |
| `n_total_opts` | `1000` | expansion iterations |

See [Optimization Passes](./optimizations.md#how-the-search-works) for how the
search uses them.

## Environment Variables

### `ZYX_DEV` — device pinning

`ZYX_DEV` overrides the config file and leaves exactly one device available;
every other backend is configured out (and a device with an ordinal keeps only
that ordinal). It accepts the same syntax as `Dev::from_str`:

```bash
ZYX_DEV=cpu          # C backend
ZYX_DEV=cuda:0        # CUDA driver ordinal 0
ZYX_DEV=tt            # Tenstorrent, device 0 (feature "tenstorrent")
ZYX_DEV=vulkan        # Vulkan device 0
ZYX_DEV=opencl:1      # OpenCL platform 1
```

Names are case-insensitive: `auto`, `c`/`cpu`, `host`, `cuda`,
`tt`/`tenstorrent`, `vulkan`/`vk`, `opencl`/`cl`, `wgpu`. An index on an
indexless device (`c:1`) is an error. A bad value panics — silently falling back
would run on the wrong hardware — and `ZYX_DEV=auto` is a no-op (no narrowing).

### `ZYX_DEBUG` — trace flags

A bitmask; add the values you want:

| Value | Output |
|-------|--------|
| 1 | Devices + configuration |
| 2 | Graph/e-graph dump |
| 4 | Kernel IR before linearization (scheduler view) |
| 8 | Winning kernel IR after the search |
| 16 | Generated assembly/source |
| 32 | Kernel launches |
| 64 | Alloc/dealloc |
| 128 | Kernel compilation |
| 256 | Disable the autotune search (`NoSearch`) |

```bash
ZYX_DEBUG=8 cargo run     # print the winning kernel IR
ZYX_DEBUG=24 cargo run    # IR (8) + generated code (16)
ZYX_DEBUG=256 cargo run   # NoSearch: compile a cleaned seed once
```

### Other variables

| Variable | Effect |
|----------|--------|
| `AGENT=1` | Colorless debug output (also auto-detected when stdout is not a terminal) |
| `ZYX_DRY_RUN=1` | Skip device launches; timings are a fixed 1 s placeholder and output buffers are uninitialized — never read them |
| `ZYX_PTX` | CUDA codegen emits PTX directly instead of CUDA C + NVRTC |
| `ZYX_BENCH_OUT` | Append one JSON line per measured autotune candidate to this path |
| `ZYX_BENCH_TAG` | Tag for the benchmark rows (`untagged` by default) |
| `ZYX_TT_IR` | Dump the Tenstorrent IR before/after `tt_tile`, `tt_reindex`, `tt_dataflow_select` |
| `ZYX_TT_DUMP` | Dump each Tenstorrent role program (bytes, entries, length) |
| `TT_TTSIM_LIB` | Path to the Tenstorrent simulator library (default `vendor/libttsim_bh.so`) |
| `TT_TTSIM_LIB_X2` / `TT_TTSIM_LIB_X4` | Simulator library for dual-/quad-chip builds |
| `TT_MATH` | Tenstorrent SFPU math mode: `precise` (default) or `approx` |
| `ZYX_KERN_TRACE` | Kernelizer split/store tracing on stderr |

## API Configuration

```rust,ignore
Tensor::manual_seed(42);        // reseed the runtime RNG
Tensor::set_training(true);     // enable training mode (autograd graph is kept)
Tensor::set_training(false);    // inference
let t = Tensor::training();     // current mode
```

Device pinning can also be done per tensor with `Tensor::to(Dev)`; see
[Backend System](./backends.md).
