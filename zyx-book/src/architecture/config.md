# Configuration

Zyx is configured through a JSON file, environment variables, and API calls.

## Config File

The runtime reads `$XDG_CONFIG_HOME/zyx/config.json` (or `~/.config/zyx/config.json` by default):

```json
{
    "c": { "enabled": true },
    "cuda": { "device_ids": [0] },
    "opencl": { "platform_ids": [0] },
    "autotune": {
        "enabled": true,
        "max_configs": 100,
        "timeout_ms": 5000
    }
}
```

### Backend Control

| Config | Effect |
|--------|--------|
| `"c": { "enabled": true }` | Enable CPU backend (off by default) |
| `"cuda": { "device_ids": [] }` | Disable all CUDA devices |
| `"opencl": { "platform_ids": [] }` | Disable all OpenCL |

## Environment Variables

### Debug Flags

Set `ZYX_DEBUG` as a bitmask:

| Value | Output |
|-------|--------|
| 1 | Devices + configuration, kernel launches |
| 2 | Egraph print (after realize) |
| 4 | Scheduler kernels — IR before linearization |
| 8 | Kernel IR after linearization + optimization |
| 16 | Generated assembly/code |
| 32 | Launch + memory movement |
| 64 | Alloc/dealloc |
| 128 | Kernel compilation |
| 256 | Autotune exploration |

```bash
ZYX_DEBUG=8 cargo run     # print kernel IR
ZYX_DEBUG=16 cargo run    # print generated code
```

### Colorless Output

```bash
AGENT=1 ZYX_DEBUG=8 cargo test
```

## API Configuration

```rust,ignore
Tensor::manual_seed(42);            // deterministic RNG
Tensor::set_training(true);         // enable training mode
Tensor::set_training(false);        // disable training mode
```
