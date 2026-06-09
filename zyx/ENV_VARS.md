Set `ZYX_DEBUG` environment variable to enable debugging. It is a bitmask with the following options:

| Value | Flag | Description |
|-------|------|-------------|
| 1     | dev  | Print hardware devices and configuration |
| 2     | perf | Print graph execution characteristics and performance |
| 4     | sched | Print kernels created by scheduler |
| 8     | ir   | Print kernels in intermediate representation |
| 16    | asm  | Print kernels in native assembly/code (OpenCL, WGSL, etc.) |
| 32    | kmd  | Print kernel launch and memory movement operations |
| 64    | memory | Print memory allocation and deallocation |
| 128   | compile | Print kernel compilation |

Combine flags by summing values (e.g., `ZYX_DEBUG=24` enables ir + asm).
