// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

// C linkage facade around the Tenstorrent (tt-metal) host runtime.
//
// Every tt-metal C++ call is routed through this module so that Rust sees
// only `extern "C"` symbols. All tt-metal C++ exceptions are caught inside
// this module and reported via `get_last_error`/`has_error`; no C++
// exceptions ever cross the FFI boundary.

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/// Return the last tt-metal error message and clear it. `out_len` is the
/// capacity of `out`; the return value is the number of bytes written
/// (excluding the NUL terminator). Returns 0 when there was no error.
void get_last_error(char* out, int out_len);

/// True iff the last call recorded an error.
bool has_error();

// ---------------------------------------------------------------------------
// Device lifecycle: opaque device handle (owns MeshDevice + program cache).
// ---------------------------------------------------------------------------

/// Create a unit-mesh device (device id 0). Returns NULL and records an
/// error on failure. The returned handle is owned by the caller and must be
/// destroyed with `destroy_device`.
void* create_device();

/// Close the device and release all cached programs.
void destroy_device(void* dev);

/// Destroy all MetalContext instances. Must be called on the same thread
/// that created the device: UMD's CHIP_IN_USE mutex is owned by the creating
/// thread, and the exit-time MetalContext teardown runs on the main thread,
/// which would unlock as a non-owner (EPERM) and terminate. Returns with an
/// error recorded (never throws) if teardown fails.
void teardown_metal();

/// Query the logical tensix compute grid (rows = gidx0, cols = gidx1).
/// Grid size is fixed for the lifetime of the device.
void get_grid_size(void* dev, uint32_t* rows, uint32_t* cols);

// ---------------------------------------------------------------------------
// Device DRAM buffers: opaque buffer handle (MeshBuffer*).
// ---------------------------------------------------------------------------

/// Allocate device DRAM of at least `size` bytes (rounded up to a 4 KiB
/// page multiple). `tile_bytes` is accepted for interface compat and
/// ignored. Returns NULL on error (including zero size).
void* alloc_buffer(void* dev, uint64_t size, uint64_t tile_bytes);

/// Free a previously allocated buffer. The buffer must not be in use.
void free_buffer(void* dev, void* buf);

// ---------------------------------------------------------------------------
// Host <-> device transfers. Synchronous (blocking): enqueue + Finish.
// The `src` / `dst` buffers must remain valid for the duration of the call.
// ---------------------------------------------------------------------------

/// Host -> device. Enqueues a write, blocks until the device has the data.
void write_buffer(void* dev, void* buf, const void* src, uint64_t len);

/// Device -> host. Enqueues a read, blocks until the device is done; writes
/// up to `len` bytes into `dst`.
void read_buffer(void* dev, void* buf, void* dst, uint64_t len);

// ---------------------------------------------------------------------------
// Programs: per-device cache id (index into the device's program table).
// ---------------------------------------------------------------------------

/// Cache the compilation of a kernel: three RISC-V source sections, circular
/// buffer config, per-section param ordinals and their head-order counts,
/// plus the 32-bit DST flag. Only validates + caches; kernels are built per
/// launch with launch-time accessor args and grid. On success returns the
/// program id (0-based); on failure records an error and returns `UINT32_MAX`.
uint32_t compile_program(
    void* dev,
    const uint8_t* reader_src, size_t reader_src_len,
    const uint8_t* compute_src, size_t compute_src_len,
    const uint8_t* writer_src, size_t writer_src_len,
    const uint32_t* cb_indices, const uint32_t* cb_formats,
    const uint32_t* cb_tile_bytes, const uint32_t* cb_num_tiles,
    size_t n_cbs,
    const uint32_t* reader_params, size_t n_reader_params,
    const uint32_t* compute_params, size_t n_compute_params,
    const uint32_t* writer_params, size_t n_writer_params,
    uint32_t n_params,
    bool fp32_dest_acc_en);

/// Launch a cached program. `src_buffers` / `dst_buffers` are device buffer
/// handles (in the same order as the launch's src / dst args); `vars` is the
/// (ordinal, value) map for Variable params. Returns true on success.
bool run_program(
    void* dev, uint32_t prog,
    const void** src_buffers, uint32_t n_src,
    const void** dst_buffers, uint32_t n_dst,
    uint32_t grid_rows, uint32_t grid_cols,
    const uint32_t* var_ordinals, const uint32_t* var_values, uint32_t n_vars);

/// Release a cached program back to the device.
void destroy_program(void* dev, uint32_t prog);

#ifdef __cplusplus
}
#endif
