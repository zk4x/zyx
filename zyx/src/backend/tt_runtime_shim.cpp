// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

// C++ shim wrapping the Tenstorrent (tt-metal) host runtime. All tt-metal API
// calls are here, so Rust sees only the `extern "C"` facade in
// `tt_runtime_shim.h`.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_metal/impl/context/metal_context.hpp>

#include "tt_runtime_shim.h"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

constexpr uint64_t PAGE_SIZE = 4096;

// Round up to a whole-page multiple (at least one page). Mesh buffers with
// non-interleaved layout require exact page multiples; callers emit small
// single-tile configs, so the tail is zero-padded by write_buffer.
static uint64_t round_up_pages(uint64_t size) {
    if (size == 0) {
        return 0;
    }
    return ((size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
}

// Thread-local error: every tt-metal call is wrapped so exceptions never
// cross the FFI boundary. Each calling thread reads its own error on the
// same thread immediately after the call.
static thread_local std::string s_last_error;

static void set_last_error(const char* msg) {
    s_last_error.assign(msg);
}

static void catch_and_set(std::function<void()> f) {
    try {
        f();
    } catch (const std::exception& e) {
        set_last_error(e.what());
    } catch (const std::string& s) {
        set_last_error(s.c_str());
    } catch (...) {
        set_last_error("unknown tt-metal error");
    }
}

void get_last_error(char* out, int out_len) {
    int n = static_cast<int>(s_last_error.size());
    if (out && out_len > 0) {
        int copy = std::min(n, out_len - 1);
        std::memcpy(out, s_last_error.data(), static_cast<size_t>(copy));
        out[copy] = '\0';
    }
    s_last_error.clear();
}

bool has_error() {
    return !s_last_error.empty();
}

// ---------------------------------------------------------------------------
// Opaque handles (all C++ objects stay in this TU).
// ---------------------------------------------------------------------------

struct ProgramHandle {
    std::string reader_source;
    std::string compute_source;
    std::string writer_source;
    std::vector<uint32_t> cb_indices;
    std::vector<uint32_t> cb_formats;
    std::vector<uint32_t> cb_tile_bytes;
    std::vector<uint32_t> cb_num_tiles;
    uint32_t n_params = 0;
    std::vector<uint32_t> reader_params;
    std::vector<uint32_t> compute_params;
    std::vector<uint32_t> writer_params;
    bool fp32_dest_acc_en = false;
};

struct DeviceHandle {
    std::shared_ptr<MeshDevice> device;
    MeshCommandQueue* cq = nullptr;
    // Per-device program cache (never global: two devices must not share ids).
    std::vector<std::shared_ptr<ProgramHandle>> programs;
};

struct BufferHandle {
    // Command queue on the pinned device that owns this buffer. The device is
    // pinned for the lifetime of the runtime, so the pointer is valid while
    // the buffer exists.
    MeshCommandQueue* cq = nullptr;
    std::shared_ptr<MeshBuffer> buffer;
    // Requested size (bytes) and page-rounded mesh size (bytes).
    uint64_t size = 0;
    uint64_t mesh_size = 0;
};

// ---------------------------------------------------------------------------
// Device lifecycle
// ---------------------------------------------------------------------------

extern "C" int32_t num_devices() {
    int32_t n = -1;
    catch_and_set([&] {
        n = static_cast<int32_t>(GetNumAvailableDevices());
    });
    return n;
}

extern "C" void* create_device(int32_t device_id) {
    DeviceHandle* h = new DeviceHandle;
    catch_and_set([&] {
        // tt-metal needs its root (kernels, fw) when creating a device.
        // Keep the old runtime-exe behavior: fall back to the compile-time
        // TT_METAL_ROOT when the env is unset.
        if (!getenv("TT_METAL_RUNTIME_ROOT")) {
#ifdef TT_METAL_ROOT_DEFAULT
            setenv("TT_METAL_RUNTIME_ROOT", TT_METAL_ROOT_DEFAULT, 0);
#endif
        }
        h->device = MeshDevice::create_unit_mesh(device_id);
        h->cq = &h->device->mesh_command_queue();
    });
    if (has_error()) {
        delete h;
        return nullptr;
    }
    return h;
}

extern "C" void destroy_device(void* dev_raw) {
    if (!dev_raw) return;
    auto* h = static_cast<DeviceHandle*>(dev_raw);
    catch_and_set([&] {
        h->programs.clear();
        h->device->close();
        delete h;
    });
}

extern "C" void teardown_metal() {
    catch_and_set([] {
        MetalContext::destroy_all_instances();
    });
}

extern "C" void get_grid_size(void* dev_raw, uint32_t* rows, uint32_t* cols) {
    auto* h = static_cast<DeviceHandle*>(dev_raw);
    catch_and_set([&] {
        CoreCoord g = h->device->compute_with_storage_grid_size();
        *rows = g.y;
        *cols = g.x;
    });
}

// ---------------------------------------------------------------------------
// DRAM buffers
// ---------------------------------------------------------------------------

extern "C" void* alloc_buffer(void* dev_raw, uint64_t size, uint64_t) {
    auto* dev = static_cast<DeviceHandle*>(dev_raw);
    if (size == 0) {
        set_last_error("alloc_buffer: zero size");
        return nullptr;
    }
    uint64_t mesh_size = round_up_pages(size);
    if (mesh_size > UINT32_MAX) {
        set_last_error("alloc_buffer: size exceeds 32-bit mesh limit");
        return nullptr;
    }
    BufferHandle* h = new BufferHandle;
    catch_and_set([&] {
        DeviceLocalBufferConfig dram_config;
        dram_config.page_size = static_cast<uint32_t>(PAGE_SIZE);
        dram_config.buffer_type = BufferType::DRAM;
        ReplicatedBufferConfig buf_config{.size = static_cast<uint32_t>(mesh_size)};
        h->buffer = MeshBuffer::create(buf_config, dram_config, dev->device.get());
        h->size = size;
        h->mesh_size = mesh_size;
        h->cq = dev->cq;
    });
    if (has_error()) {
        delete h;
        return nullptr;
    }
    return h;
}

extern "C" void free_buffer(void* dev_raw, void* buf_raw) {
    (void)dev_raw;
    if (!buf_raw) return;
    auto* h = static_cast<BufferHandle*>(buf_raw);
    h->buffer.reset();
    delete h;
}

extern "C" void write_buffer(void* dev_raw, void* buf_raw, const void* src, uint64_t len) {
    (void)dev_raw;
    auto* h = static_cast<BufferHandle*>(buf_raw);
    catch_and_set([&] {
        // Pad to the page-aligned mesh size so the device reads a fully-valid
        // page instead of tail garbage; mirrors alloc_buffer.
        uint64_t mesh_size = h->mesh_size;
        if (mesh_size == 0) {
            throw std::runtime_error("write_buffer: uninitialized buffer");
        }
        if (len > mesh_size) {
            throw std::runtime_error("write_buffer: length exceeds mesh size");
        }
        std::vector<uint8_t> data(static_cast<size_t>(mesh_size), 0);
        std::memcpy(data.data(), src, static_cast<size_t>(len));
        EnqueueWriteMeshBuffer(*h->cq, h->buffer, data, false);
        Finish(*h->cq);
    });
}

extern "C" void read_buffer(void* dev_raw, void* buf_raw, void* dst, uint64_t len) {
    (void)dev_raw;
    auto* h = static_cast<BufferHandle*>(buf_raw);
    catch_and_set([&] {
        std::vector<uint8_t> result;
        EnqueueReadMeshBuffer(*h->cq, result, h->buffer, true);
        uint64_t copy_sz = std::min(len, static_cast<uint64_t>(result.size()));
        std::memcpy(dst, result.data(), static_cast<size_t>(copy_sz));
    });
}

// ---------------------------------------------------------------------------
// Programs: validate + cache at compile; build kernels per launch.
// ---------------------------------------------------------------------------

static DataFormat cb_data_format(uint32_t f) {
    switch (f) {
    case 0: return DataFormat::Float32;
    case 1: return DataFormat::Float16;
    case 2: return DataFormat::Float16_b;
    case 3: return DataFormat::UInt16;
    case 4: return DataFormat::Fp8_e4m3;
    case 5: return DataFormat::UInt8;
    case 6: return DataFormat::Int8;
    case 7: return DataFormat::UInt32;
    case 8: return DataFormat::Int32;
    default: throw std::runtime_error("unsupported cb data_format " + std::to_string(f));
    }
}

extern "C" uint32_t compile_program(
    void* dev_raw,
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
    bool fp32_dest_acc_en)
{
    auto* dev = static_cast<DeviceHandle*>(dev_raw);
    auto prog = std::make_shared<ProgramHandle>();
    catch_and_set([&] {
        if (!reader_src && reader_src_len > 0) throw std::runtime_error("compile_program: null reader source");
        if (!compute_src && compute_src_len > 0) throw std::runtime_error("compile_program: null compute source");
        if (!writer_src && writer_src_len > 0) throw std::runtime_error("compile_program: null writer source");
        prog->reader_source.assign(reinterpret_cast<const char*>(reader_src), reader_src_len);
        prog->compute_source.assign(reinterpret_cast<const char*>(compute_src), compute_src_len);
        prog->writer_source.assign(reinterpret_cast<const char*>(writer_src), writer_src_len);
        prog->n_params = n_params;
        if (n_reader_params > 0) prog->reader_params.assign(reader_params, reader_params + n_reader_params);
        if (n_compute_params > 0) prog->compute_params.assign(compute_params, compute_params + n_compute_params);
        if (n_writer_params > 0) prog->writer_params.assign(writer_params, writer_params + n_writer_params);
        prog->fp32_dest_acc_en = fp32_dest_acc_en;
        prog->cb_indices.assign(cb_indices, cb_indices + n_cbs);
        prog->cb_formats.assign(cb_formats, cb_formats + n_cbs);
        prog->cb_tile_bytes.assign(cb_tile_bytes, cb_tile_bytes + n_cbs);
        prog->cb_num_tiles.assign(cb_num_tiles, cb_num_tiles + n_cbs);

        // Validate parameter ordinals against the param count.
        for (uint32_t p : prog->reader_params) {
            if (p >= n_params) throw std::runtime_error("reader param ordinal " + std::to_string(p) + " >= n_params " + std::to_string(n_params));
        }
        for (uint32_t p : prog->compute_params) {
            if (p >= n_params) throw std::runtime_error("compute param ordinal " + std::to_string(p) + " >= n_params " + std::to_string(n_params));
        }
        for (uint32_t p : prog->writer_params) {
            if (p >= n_params) throw std::runtime_error("writer param ordinal " + std::to_string(p) + " >= n_params " + std::to_string(n_params));
        }
        // Validate CB data formats now so bad configs fail at compile.
        for (uint32_t f : prog->cb_formats) {
            (void)cb_data_format(f);
        }
    });
    if (has_error()) return UINT32_MAX;
    // Reuse freed slots first so ids stay dense.
    for (uint32_t i = 0; i < dev->programs.size(); i++) {
        if (!dev->programs[i]) {
            dev->programs[i] = prog;
            return i;
        }
    }
    uint32_t idx = static_cast<uint32_t>(dev->programs.size());
    dev->programs.push_back(prog);
    return idx;
}

extern "C" bool run_program(
    void* dev_raw, uint32_t prog_idx,
    const void** src_buffers, uint32_t n_src,
    const void** dst_buffers, uint32_t n_dst,
    uint32_t grid_rows, uint32_t grid_cols,
    const uint32_t* var_ordinals, const uint32_t* var_values, uint32_t n_vars)
{
    auto* dev = static_cast<DeviceHandle*>(dev_raw);
    if (prog_idx >= dev->programs.size() || !dev->programs[prog_idx]) {
        set_last_error("program index out of range");
        return false;
    }
    if (grid_rows == 0 || grid_cols == 0) {
        set_last_error("run_program: empty grid");
        return false;
    }
    auto& cfg = *dev->programs[prog_idx];
    uint32_t n_inputs = n_src;
    (void)n_dst;

    bool ok = true;
    catch_and_set([&] {
        std::unordered_map<uint32_t, uint32_t> vars;
        for (uint32_t i = 0; i < n_vars; i++) {
            vars[var_ordinals[i]] = var_values[i];
        }

        // Compile args per section: buffer params (Global / GlobalMut), in head
        // order; Variable params get no accessor. The buffer addresses vary per
        // launch, so the accessor args are built here and attached at kernel
        // creation.
        auto section_compile_args = [&](const std::vector<uint32_t>& params) {
            std::vector<uint32_t> args;
            uint32_t next_src = 0, next_dst = 0;
            for (uint32_t p : params) {
                if (vars.count(p)) continue;
                const BufferHandle* h =
                    (p >= n_inputs) ? static_cast<const BufferHandle*>(dst_buffers[next_dst++])
                                    : static_cast<const BufferHandle*>(src_buffers[next_src++]);
                if (!h || !h->buffer) throw std::runtime_error("run_program: null buffer handle");
                TensorAccessorArgs(h->buffer->get_backing_buffer()).append_to(args);
            }
            return args;
        };

        std::vector<uint32_t> reader_compile_args = section_compile_args(cfg.reader_params);
        std::vector<uint32_t> writer_compile_args = section_compile_args(cfg.writer_params);

        CoreCoord start_core{0, 0};
        CoreCoord end_core{grid_cols - 1, grid_rows - 1};
        CoreRangeSet all_cores(CoreRange(start_core, end_core));

        Program program = CreateProgram();
        MeshWorkload workload;
        MeshCoordinateRange device_range(dev->device->shape());

        for (size_t i = 0; i < cfg.cb_indices.size(); i++) {
            DataFormat df = cb_data_format(cfg.cb_formats[i]);
            CreateCircularBuffer(
                program, all_cores,
                CircularBufferConfig(
                    cfg.cb_num_tiles[i] * cfg.cb_tile_bytes[i],
                    {{static_cast<CBIndex>(cfg.cb_indices[i]), df}})
                    .set_page_size(static_cast<CBIndex>(cfg.cb_indices[i]), cfg.cb_tile_bytes[i]));
        }

        // Reader kernel (RISCV_0), writer kernel (RISCV_1).
        auto reader = CreateKernelFromString(
            program, cfg.reader_source, all_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
                .compile_args = reader_compile_args,
                .defines = {},
                .named_compile_args = {},
                .opt_level = KernelBuildOptLevel::O2,
                .compiler_include_paths = {},
            });
        auto writer = CreateKernelFromString(
            program, cfg.writer_source, all_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
                .compile_args = writer_compile_args,
                .defines = {},
                .named_compile_args = {},
                .opt_level = KernelBuildOptLevel::O2,
                .compiler_include_paths = {},
            });

        // Compute kernel (HiFi4), per-CB unpack-to-dest from the cached config.
        std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest;
        for (uint32_t f : cfg.cb_formats) {
            bool straight = (f == 0) && cfg.fp32_dest_acc_en;
            unpack_to_dest.push_back(straight ? tt::tt_metal::UnpackToDestMode::UnpackToDestFp32
                                              : tt::tt_metal::UnpackToDestMode::Default);
        }
        unpack_to_dest.resize(64, tt::tt_metal::UnpackToDestMode::Default);
        auto compute = CreateKernelFromString(
            program, cfg.compute_source, all_cores,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = cfg.fp32_dest_acc_en,
                .dst_full_sync_en = false,
                .unpack_to_dest_mode = unpack_to_dest,
                .bfp8_pack_precise = false,
                .math_approx_mode = false,
                .compile_args = {},
                .defines = {},
                .named_compile_args = {},
                .opt_level = KernelBuildOptLevel::O3,
                .compiler_include_paths = {},
            });

        // Per-core runtime args: Global + Variable params interleaved in head
        // order, then GlobalMut (tail of the head-order param list), then the
        // core's grid coordinates (row = gidx0, col = gidx1).
        auto section_rt_args = [&](const std::vector<uint32_t>& params, uint32_t row, uint32_t col) {
            std::vector<uint32_t> rt;
            uint32_t next_src = 0, next_dst = 0;
            for (uint32_t p : params) {
                auto vit = vars.find(p);
                if (vit != vars.end()) {
                    rt.push_back(vit->second);
                } else if (p >= n_inputs) {
                    const BufferHandle* h = static_cast<const BufferHandle*>(dst_buffers[next_dst++]);
                    rt.push_back(static_cast<uint32_t>(h->buffer->address()));
                } else {
                    const BufferHandle* h = static_cast<const BufferHandle*>(src_buffers[next_src++]);
                    rt.push_back(static_cast<uint32_t>(h->buffer->address()));
                }
            }
            rt.push_back(row);
            rt.push_back(col);
            return rt;
        };

        for (uint32_t row = 0; row < grid_rows; row++) {
            for (uint32_t col = 0; col < grid_cols; col++) {
                CoreCoord core{col, row};
                SetRuntimeArgs(program, reader, core, section_rt_args(cfg.reader_params, row, col));
                SetRuntimeArgs(program, writer, core, section_rt_args(cfg.writer_params, row, col));
                SetRuntimeArgs(program, compute, core, section_rt_args(cfg.compute_params, row, col));
            }
        }

        workload.add_program(device_range, std::move(program));
        EnqueueMeshWorkload(*dev->cq, workload, false);
        Finish(*dev->cq);
    });
    ok = !has_error();
    return ok;
}

extern "C" void destroy_program(void* dev_raw, uint32_t prog_idx) {
    auto* dev = static_cast<DeviceHandle*>(dev_raw);
    if (!dev) return;
    if (prog_idx < dev->programs.size()) {
        dev->programs[prog_idx].reset();
    }
}
