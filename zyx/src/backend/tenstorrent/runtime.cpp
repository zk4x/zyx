// SPDX-FileCopyrightText: (c) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

// Long-lived C++ runtime process for zyx Tenstorrent backend.
// Reads JSON commands from stdin, executes kernels on tt-metal hardware,
// writes JSON responses to stdout.
//
// Tensor data is transferred via a shared memory region (shm_open) whose
// path is received during init. Input/output offsets within this region
// are passed in the run command.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>

using namespace std;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

constexpr uint32_t TILE_ELEMS = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT; // 1024
constexpr uint32_t TILE_BYTES_BF16 = sizeof(bfloat16) * TILE_ELEMS;                      // 2048

// ---------------------------------------------------------------------------
// Helper: parse DataFormat from integer (matches Rust DataFormat enum)
// ---------------------------------------------------------------------------

static DataFormat int_to_data_format(uint32_t df) {
    switch (df) {
        case 0: return DataFormat::Float32;
        case 1: return DataFormat::Float16;
        case 2: return DataFormat::Float16_b;
        case 3: return DataFormat::Bfp8;
        default: return DataFormat::Float32;
    }
}

// ---------------------------------------------------------------------------
// Cache directory resolution (XDG convention, like zyx C backend)
// ---------------------------------------------------------------------------

static string default_cache_dir() {
    const char* xdg = getenv("XDG_CONFIG_HOME");
    if (xdg && xdg[0] == '/') {
        return string(xdg) + "/zyx/cache/tt";
    }
    const char* home = getenv("HOME");
    if (home) {
        return string(home) + "/.config/zyx/cache/tt";
    }
    return "";
}

// ---------------------------------------------------------------------------
// Minimal JSON helpers (no external dependency)
// ---------------------------------------------------------------------------

static string trim(string s) {
    auto f = s.find_first_not_of(" \t\r\n");
    if (f == string::npos) return "";
    auto l = s.find_last_not_of(" \t\r\n");
    return s.substr(f, l - f + 1);
}

static string extract_str(const string& json, const string& key) {
    auto k = json.find("\"" + key + "\"");
    if (k == string::npos) return "";
    auto sep = json.find(':', k);
    if (sep == string::npos) return "";
    auto start = json.find_first_of("\"", sep);
    if (start == string::npos) return "";
    ++start;
    auto end = json.find("\"", start);
    if (end == string::npos) return "";
    return json.substr(start, end - start);
}

static uint32_t extract_u32(const string& json, const string& key) {
    auto k = json.find("\"" + key + "\"");
    if (k == string::npos) return 0;
    auto sep = json.find(':', k);
    if (sep == string::npos) return 0;
    auto start = json.find_first_of("0123456789", sep);
    if (start == string::npos) return 0;
    size_t end = 0;
    return (uint32_t)stoul(json.substr(start), &end);
}

static uint64_t extract_u64(const string& json, const string& key) {
    auto k = json.find("\"" + key + "\"");
    if (k == string::npos) return 0;
    auto sep = json.find(':', k);
    if (sep == string::npos) return 0;
    auto start = json.find_first_of("0123456789", sep);
    if (start == string::npos) return 0;
    size_t end = 0;
    return stoull(json.substr(start), &end);
}

// ---------------------------------------------------------------------------
// Shared memory state
// ---------------------------------------------------------------------------

struct ShmState {
    int fd = -1;
    void* ptr = nullptr;
    size_t size = 0;
} g_shm;

static void shm_init(const string& path) {
    if (g_shm.fd >= 0) close(g_shm.fd);
    g_shm.fd = shm_open(path.c_str(), O_RDWR, 0);
    if (g_shm.fd < 0) {
        throw runtime_error("shm_open " + path + " failed");
    }
    struct stat st;
    fstat(g_shm.fd, &st);
    g_shm.size = st.st_size;
    g_shm.ptr = mmap(nullptr, g_shm.size, PROT_READ | PROT_WRITE, MAP_SHARED, g_shm.fd, 0);
    if (g_shm.ptr == MAP_FAILED) {
        close(g_shm.fd);
        g_shm.fd = -1;
        throw runtime_error("mmap shm failed");
    }
}

static void shm_close() {
    if (g_shm.ptr && g_shm.ptr != MAP_FAILED) munmap(g_shm.ptr, g_shm.size);
    if (g_shm.fd >= 0) close(g_shm.fd);
    g_shm.ptr = nullptr;
    g_shm.fd = -1;
    g_shm.size = 0;
}

// ---------------------------------------------------------------------------
// Main IPC loop
// ---------------------------------------------------------------------------

int main() {
    if (!getenv("TT_METAL_RUNTIME_ROOT")) {
        setenv("TT_METAL_RUNTIME_ROOT", TT_METAL_ROOT_DEFAULT, 0);
    }

    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string kernel_dir;
    string cache_dir;
    shared_ptr<MeshDevice> mesh_device = nullptr;
    MeshCommandQueue* cq = nullptr;

    string line;
    while (getline(cin, line)) {
        line = trim(line);
        if (line.empty()) continue;

        string cmd = extract_str(line, "cmd");

        // ---- init ----
        if (cmd == "init") {
            kernel_dir = extract_str(line, "kernel_dir");
            cache_dir = extract_str(line, "cache_dir");
            if (cache_dir.empty()) {
                cache_dir = default_cache_dir();
            }
            if (!cache_dir.empty()) {
                string mkdir_cmd = "mkdir -p " + cache_dir;
                (void)system(mkdir_cmd.c_str());
            }

            // Open shared memory
            string shm_path = extract_str(line, "shm_path");
            try {
                shm_init(shm_path);
                mesh_device = MeshDevice::create_unit_mesh(0);
                cq = &mesh_device->mesh_command_queue();
                cout << R"({"status":"ready"})" << endl;
            } catch (const exception& e) {
                cerr << "init error: " << e.what() << endl;
                cout << R"({"status":"error","msg":")" << e.what() << R"("})" << endl;
            }
        }

        // ---- run ----
        else if (cmd == "run") {
            if (!mesh_device.get()) {
                cout << R"({"status":"error","msg":"not initialized"})" << endl;
                continue;
            }

            string hash = extract_str(line, "hash");
            uint32_t n_inputs = extract_u32(line, "n_inputs");
            uint32_t n_outputs = extract_u32(line, "n_outputs");
            uint32_t n_tiles = extract_u32(line, "n_tiles");
            uint32_t data_format = extract_u32(line, "data_format");
            uint32_t tile_bytes = extract_u32(line, "tile_bytes");

            // Parse input offsets/sizes: in_ofs0, in_sz0, in_ofs1, in_sz1, ...
            vector<uint64_t> in_offsets(n_inputs);
            vector<uint64_t> in_sizes(n_inputs);
            for (uint32_t i = 0; i < n_inputs; i++) {
                in_offsets[i] = extract_u64(line, "in_ofs" + to_string(i));
                in_sizes[i] = extract_u64(line, "in_sz" + to_string(i));
            }
            uint64_t out_offset = extract_u64(line, "out_ofs");
            uint64_t out_size = extract_u64(line, "out_sz");

            (void)n_outputs;

            if (hash.empty()) {
                cout << R"({"status":"error","msg":"missing hash"})" << endl;
                continue;
            }
            if (n_tiles == 0) n_tiles = 1;

            try {
                // --- Allocate DRAM buffers ---
                DataFormat df = int_to_data_format(data_format);

                DeviceLocalBufferConfig dram_config{};
                dram_config.page_size = tile_bytes;
                dram_config.buffer_type = BufferType::DRAM;
                ReplicatedBufferConfig buf_config{.size = (uint64_t)n_tiles * tile_bytes};

                auto src0_buf = MeshBuffer::create(buf_config, dram_config, mesh_device.get());
                auto src1_buf = n_inputs > 1 ? MeshBuffer::create(buf_config, dram_config, mesh_device.get()) : nullptr;
                auto dst_buf  = MeshBuffer::create(buf_config, dram_config, mesh_device.get());

                // --- Copy input data from shared memory → host vectors → DRAM ---
                auto vec_from_shm = [&](uint64_t ofs, uint64_t sz) {
                    vector<bfloat16> v(sz / sizeof(bfloat16));
                    memcpy(v.data(), (uint8_t*)g_shm.ptr + ofs, sz);
                    return v;
                };

                vector<bfloat16> a_data = vec_from_shm(in_offsets[0], in_sizes[0]);
                EnqueueWriteMeshBuffer(*cq, src0_buf, a_data, false);
                if (n_inputs > 1) {
                    vector<bfloat16> b_data = vec_from_shm(in_offsets[1], in_sizes[1]);
                    EnqueueWriteMeshBuffer(*cq, *src1_buf, b_data, false);
                }

                // --- Create program ---
                Program program = CreateProgram();
                CoreCoord core = {0, 0};
                MeshWorkload workload;
                MeshCoordinateRange device_range(mesh_device->shape());

                // Circular buffers
                constexpr uint32_t tiles_per_cb = 2;
                auto mk_cb = [&](CBIndex idx) {
                    CreateCircularBuffer(program, core,
                        CircularBufferConfig(tiles_per_cb * tile_bytes, {{idx, df}})
                            .set_page_size(idx, tile_bytes));
                };
                mk_cb(CBIndex::c_0);
                if (n_inputs > 1) mk_cb(CBIndex::c_1);
                mk_cb(CBIndex::c_16);

                // Reader kernel (read_tiles.cpp)
                vector<uint32_t> reader_args;
                TensorAccessorArgs(*src0_buf).append_to(reader_args);
                if (src1_buf) TensorAccessorArgs(*src1_buf).append_to(reader_args);

                auto reader = CreateKernel(program, kernel_dir + "/read_tiles.cpp", core,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .compile_args = reader_args});

                // Writer kernel (write_tile.cpp)
                vector<uint32_t> writer_args;
                TensorAccessorArgs(*dst_buf).append_to(writer_args);
                auto writer = CreateKernel(program, kernel_dir + "/write_tile.cpp", core,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                        .compile_args = writer_args});

                // Compute kernel (tiles_add.cpp or cached per-hash kernel)
                string compute_path = cache_dir + "/" + hash + ".cpp";
                // If cached kernel doesn't exist, fall back to tiles_add.cpp
                if (access(compute_path.c_str(), F_OK) != 0) {
                    compute_path = kernel_dir + "/tiles_add.cpp";
                }
                auto compute = CreateKernel(program, compute_path, core,
                    ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

                // Set runtime args
                if (n_inputs > 1) {
                    SetRuntimeArgs(program, reader, core, {(uint32_t)src0_buf->address(), (uint32_t)src1_buf->address(), n_tiles});
                } else {
                    SetRuntimeArgs(program, reader, core, {(uint32_t)src0_buf->address(), 0, n_tiles});
                }
                SetRuntimeArgs(program, writer, core, {(uint32_t)dst_buf->address(), n_tiles});
                SetRuntimeArgs(program, compute, core, {n_tiles});

                // --- Enqueue and run ---
                workload.add_program(device_range, move(program));
                EnqueueMeshWorkload(*cq, workload, false);
                Finish(*cq);

                // --- Read result back to shared memory ---
                vector<bfloat16> result;
                EnqueueReadMeshBuffer(*cq, result, dst_buf, true);
                memcpy((uint8_t*)g_shm.ptr + out_offset, result.data(), out_size);

                cout << R"({"status":"ok"})" << endl;

            } catch (const exception& e) {
                cerr << "run error: " << e.what() << endl;
                cout << R"({"status":"error","msg":")" << e.what() << R"("})" << endl;
            }
        }

        // ---- exit ----
        else if (cmd == "exit") {
            if (mesh_device.get()) mesh_device->close();
            shm_close();
            cout << R"({"status":"bye"})" << endl;
            break;
        }

        else {
            cout << R"({"status":"error","msg":"unknown cmd: )" << cmd << R"("})" << endl;
        }
    }

    return 0;
}
