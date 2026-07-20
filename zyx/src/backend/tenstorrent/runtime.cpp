// SPDX-FileCopyrightText: (c) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

// Long-lived C++ runtime process for zyx Tenstorrent backend.
// Reads JSON commands from stdin, manages persistent DRAM buffers,
// executes kernels on tt-metal hardware, writes JSON responses to stdout.
//
// Tensor data is transferred via temporary shared memory regions
// (shm_open + unlink per transfer), created by the Rust side.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
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
// Cache directory resolution (XDG convention)
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
// Temporary shared memory: open by path, mmap, read/write, close, unlink
// ---------------------------------------------------------------------------

struct TempShm {
    int fd = -1;
    void* ptr = nullptr;
    size_t size = 0;

    ~TempShm() { close(); }

    void open_read(const string& path) {
        close();
        fd = shm_open(path.c_str(), O_RDWR, 0);
        if (fd < 0) throw runtime_error("shm_open " + path + " failed (read)");
        struct stat st;
        fstat(fd, &st);
        size = st.st_size;
        ptr = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) { close(); throw runtime_error("mmap shm read failed"); }
    }

    void open_write(const string& path, uint64_t sz) {
        close();
        fd = shm_open(path.c_str(), O_RDWR, 0);
        if (fd < 0) throw runtime_error("shm_open " + path + " failed (write)");
        size = sz;
        ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) { close(); throw runtime_error("mmap shm write failed"); }
    }

    void close() {
        if (ptr && ptr != MAP_FAILED) munmap(ptr, size);
        if (fd >= 0) ::close(fd);
        fd = -1; ptr = nullptr; size = 0;
    }
};

// ---------------------------------------------------------------------------
// Main IPC loop
// ---------------------------------------------------------------------------

int main() {
    if (!getenv("TT_METAL_RUNTIME_ROOT")) {
        setenv("TT_METAL_RUNTIME_ROOT", TT_METAL_ROOT_DEFAULT, 0);
    }

    // Keep default sync for pipe compatibility with Rust BufWriter
    // ios::sync_with_stdio(false);
    // cin.tie(nullptr);

    string kernel_dir;
    string cache_dir;
    shared_ptr<MeshDevice> mesh_device = nullptr;
    MeshCommandQueue* cq = nullptr;

    // Persistent DRAM buffers — indexed by position in this vector
    vector<shared_ptr<MeshBuffer>> buffers;

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

            try {
                mesh_device = MeshDevice::create_unit_mesh(0);
                cq = &mesh_device->mesh_command_queue();
                cout << R"({"status":"ready"})" << endl;
            } catch (const exception& e) {
                cerr << "init error: " << e.what() << endl;
                cout << R"({"status":"error","msg":")" << e.what() << R"("})" << endl;
            }
        }

        // ---- alloc_buf ----
        else if (cmd == "alloc_buf") {
            if (!mesh_device.get()) {
                cout << R"({"status":"error","msg":"not initialized"})" << endl;
                continue;
            }

            uint64_t size = extract_u64(line, "size");
            if (size == 0) {
                cout << R"({"status":"error","msg":"alloc_buf: zero size"})" << endl;
                continue;
            }

            try {
                uint32_t tile_bytes = extract_u32(line, "tile_bytes");
                if (tile_bytes == 0) tile_bytes = 4096;
                uint32_t n_tiles = (size + tile_bytes - 1) / tile_bytes;
                if (n_tiles == 0) n_tiles = 1;

                DeviceLocalBufferConfig dram_config{};
                dram_config.page_size = tile_bytes;
                dram_config.buffer_type = BufferType::DRAM;
                ReplicatedBufferConfig buf_config{.size = (uint64_t)n_tiles * tile_bytes};

                auto buf = MeshBuffer::create(buf_config, dram_config, mesh_device.get());
                uint32_t idx = buffers.size();
                cerr << "[TT_ALLOC] idx=" << idx << " size=" << size << " page=" << tile_bytes << " addr=" << buf->address() << " actual_sz=" << buf->size() << endl;
                buffers.push_back(move(buf));
                cout << R"({"status":"ok","index":")" << idx << R"("})" << endl;
            } catch (const exception& e) {
                cerr << "alloc_buf error: " << e.what() << endl;
                cout << R"({"status":"error","msg":")" << e.what() << R"("})" << endl;
            }
        }

        // ---- free_buf ----
        else if (cmd == "free_buf") {
            uint32_t idx = extract_u32(line, "index");
            if (idx < buffers.size() && buffers[idx]) {
                buffers[idx].reset();
            }
            cout << R"({"status":"ok"})" << endl;
        }

        // ---- write_buf ----
        else if (cmd == "write_buf") {
            uint32_t idx = extract_u32(line, "index");
            string shm_path = extract_str(line, "shm_path");
            uint64_t size = extract_u64(line, "size");

            if (idx >= buffers.size() || !buffers[idx]) {
                cout << R"({"status":"error","msg":"write_buf: invalid index ")" << idx << R"("})" << endl;
                continue;
            }

            try {
                TempShm shm;
                shm.open_read(shm_path);

                uint64_t buf_bytes = buffers[idx]->size();
                uint64_t num_float = buf_bytes / sizeof(float);
                vector<float> data(num_float, 0);
                uint64_t copy_sz = min(size, shm.size);
                memcpy(data.data(), shm.ptr, copy_sz);
                shm.close();

                EnqueueWriteMeshBuffer(*cq, buffers[idx], data, false);
                // Skip Finish here — let the workload Finish drain everything
                // Unlink the temp shm (Rust side already did munmap)
                shm_unlink(shm_path.c_str());

                cout << R"({"status":"ok"})" << endl;
            } catch (const exception& e) {
                cerr << "write_buf error: " << e.what() << endl;
                cout << R"({"status":"error","msg":")" << e.what() << R"("})" << endl;
            }
        }

        // ---- read_buf ----
        else if (cmd == "read_buf") {
            uint32_t idx = extract_u32(line, "index");
            string shm_path = extract_str(line, "shm_path");
            uint64_t size = extract_u64(line, "size");

            if (idx >= buffers.size() || !buffers[idx]) {
                cout << R"({"status":"error","msg":"read_buf: invalid index ")" << idx << R"("})" << endl;
                continue;
            }

            try {
                vector<float> result;
                EnqueueReadMeshBuffer(*cq, result, buffers[idx], true);
                // blocking=true already does Finish internally

                uint64_t copy_sz = min(size, result.size() * sizeof(float));

                TempShm shm;
                shm.open_write(shm_path, copy_sz);
                memcpy(shm.ptr, result.data(), copy_sz);
                shm.close();

                // Rust side will munmap + shm_unlink after reading
                cout << R"({"status":"ok"})" << endl;
            } catch (const exception& e) {
                cerr << "read_buf error: " << e.what() << endl;
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
            (void)n_outputs;

            // Parse buffer indices: src0, src1, ..., dst0
            vector<uint32_t> src_indices(n_inputs);
            for (uint32_t i = 0; i < n_inputs; i++) {
                src_indices[i] = extract_u32(line, "src" + to_string(i));
            }
            uint32_t dst_index = extract_u32(line, "dst0");

            if (hash.empty()) {
                cout << R"({"status":"error","msg":"missing hash"})" << endl;
                continue;
            }
            if (n_tiles == 0) n_tiles = 1;

            // Validate indices
            for (uint32_t i = 0; i < n_inputs; i++) {
                if (src_indices[i] >= buffers.size() || !buffers[src_indices[i]]) {
                    cout << R"({"status":"error","msg":"run: invalid src index )" << src_indices[i] << R"("})" << endl;
                    continue;
                }
            }
            if (dst_index >= buffers.size() || !buffers[dst_index]) {
                cout << R"({"status":"error","msg":"run: invalid dst index )" << dst_index << R"("})" << endl;
                continue;
            }

            try {
                // Use f32 format (matches Rust f32 data)
                (void)data_format;
                DataFormat df = DataFormat::Float32;
                tile_bytes = sizeof(float) * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;

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
                for (uint32_t i = 0; i < n_inputs; i++) {
                    mk_cb(static_cast<CBIndex>(static_cast<uint32_t>(CBIndex::c_0) + i));
                }
                mk_cb(CBIndex::c_16);

                // Reader kernel — one TensorAccessor per input
                vector<uint32_t> reader_args;
                for (uint32_t i = 0; i < n_inputs; i++) {
                    TensorAccessorArgs(*buffers[src_indices[i]]).append_to(reader_args);
                }
                auto reader = CreateKernel(program, kernel_dir + "/read_tiles.cpp", core,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .compile_args = reader_args});

                // Writer kernel
                vector<uint32_t> writer_args;
                TensorAccessorArgs(*buffers[dst_index]).append_to(writer_args);
                auto writer = CreateKernel(program, kernel_dir + "/write_tile.cpp", core,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                        .compile_args = writer_args});

                // Always use tiles_add.cpp for now (no per-hash kernel cache)
                auto compute = CreateKernel(program, kernel_dir + "/tiles_add.cpp", core,
                    ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

                // Set runtime args — pass DRAM addresses + n_tiles
                cerr << "[TT] setting rt args n_tiles=" << n_tiles << endl;
                {
                    vector<uint32_t> reader_rt_args;
                    for (uint32_t i = 0; i < n_inputs; i++) {
                        uint64_t a = buffers[src_indices[i]]->address();
                        cerr << "[TT]  src" << i << " idx=" << src_indices[i] << " addr=" << a << " sz=" << buffers[src_indices[i]]->size() << endl;
                        reader_rt_args.push_back((uint32_t)a);
                    }
                    reader_rt_args.push_back(n_tiles);
                    SetRuntimeArgs(program, reader, core, reader_rt_args);
                }
                {
                    uint64_t a = buffers[dst_index]->address();
                    cerr << "[TT]  dst idx=" << dst_index << " addr=" << a << " sz=" << buffers[dst_index]->size() << endl;
                    SetRuntimeArgs(program, writer, core, {(uint32_t)a, n_tiles});
                }
                SetRuntimeArgs(program, compute, core, {n_tiles});

                cerr << "[TT] before add_program" << endl;
                workload.add_program(device_range, move(program));
                cerr << "[TT] before EnqueueMeshWorkload" << endl;
                EnqueueMeshWorkload(*cq, workload, false);
                cerr << "[TT] before Finish" << endl;
                Finish(*cq);
                cerr << "[TT] after Finish" << endl;

                cout << R"({"status":"ok"})" << endl;

            } catch (const exception& e) {
                cerr << "run error: " << e.what() << endl;
                cout << R"({"status":"error","msg":")" << e.what() << R"("})" << endl;
            }
        }

        // ---- exit ----
        else if (cmd == "exit") {
            buffers.clear();
            if (mesh_device.get()) mesh_device->close();
            cout << R"({"status":"bye"})" << endl;
            break;
        }

        else {
            cout << R"({"status":"error","msg":"unknown cmd: )" << cmd << R"("})" << endl;
        }
    }

    return 0;
}
