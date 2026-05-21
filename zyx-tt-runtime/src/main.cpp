// SPDX-FileCopyrightText: (c) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

// Long-lived C++ runtime process for zyx Tenstorrent backend.
// Reads JSON commands from stdin, executes kernels on tt-metal hardware,
// writes JSON responses to stdout.
//
// NOC addresses for input/output buffers are passed in the run command
// (allocated by the Rust side via ioctl). The reader kernel copies from
// src NOC DRAM → circular buffer, compute kernel applies SFPU op,
// writer kernel copies from circular buffer → dst NOC DRAM.
// No data crosses the IPC channel — only NOC addresses as uint64 values.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>

using namespace std;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

constexpr uint32_t TILE_ELEMS = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT; // 1024
constexpr uint32_t TILE_BYTES = sizeof(bfloat16) * TILE_ELEMS;                           // 2048

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
// Main IPC loop
// ---------------------------------------------------------------------------

int main() {
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

            try {
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
            uint32_t n_tiles = extract_u32(line, "n_tiles");
            uint64_t src_noc = extract_u64(line, "src_noc");
            uint64_t dst_noc = extract_u64(line, "dst_noc");

            if (hash.empty()) {
                cout << R"({"status":"error","msg":"missing hash"})" << endl;
                continue;
            }
            if (n_tiles == 0) n_tiles = 1;

            try {
                // --- Create program (fresh each run; JIT build cache avoids recompilation) ---
                Program program = CreateProgram();
                CoreCoord core = {0, 0};

                // Circular buffers (input = c_0, output = c_16)
                CreateCircularBuffer(program, core,
                    CircularBufferConfig(2 * TILE_BYTES, {{CBIndex::c_0, DataFormat::Float16_b}})
                        .set_page_size(CBIndex::c_0, TILE_BYTES));
                CreateCircularBuffer(program, core,
                    CircularBufferConfig(2 * TILE_BYTES, {{CBIndex::c_16, DataFormat::Float16_b}})
                        .set_page_size(CBIndex::c_16, TILE_BYTES));

                // Reader kernel (static) — reads from src_noc DRAM → CB
                string reader_path = kernel_dir + "/reader.cpp";
                vector<uint32_t> empty_args;
                KernelHandle reader_id = CreateKernel(program, reader_path, core,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                        .compile_args = empty_args});

                // Writer kernel (static) — writes from CB → dst_noc DRAM
                string writer_path = kernel_dir + "/writer.cpp";
                KernelHandle writer_id = CreateKernel(program, writer_path, core,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .compile_args = empty_args});

                // Compute kernel (generated per hash)
                string compute_path = cache_dir + "/" + hash + ".cpp";
                KernelHandle compute_id = CreateKernel(program, compute_path, core,
                    ComputeConfig{
                        .math_fidelity = MathFidelity::HiFi4,
                        .math_approx_mode = false});

                // --- Set runtime args ---
                // Reader: (src_noc_low, src_noc_high, n_tiles)
                SetRuntimeArgs(program, reader_id, core, {
                    (uint32_t)(src_noc & 0xFFFFFFFF),
                    (uint32_t)(src_noc >> 32),
                    n_tiles});
                // Writer: (dst_noc_low, dst_noc_high, n_tiles)
                SetRuntimeArgs(program, writer_id, core, {
                    (uint32_t)(dst_noc & 0xFFFFFFFF),
                    (uint32_t)(dst_noc >> 32),
                    n_tiles});
                // Compute: (n_tiles)
                SetRuntimeArgs(program, compute_id, core, {n_tiles});

                // --- Enqueue and run ---
                MeshWorkload workload;
                MeshCoordinateRange device_range(mesh_device->shape());
                workload.add_program(device_range, move(program));
                EnqueueMeshWorkload(*cq, workload, false);
                Finish(*cq);

                cout << R"({"status":"ok"})" << endl;

            } catch (const exception& e) {
                cerr << "run error: " << e.what() << endl;
                cout << R"({"status":"error","msg":")" << e.what() << R"("})" << endl;
            }
        }

        // ---- exit ----
        else if (cmd == "exit") {
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
