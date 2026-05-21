// SPDX-FileCopyrightText: (c) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

// Long-lived C++ runtime process for zyx Tenstorrent backend.
// Reads JSON commands from stdin, executes kernels on tt-metal hardware,
// writes JSON responses to stdout.

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace std;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

constexpr uint32_t TILE_ELEMS = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT; // 1024
constexpr uint32_t TILE_BYTES = sizeof(bfloat16) * TILE_ELEMS;                           // 2048

// ---------------------------------------------------------------------------
// bfloat16 raw-bit helpers (the class only exposes float conversion)
// ---------------------------------------------------------------------------

static bfloat16 bf16_from_raw(uint16_t raw) {
    uint32_t tmp = (uint32_t)raw << 16;
    float f;
    memcpy(&f, &tmp, sizeof(f));
    return bfloat16(f);
}

static uint16_t bf16_to_raw(bfloat16 v) {
    float f = (float)v;
    uint32_t tmp;
    memcpy(&tmp, &f, sizeof(tmp));
    return (uint16_t)(tmp >> 16);
}

// ---------------------------------------------------------------------------
// Minimal JSON helpers (no external dependency)
// ---------------------------------------------------------------------------

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

static vector<bfloat16> hex_to_bf16(const string& hex) {
    vector<bfloat16> out;
    for (size_t i = 0; i + 3 < hex.size(); i += 4) {
        auto sub = hex.substr(i, 4);
        uint16_t val = (uint16_t)stoul(sub, nullptr, 16);
        out.push_back(bf16_from_raw(val));
    }
    return out;
}

static string bf16_to_hex(const vector<bfloat16>& v) {
    string out;
    for (auto& x : v) {
        char buf[5];
        snprintf(buf, sizeof(buf), "%04x", bf16_to_raw(x));
        out += buf;
    }
    return out;
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
            string input_hex = extract_str(line, "input");

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

                // Reader kernel (static)
                string reader_path = kernel_dir + "/reader.cpp";
                vector<uint32_t> empty_args;
                KernelHandle reader_id = CreateKernel(program, reader_path, core,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                        .compile_args = empty_args});

                // Writer kernel (static)
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

                // --- Allocate DRAM buffers via tt-metal ---
                auto mk_buf = [&](uint32_t sz) {
                    return MeshBuffer::create(
                        ReplicatedBufferConfig{.size = sz},
                        DeviceLocalBufferConfig{
                            .page_size = TILE_BYTES,
                            .buffer_type = BufferType::DRAM},
                        mesh_device.get());
                };
                auto src_buf = mk_buf(TILE_BYTES * n_tiles);
                auto dst_buf = mk_buf(TILE_BYTES * n_tiles);

                // --- Write input data to device ---
                vector<bfloat16> src_vec;
                if (!input_hex.empty()) {
                    src_vec = hex_to_bf16(input_hex);
                } else {
                    mt19937 rng(random_device{}());
                    uniform_real_distribution<float> dist(0.f, 1.0f);
                    src_vec.resize(n_tiles * TILE_ELEMS);
                    for (auto& v : src_vec) v = bfloat16(dist(rng));
                }
                src_vec.resize(n_tiles * TILE_ELEMS, bfloat16(0.0f));
                EnqueueWriteMeshBuffer(*cq, src_buf, src_vec, false);

                // --- Set runtime args ---
                SetRuntimeArgs(program, compute_id, core, {n_tiles});
                SetRuntimeArgs(program, reader_id, core, {(uint32_t)src_buf->address(), n_tiles});
                SetRuntimeArgs(program, writer_id, core, {(uint32_t)dst_buf->address(), n_tiles});

                // --- Enqueue and run ---
                MeshWorkload workload;
                MeshCoordinateRange device_range(mesh_device->shape());
                workload.add_program(device_range, move(program));
                EnqueueMeshWorkload(*cq, workload, false);
                Finish(*cq);

                // --- Read output ---
                vector<bfloat16> result;
                EnqueueReadMeshBuffer(*cq, result, dst_buf, true);

                // --- Verify and respond ---
                bool pass = true;
                for (uint32_t i = 0; i < src_vec.size(); i++) {
                    float expected = exp2((float)src_vec[i]);
                    float actual = (float)result[i];
                    if (fabs(expected - actual) > 5e-2f) {
                        pass = false;
                        if (i < 10) {
                            cerr << "MISMATCH[" << i << "]: exp2(" << (float)src_vec[i] << ") = "
                                 << expected << " (got " << actual << ")" << endl;
                        }
                    }
                }

                string out_hex = bf16_to_hex(result);
                cout << R"({"status":")" << (pass ? "ok" : "mismatch")
                     << R"(","output":")" << out_hex << R"("})" << endl;

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
