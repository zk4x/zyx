// SPDX-FileCopyrightText: (c) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

// Long-lived C++ runtime process for zyx Tenstorrent backend.
// Reads JSON commands from stdin, manages persistent DRAM buffers,
// executes kernels on tt-metal hardware, writes JSON responses to stdout.
//
// Tensor data is transferred via temporary shared memory regions
// (shm_open + unlink per transfer), created by the Rust side.

#include "tt-metalium/kernel_types.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace std;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

constexpr uint32_t PAGE_SIZE = 4096;

// ---------------------------------------------------------------------------
// Cache directory resolution (XDG convention)
// ---------------------------------------------------------------------------

static string default_cache_dir() {
  const char *xdg = getenv("XDG_CONFIG_HOME");
  if (xdg && xdg[0] == '/') {
    return string(xdg) + "/zyx/cache/tt";
  }
  const char *home = getenv("HOME");
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
  if (f == string::npos)
    return "";
  auto l = s.find_last_not_of(" \t\r\n");
  return s.substr(f, l - f + 1);
}

static string extract_str(const string &json, const string &key) {
  auto k = json.find("\"" + key + "\"");
  if (k == string::npos)
    return "";
  auto sep = json.find(':', k);
  if (sep == string::npos)
    return "";
  auto start = json.find_first_of("\"", sep);
  if (start == string::npos)
    return "";
  ++start;
  auto end = json.find("\"", start);
  if (end == string::npos)
    return "";
  return json.substr(start, end - start);
}

static uint32_t extract_u32(const string &json, const string &key) {
  auto k = json.find("\"" + key + "\"");
  if (k == string::npos)
    return 0;
  auto sep = json.find(':', k);
  if (sep == string::npos)
    return 0;
  auto start = json.find_first_of("0123456789", sep);
  if (start == string::npos)
    return 0;
  size_t end = 0;
  return (uint32_t)stoul(json.substr(start), &end);
}

static uint64_t extract_u64(const string &json, const string &key) {
  auto k = json.find("\"" + key + "\"");
  if (k == string::npos)
    return 0;
  auto sep = json.find(':', k);
  if (sep == string::npos)
    return 0;
  auto start = json.find_first_of("0123456789", sep);
  if (start == string::npos)
    return 0;
  size_t end = 0;
  return stoull(json.substr(start), &end);
}

// ---------------------------------------------------------------------------
// Temporary shared memory: open by path, mmap, read/write, close, unlink
// ---------------------------------------------------------------------------

struct TempShm {
  int fd = -1;
  void *ptr = nullptr;
  size_t size = 0;

  ~TempShm() { close(); }

  void open_read(const string &path) {
    close();
    fd = shm_open(path.c_str(), O_RDWR, 0);
    if (fd < 0)
      throw runtime_error("shm_open " + path + " failed (read)");
    struct stat st;
    fstat(fd, &st);
    size = st.st_size;
    ptr = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
      close();
      throw runtime_error("mmap shm read failed");
    }
  }

  void open_write(const string &path, uint64_t sz) {
    close();
    fd = shm_open(path.c_str(), O_RDWR, 0);
    if (fd < 0)
      throw runtime_error("shm_open " + path + " failed (write)");
    size = sz;
    ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
      close();
      throw runtime_error("mmap shm write failed");
    }
  }

  void close() {
    if (ptr && ptr != MAP_FAILED)
      munmap(ptr, size);
    if (fd >= 0)
      ::close(fd);
    fd = -1;
    ptr = nullptr;
    size = 0;
  }
};

// ---------------------------------------------------------------------------
// Main IPC loop
// ---------------------------------------------------------------------------

// Compiled program configs cached by sequential ID (matching Rust's
// DeviceProgramId)
struct ProgramConfig {
  string reader_source;
  string compute_source;
  string writer_source;
  vector<uint32_t> cb_indices;
  vector<uint32_t> cb_formats;
  vector<uint32_t> cb_tile_bytes;
};

int main() {
  cerr << "[TT_CPP] runtime started" << endl;
  if (!getenv("TT_METAL_RUNTIME_ROOT")) {
    setenv("TT_METAL_RUNTIME_ROOT", TT_METAL_ROOT_DEFAULT, 0);
  }

  // Keep default sync for pipe compatibility with Rust BufWriter
  // ios::sync_with_stdio(false);
  // cin.tie(nullptr);

  string kernel_dir;
  string cache_dir;
  shared_ptr<MeshDevice> mesh_device = nullptr;
  MeshCommandQueue *cq = nullptr;

  vector<shared_ptr<MeshBuffer>> buffers;
  vector<ProgramConfig> program_cache;

  string line;
  while (getline(cin, line)) {
    line = trim(line);
    if (line.empty())
      continue;

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
        cerr << "[TT_CPP] calling create_unit_mesh(0)" << endl;
        mesh_device = MeshDevice::create_unit_mesh(0);
        cq = &mesh_device->mesh_command_queue();
        cout << R"({"status":"ready"})" << endl;
      } catch (const exception &e) {
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
        uint32_t n_pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
        if (n_pages == 0)
          n_pages = 1;

        DeviceLocalBufferConfig dram_config{};
        dram_config.page_size = PAGE_SIZE;
        dram_config.buffer_type = BufferType::DRAM;
        ReplicatedBufferConfig buf_config{.size =
                                              (uint64_t)n_pages * PAGE_SIZE};

        auto buf =
            MeshBuffer::create(buf_config, dram_config, mesh_device.get());
        uint32_t idx = buffers.size();
        cerr << "[TT_ALLOC] idx=" << idx << " size=" << size
             << " page=" << PAGE_SIZE << " addr=" << buf->address()
             << " actual_sz=" << buf->size() << endl;
        buffers.push_back(buf);
        cout << R"({"status":"ok","index":")" << idx << R"("})" << endl;
      } catch (const exception &e) {
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
        cout << R"({"status":"error","msg":"write_buf: invalid index ")" << idx
             << R"("})" << endl;
        continue;
      }

      try {
        TempShm shm;
        shm.open_read(shm_path);

        uint64_t buf_bytes = buffers[idx]->size();
        vector<uint8_t> data(buf_bytes, 0);
        uint64_t copy_sz = min(size, shm.size);
        memcpy(data.data(), shm.ptr, copy_sz);
        shm.close();

        EnqueueWriteMeshBuffer(*cq, buffers[idx], data, false);
        Finish(*cq);
        // Unlink the temp shm (Rust side already did munmap)
        shm_unlink(shm_path.c_str());

        cout << R"({"status":"ok"})" << endl;
      } catch (const exception &e) {
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
        cout << R"({"status":"error","msg":"read_buf: invalid index ")" << idx
             << R"("})" << endl;
        continue;
      }

      try {
        vector<uint8_t> result;
        EnqueueReadMeshBuffer(*cq, result, buffers[idx], true);
        // blocking=true already does Finish internally

        uint64_t copy_sz = min(size, result.size());

        TempShm shm;
        shm.open_write(shm_path, copy_sz);
        memcpy(shm.ptr, result.data(), copy_sz);
        shm.close();

        // Rust side will munmap + shm_unlink after reading
        cout << R"({"status":"ok"})" << endl;
      } catch (const exception &e) {
        cerr << "read_buf error: " << e.what() << endl;
        cout << R"({"status":"error","msg":")" << e.what() << R"("})" << endl;
      }
    }

    // ---- compile_program ----
    else if (cmd == "compile_program") {
      if (!mesh_device.get()) {
        cout << R"({"status":"error","msg":"not initialized"})" << endl;
        continue;
      }

      uint32_t id = extract_u32(line, "id");
      uint32_t n_cbs = extract_u32(line, "n_cbs");

      try {
        vector<uint32_t> cb_indices(n_cbs);
        vector<uint32_t> cb_formats(n_cbs);
        vector<uint32_t> cb_tile_bytes(n_cbs);
        for (uint32_t i = 0; i < n_cbs; i++) {
          cb_indices[i] = extract_u32(line, "cb_idx" + to_string(i));
          cb_formats[i] = extract_u32(line, "cb_fmt" + to_string(i));
          cb_tile_bytes[i] = extract_u32(line, "cb_tb" + to_string(i));
        }

        // Read reader + compute + writer sources sent as raw bytes after JSON line
        uint32_t reader_source_len = extract_u32(line, "reader_source_len");
        string reader_source(reader_source_len, '\0');
        cin.read(&reader_source[0], reader_source_len);
        uint32_t compute_source_len = extract_u32(line, "compute_source_len");
        string compute_source(compute_source_len, '\0');
        cin.read(&compute_source[0], compute_source_len);
        uint32_t writer_source_len = extract_u32(line, "writer_source_len");
        string writer_source(writer_source_len, '\0');
        cin.read(&writer_source[0], writer_source_len);

        ProgramConfig cfg;
        cfg.reader_source = reader_source;
        cfg.compute_source = compute_source;
        cfg.writer_source = writer_source;
        cfg.cb_indices = cb_indices;
        cfg.cb_formats = cb_formats;
        cfg.cb_tile_bytes = cb_tile_bytes;

        if (id >= program_cache.size()) {
          program_cache.resize(id + 1);
        }
        program_cache[id] = cfg;

        cout << R"({"status":"ok"})" << endl;

      } catch (const exception &e) {
        cerr << "compile_program error: " << e.what() << endl;
        cout << R"({"status":"error","msg":")" << e.what() << R"("})" << endl;
      }
    }

    // ---- run ----
    else if (cmd == "run") {
      if (!mesh_device.get()) {
        cout << R"({"status":"error","msg":"not initialized"})" << endl;
        continue;
      }

      uint32_t id = extract_u32(line, "id");

      // Parse buffer indices: src0, src1, ..., dst0, dst1, ...
      vector<uint32_t> src_indices;
      for (uint32_t i = 0;; i++) {
        string key = "src" + to_string(i);
        auto k = line.find("\"" + key + "\"");
        if (k == string::npos)
          break;
        src_indices.push_back(extract_u32(line, key));
      }
      vector<uint32_t> dst_indices;
      for (uint32_t i = 0;; i++) {
        string key = "dst" + to_string(i);
        auto k = line.find("\"" + key + "\"");
        if (k == string::npos)
          break;
        dst_indices.push_back(extract_u32(line, key));
      }
      uint32_t n_inputs = src_indices.size();
      uint32_t n_outputs = dst_indices.size();

      // Validate indices
      for (uint32_t i = 0; i < n_inputs; i++) {
        if (src_indices[i] >= buffers.size() || !buffers[src_indices[i]]) {
          cout << R"({"status":"error","msg":"run: invalid src index )"
               << src_indices[i] << R"("})" << endl;
          continue;
        }
      }
      for (uint32_t i = 0; i < n_outputs; i++) {
        if (dst_indices[i] >= buffers.size() || !buffers[dst_indices[i]]) {
          cout << R"({"status":"error","msg":"run: invalid dst index )"
               << dst_indices[i] << R"("})" << endl;
          continue;
        }
      }

      // Look up cached program config by sequential ID
      if (id >= program_cache.size()) {
        cout << R"({"status":"error","msg":"program not found for id )" << id
             << R"("})" << endl;
        continue;
      }
      auto &cfg = program_cache[id];

      try {
        Program program = CreateProgram();
        CoreCoord core = {0, 0};
        MeshWorkload workload;
        MeshCoordinateRange device_range(mesh_device->shape());

        // Circular buffers from cached config
        constexpr uint32_t tiles_per_cb = 2;
        auto mk_cb = [&](CBIndex idx, uint32_t fmt, uint32_t tb) {
          DataFormat df;
          switch (fmt) {
          case 0:
            df = DataFormat::Float32;
            break;
          case 1:
            df = DataFormat::Float16;
            break;
          case 2:
            df = DataFormat::Float16_b;
            break;
          default:
            throw runtime_error("unsupported data_format " + to_string(fmt));
            break;
          }
          CreateCircularBuffer(
              program, core,
              CircularBufferConfig(tiles_per_cb * tb, {{idx, df}})
                  .set_page_size(idx, tb));
        };
        for (uint32_t i = 0; i < cfg.cb_indices.size(); i++) {
          mk_cb(static_cast<CBIndex>(cfg.cb_indices[i]),
                cfg.cb_formats[i], cfg.cb_tile_bytes[i]);
        }

        cerr << "[TT] creating TensorAccessorArgs from buffers" << endl;
        // Build compile-time args from actual buffer pointers
        vector<uint32_t> reader_compile_args;
        for (uint32_t i = 0; i < n_inputs; i++) {
          TensorAccessorArgs(*buffers[src_indices[i]])
              .append_to(reader_compile_args);
        }
        vector<uint32_t> writer_compile_args;
        for (uint32_t i = 0; i < n_outputs; i++) {
          TensorAccessorArgs(*buffers[dst_indices[i]])
              .append_to(writer_compile_args);
        }

        cerr << "[TT] creating reader kernel" << endl;
        auto reader = CreateKernelFromString(
            program, cfg.reader_source, core,
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
        cerr << "[TT] creating writer kernel" << endl;
        auto writer = CreateKernelFromString(
            program, cfg.writer_source, core,
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
        cerr << "[TT] creating compute kernel" << endl;
        auto compute = CreateKernelFromString(
            program, cfg.compute_source, core,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .dst_full_sync_en = false,
                .unpack_to_dest_mode = {},
                .bfp8_pack_precise = false,
                .math_approx_mode = false,
                .compile_args = {},
                .defines = {},
                .named_compile_args = {},
                .opt_level = KernelBuildOptLevel::O3,
                .compiler_include_paths = {},
            });

        // Set runtime args — buffer addresses + core index
        cerr << "[TT] setting rt args" << endl;
        {
          vector<uint32_t> reader_rt_args;
          for (uint32_t i = 0; i < n_inputs; i++) {
            uint64_t a = buffers[src_indices[i]]->address();
            cerr << "[TT]  src" << i << " idx=" << src_indices[i]
                 << " addr=" << a << " sz=" << buffers[src_indices[i]]->size()
                 << endl;
            reader_rt_args.push_back(static_cast<uint32_t>(a));
          }
          // Core index for gidx0 — axis after buffer args
          reader_rt_args.push_back(0);
          SetRuntimeArgs(program, reader, core, reader_rt_args);
        }
        {
          vector<uint32_t> writer_rt_args;
          for (uint32_t i = 0; i < n_outputs; i++) {
            uint64_t a = buffers[dst_indices[i]]->address();
            cerr << "[TT]  dst" << i << " idx=" << dst_indices[i]
                 << " addr=" << a << " sz=" << buffers[dst_indices[i]]->size()
                 << endl;
            writer_rt_args.push_back(static_cast<uint32_t>(a));
          }
          // Core index for gidx0 — axis after buffer args
          writer_rt_args.push_back(0);
          SetRuntimeArgs(program, writer, core, writer_rt_args);
        }
        {
          SetRuntimeArgs(program, compute, core, {});
        }
        cerr << "[TT] before add_program" << endl;
        workload.add_program(device_range, std::move(program));
        cerr << "[TT] after add_program" << endl;
        cerr << "[TT] before EnqueueMeshWorkload" << endl;
        EnqueueMeshWorkload(*cq, workload, false);
        cerr << "[TT] after EnqueueMeshWorkload" << endl;
        cerr << "[TT] before Finish" << endl;
        Finish(*cq);
        cerr << "[TT] after Finish" << endl;

        cout << R"({"status":"ok"})" << endl;

      } catch (const exception &e) {
        cerr << "run error: " << e.what() << endl;
        cout << R"({"status":"error","msg":")" << e.what() << R"("})" << endl;
      }
    }

    // ---- exit ----
    else if (cmd == "exit") {
      buffers.clear();
      if (mesh_device.get())
        mesh_device->close();
      cout << R"({"status":"bye"})" << endl;
      break;
    }

    else {
      cout << R"({"status":"error","msg":"unknown cmd: )" << cmd << R"("})"
           << endl;
    }
  }

  return 0;
}
