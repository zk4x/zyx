// SPDX-FileCopyrightText: (c) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

// Long-lived C++ runtime process for zyx Tenstorrent backend.
// Reads JSON commands from stdin, manages persistent DRAM buffers,
// executes kernels on tt-metal hardware, writes JSON responses to stdout.
//
// Tensor data is transferred via temporary shared memory regions
// (shm_open + unlink per transfer), created by the Rust side.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <tt-metalium/kernel_types.hpp>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#if __has_include(<umd/device/utils/mmio_timeout_config.hpp>)
#define ZYX_HAS_MMIO_TIMEOUT_CONFIG 1
#include <umd/device/utils/mmio_timeout_config.hpp>
#endif

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

// Parse `"key":[1,2,3]` into a vector of u32.
static vector<uint32_t> extract_u32_vec(const string &json, const string &key) {
  vector<uint32_t> out;
  auto k = json.find("\"" + key + "\"");
  if (k == string::npos)
    return {};
  auto start = json.find('[', k);
  if (start == string::npos)
    return {};
  auto end = json.find(']', start);
  if (end == string::npos)
    return {};
  string body = json.substr(start + 1, end - start - 1);
  size_t pos = 0;
  while (pos < body.size()) {
    auto next = body.find(',', pos);
    if (next == string::npos)
      next = body.size();
    string tok = trim(body.substr(pos, next - pos));
    if (!tok.empty())
      out.push_back((uint32_t)stoul(tok));
    pos = next + 1;
  }
  return {};
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
  // Total kernel param count (Global + Variable + GlobalMut, head order).
  uint32_t n_params = 0;
  // Head-order param ordinals each section needs as runtime args
  // (transitive closure of that section's scalar deps, sorted).
  vector<uint32_t> reader_params;
  vector<uint32_t> compute_params;
  vector<uint32_t> writer_params;
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

        DeviceLocalBufferConfig dram_config{
            .page_size = PAGE_SIZE,
            .buffer_type = BufferType::DRAM
        };
        ReplicatedBufferConfig buf_config{.size = (uint32_t)n_pages * PAGE_SIZE};

        auto buf = MeshBuffer::create(buf_config, dram_config, mesh_device.get());
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

        uint32_t n_params = extract_u32(line, "n_params");
        uint32_t n_reader_params = extract_u32(line, "n_reader_params");
        uint32_t n_compute_params = extract_u32(line, "n_compute_params");
        uint32_t n_writer_params = extract_u32(line, "n_writer_params");
        vector<uint32_t> reader_params(n_reader_params);
        vector<uint32_t> compute_params(n_compute_params);
        vector<uint32_t> writer_params(n_writer_params);
        for (uint32_t i = 0; i < n_reader_params; i++) {
          reader_params[i] = extract_u32(line, "rp" + to_string(i));
        }
        for (uint32_t i = 0; i < n_compute_params; i++) {
          compute_params[i] = extract_u32(line, "cp" + to_string(i));
        }
        for (uint32_t i = 0; i < n_writer_params; i++) {
          writer_params[i] = extract_u32(line, "wp" + to_string(i));
        }
        for (uint32_t p : reader_params)
          if (p >= n_params)
            throw runtime_error("reader param ordinal " + to_string(p) +
                                " >= n_params " + to_string(n_params));
        for (uint32_t p : compute_params)
          if (p >= n_params)
            throw runtime_error("compute param ordinal " + to_string(p) +
                                " >= n_params " + to_string(n_params));
        for (uint32_t p : writer_params)
          if (p >= n_params)
            throw runtime_error("writer param ordinal " + to_string(p) +
                                " >= n_params " + to_string(n_params));

        // Read reader + compute + writer sources sent as raw bytes after JSON
        // line
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
        cfg.n_params = n_params;
        cfg.reader_params = reader_params;
        cfg.compute_params = compute_params;
        cfg.writer_params = writer_params;

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

      // Variable param values: vord{i} -> vval{i}. Buffer params (src/dst)
      // are NOT in this map; their addresses come from the buffers.
      unordered_map<uint32_t, uint32_t> vars;
      uint32_t n_vars = extract_u32(line, "n_vars");
      for (uint32_t i = 0; i < n_vars; i++) {
        vars[extract_u32(line, "vord" + to_string(i))] =
            extract_u32(line, "vval" + to_string(i));
      }

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

      uint32_t gidx0_sz = extract_u32(line, "gd0");
      uint32_t gidx1_sz = extract_u32(line, "gd1");

      // GlobalMut params occupy the tail of the head-order param list
      // (n_inputs .. n_params-1); Global params are src_indices[i], Variable
      // params are in `vars`. Per-section runtime args (reader, compute,
      // writer each get ONLY the params they need): the section's Global +
      // Variable params interleaved in head order, then its GlobalMut
      // params, then gidx0 (row) and gidx1 (col) — the core's coordinates
      // in the tensix grid.
      uint32_t n_params = cfg.n_params;

      // TensorAccessorArgs compile args for ONE section: buffer params of the
      // section in ascending ordinal order (Variable params get no accessor).
      auto section_compile_args = [&](const vector<uint32_t> &params) {
        vector<uint32_t> args;
        uint32_t next_src = 0, next_dst = 0;
        for (uint32_t p : params) {
          if (vars.count(p))
            continue;
          const shared_ptr<MeshBuffer> &b =
              (p >= n_inputs) ? buffers[dst_indices[next_dst++]]
                              : buffers[src_indices[next_src++]];
          TensorAccessorArgs(*b).append_to(args);
        }
        return args;
      };

      // Runtime args for ONE section: exactly the params this section needs,
      // in its section list order (Global|Variable interleaved first, then
      // GlobalMut — the lists are sorted by head ordinal and GlobalMut
      // occupies the tail of the head-order param list), followed by the
      // core's coordinates in the tensix grid: gidx0 = row, gidx1 = col
      // (different in each core).
      auto section_rt_args = [&](const vector<uint32_t> &params, uint32_t row,
                                 uint32_t col) {
        vector<uint32_t> rt;
        uint32_t next_src = 0, next_dst = 0;
        for (uint32_t p : params) {
          auto vit = vars.find(p);
          if (vit != vars.end()) {
            rt.push_back(vit->second);
          } else if (p >= n_inputs) {
            uint64_t a = buffers[dst_indices[next_dst++]]->address();
            rt.push_back(static_cast<uint32_t>(a));
          } else {
            uint64_t a = buffers[src_indices[next_src++]]->address();
            rt.push_back(static_cast<uint32_t>(a));
          }
        }
        rt.push_back(row);
        rt.push_back(col);
        return rt;
      };

      try {
        Program program = CreateProgram();
        MeshWorkload workload;
        MeshCoordinateRange device_range(mesh_device->shape());

        vector<uint32_t> reader_compile_args =
            section_compile_args(cfg.reader_params);
        vector<uint32_t> writer_compile_args =
            section_compile_args(cfg.writer_params);

        // Circular buffers from cached config
        constexpr uint32_t tiles_per_cb = 2;

        // Build CoreRangeSet covering all cores
        CoreCoord start_core{0, 0};
        CoreCoord end_core{gidx1_sz - 1, gidx0_sz - 1};
        CoreRangeSet all_cores(CoreRange(start_core, end_core));

        // Create CBs on all cores at once (standard TT pattern)
        for (uint32_t i = 0; i < cfg.cb_indices.size(); i++) {
          DataFormat df;
          switch (cfg.cb_formats[i]) {
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
            throw runtime_error("unsupported data_format " +
                                to_string(cfg.cb_formats[i]));
          }
          CreateCircularBuffer(
              program, all_cores,
              CircularBufferConfig(
                  tiles_per_cb * cfg.cb_tile_bytes[i],
                  {{static_cast<CBIndex>(cfg.cb_indices[i]), df}})
                  .set_page_size(static_cast<CBIndex>(cfg.cb_indices[i]),
                                 cfg.cb_tile_bytes[i]));
        }

        // Create ONE reader kernel on all cores (standard TT SPMD pattern)
        cerr << "[TT] creating reader kernel on cores [0,0.." << (gidx1_sz - 1)
             << "," << (gidx0_sz - 1) << "]" << endl;
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

        // Create ONE writer kernel on all cores
        cerr << "[TT] creating writer kernel on cores [0,0.." << (gidx1_sz - 1)
             << "," << (gidx0_sz - 1) << "]" << endl;
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

        // Create ONE compute kernel on all cores
        cerr << "[TT] creating compute kernel on cores [0,0.." << (gidx1_sz - 1)
             << "," << (gidx0_sz - 1) << "]" << endl;
        auto compute =
            CreateKernelFromString(program, cfg.compute_source, all_cores,
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

        // Set per-core runtime args using the single kernel handle
        for (uint32_t row = 0; row < gidx0_sz; row++) {
          for (uint32_t col = 0; col < gidx1_sz; col++) {
            CoreCoord core = {col, row};
            SetRuntimeArgs(program, reader, core,
                           section_rt_args(cfg.reader_params, row, col));
            SetRuntimeArgs(program, writer, core,
                           section_rt_args(cfg.writer_params, row, col));
            SetRuntimeArgs(program, compute, core,
                           section_rt_args(cfg.compute_params, row, col));
          }
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
