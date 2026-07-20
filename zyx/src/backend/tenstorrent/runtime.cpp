// SPDX-FileCopyrightText: (c) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
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

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

int main() {
    if (!getenv("TT_METAL_RUNTIME_ROOT")) {
        setenv("TT_METAL_RUNTIME_ROOT", TT_METAL_ROOT_DEFAULT, 0);
    }

    constexpr uint32_t n_tiles = 1;
    constexpr uint32_t tile_elems = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    constexpr uint32_t tile_bytes = sizeof(bfloat16) * tile_elems;
    constexpr uint32_t buf_size = n_tiles * tile_bytes;

    try {
        auto mesh_device = MeshDevice::create_unit_mesh(0);
        MeshCommandQueue& cq = mesh_device->mesh_command_queue();

        DeviceLocalBufferConfig dram_config{};
        dram_config.page_size = tile_bytes;
        dram_config.buffer_type = BufferType::DRAM;
        ReplicatedBufferConfig buf_config{.size = buf_size};

        auto src0 = MeshBuffer::create(buf_config, dram_config, mesh_device.get());
        auto src1 = MeshBuffer::create(buf_config, dram_config, mesh_device.get());
        auto dst  = MeshBuffer::create(buf_config, dram_config, mesh_device.get());

        vector<bfloat16> a_data(tile_elems * n_tiles, bfloat16(1.0f));
        vector<bfloat16> b_data(tile_elems * n_tiles, bfloat16(2.0f));

        EnqueueWriteMeshBuffer(cq, src0, a_data, false);
        EnqueueWriteMeshBuffer(cq, src1, b_data, false);

        Program program = CreateProgram();
        CoreCoord core = {0, 0};
        MeshWorkload workload;
        MeshCoordinateRange device_range(mesh_device->shape());

        constexpr uint32_t tiles_per_cb = 2;
        auto cb = [&](CBIndex idx) {
            CreateCircularBuffer(program, core,
                CircularBufferConfig(tiles_per_cb * tile_bytes, {{idx, DataFormat::Float16_b}})
                    .set_page_size(idx, tile_bytes));
        };
        cb(CBIndex::c_0);
        cb(CBIndex::c_1);
        cb(CBIndex::c_16);

        string kdir = KERNEL_DIR;

        vector<uint32_t> reader_args;
        TensorAccessorArgs(*src0).append_to(reader_args);
        TensorAccessorArgs(*src1).append_to(reader_args);
        auto reader = CreateKernel(program, kdir + "/read_tiles.cpp", core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_args});

        vector<uint32_t> writer_args;
        TensorAccessorArgs(*dst).append_to(writer_args);
        auto writer = CreateKernel(program, kdir + "/write_tile.cpp", core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = writer_args});

        auto compute = CreateKernel(program, kdir + "/tiles_add.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

        SetRuntimeArgs(program, reader, core, {(uint32_t)src0->address(), (uint32_t)src1->address(), n_tiles});
        SetRuntimeArgs(program, writer, core, {(uint32_t)dst->address(), n_tiles});
        SetRuntimeArgs(program, compute, core, {n_tiles});

        workload.add_program(device_range, move(program));
        EnqueueMeshWorkload(cq, workload, false);
        Finish(cq);

        vector<bfloat16> result;
        EnqueueReadMeshBuffer(cq, result, dst, true);

        cout << "Add result (1.0 + 2.0)" << endl;
        for (int i = 0; i < min(8, (int)result.size()); i++)
            cout << "  [" << i << "] = " << (float)result[i] << endl;

        bool pass = true;
        for (size_t i = 0; i < result.size(); i++) {
            float expected = (float)a_data[i] + (float)b_data[i];
            float actual = (float)result[i];
            if (abs(expected - actual) > 0.01f) {
                cerr << "FAIL at " << i << ": expected " << expected << " got " << actual << endl;
                pass = false;
                break;
            }
        }

        mesh_device->close();

        if (pass) {
            cout << "TEST PASSED" << endl;
            return 0;
        } else {
            cerr << "TEST FAILED" << endl;
            return 1;
        }
    } catch (const exception& e) {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }
}
