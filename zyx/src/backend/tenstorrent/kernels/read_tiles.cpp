// SPDX-FileCopyrightText: (c) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t n_tiles = get_arg_val<uint32_t>(2);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto in0_args = TensorAccessorArgs<0>();
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const uint32_t src0_tile_bytes = get_tile_size(cb_in0);
    const uint32_t src1_tile_bytes = get_tile_size(cb_in1);
    const auto a = TensorAccessor(in0_args, src0_addr, src0_tile_bytes);
    const auto b = TensorAccessor(in1_args, src1_addr, src1_tile_bytes);

    constexpr uint32_t cb_intermed0 = tt::CBIndex::c_24;
    constexpr uint32_t cb_intermed1 = tt::CBIndex::c_25;

    for (uint32_t i = 0; i < n_tiles; i++) {
        // Read src0 into cb_in0
        cb_reserve_back(cb_in0, 1);
        uint32_t l1_addr0 = get_write_ptr(cb_in0);
        noc_async_read_tile(i, a, l1_addr0);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);

        // Read src1 into cb_in1
        cb_reserve_back(cb_in1, 1);
        uint32_t l1_addr1 = get_write_ptr(cb_in1);
        noc_async_read_tile(i, b, l1_addr1);
        noc_async_read_barrier();
        cb_push_back(cb_in1, 1);
    }
}
