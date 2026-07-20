// SPDX-FileCopyrightText: (c) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

// Combined NOC data mover: reads inputs from DRAM, then writes output to DRAM.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t n_srcs = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1 + n_srcs * 2 + 2);

    // Read dst NOC address
    uint32_t dst_noc_low = get_arg_val<uint32_t>(1 + n_srcs * 2);
    uint32_t dst_noc_high = get_arg_val<uint32_t>(1 + n_srcs * 2 + 1);
    uint64_t dst_noc_addr = (uint64_t)dst_noc_high << 32 | dst_noc_low;

    // Phase 1: Read all input tiles
    for (uint32_t s = 0; s < n_srcs; s++) {
        uint32_t src_noc_low = get_arg_val<uint32_t>(1 + s * 2);
        uint32_t src_noc_high = get_arg_val<uint32_t>(1 + s * 2 + 1);
        uint64_t src_noc_addr = (uint64_t)src_noc_high << 32 | src_noc_low;
        uint32_t cb_id = tt::CBIndex::c_0 + s;
        uint32_t tile_bytes = get_tile_size(cb_id);

        for (uint32_t i = 0; i < n_tiles; i++) {
            cb_reserve_back(cb_id, 1);
            uint32_t l1_addr = get_write_ptr(cb_id);
            uint64_t noc_addr = src_noc_addr + i * tile_bytes;
            noc_async_read(noc_addr, l1_addr, tile_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_id, 1);
        }
    }

    // Phase 2: Wait for compute to finish, then write output
    // Compute fills CB c_16 with results
    constexpr uint32_t cb_id = tt::CBIndex::c_16;
    uint32_t tile_bytes = get_tile_size(cb_id);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_addr = get_read_ptr(cb_id);
        uint64_t noc_addr = dst_noc_addr + i * tile_bytes;
        noc_async_write(l1_addr, noc_addr, tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    }
}
