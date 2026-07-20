// SPDX-FileCopyrightText: (c) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t n_srcs = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1 + n_srcs * 2); // after all (low, high) pairs

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
}
