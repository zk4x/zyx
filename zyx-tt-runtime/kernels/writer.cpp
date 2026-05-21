// SPDX-FileCopyrightText: (c) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_noc_low = get_arg_val<uint32_t>(0);
    uint32_t dst_noc_high = get_arg_val<uint32_t>(1);
    uint32_t n_tiles = get_arg_val<uint32_t>(2);

    uint64_t dst_noc_addr = (uint64_t)dst_noc_high << 32 | dst_noc_low;
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
