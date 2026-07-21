// SPDX-FileCopyrightText: (c) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);

    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto args_c = TensorAccessorArgs<0>();
    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto c = TensorAccessor(args_c, dst_addr, tile_bytes);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_addr = get_read_ptr(cb_out);
        noc_async_write_tile(i, c, l1_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
